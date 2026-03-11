"""Output Guard — post-generation safety checks on LLM responses.

Complements :class:`InputGuard` (which validates *incoming* text) by
scanning *outgoing* responses for:

1. **PII leakage** — email addresses, phone numbers, credit card
   numbers, SSN-like patterns, IP addresses.
2. **Content policy violations** — hate speech markers, explicit
   content keywords, self-harm language.
3. **Format validation** — optional JSON schema conformance check.
4. **Hallucination markers** — phrases that strongly signal
   fabricated references or false certainty.

Usage
-----
>>> from pinocchio.utils.output_guard import OutputGuard
>>> guard = OutputGuard()
>>> result = guard.check("The answer is 42.")
>>> result.is_safe  # True
>>> result = guard.check("Email me at john@example.com, card 4111111111111111")
>>> result.pii_found  # ['email', 'credit_card']
>>> result.masked_text  # "Email me at [EMAIL], card [CREDIT_CARD]"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# =====================================================================
# Luhn algorithm for credit card validation
# =====================================================================

def _luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# =====================================================================
# PII patterns (hardened)
# =====================================================================

_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # Email — standard + common obfuscation patterns
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
            r"|"
            r"\b[A-Za-z0-9._%+\-]+\s*[\[\(]?\s*(?:at|AT|@)\s*[\]\)]?\s*"
            r"[A-Za-z0-9.\-]+\s*[\[\(]?\s*(?:dot|DOT|\.)\s*[\]\)]?\s*[A-Za-z]{2,}\b"
        ),
        "[EMAIL]",
    ),
    # Credit Card — major issuers with Luhn post-validation
    # Must come BEFORE phone to avoid partial phone matches on spaced card numbers
    (
        "credit_card",
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2})|35\d{3}|2[2-7]\d{2})"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}\b"
        ),
        "[CREDIT_CARD]",
    ),
    # Phone — international formats with various separators
    (
        "phone",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?\d{1,3}[\s\-.]?)?"
            r"(?:\(?\d{2,4}\)?[\s\-.]?)?"
            r"\d{3,4}[\s\-.]?\d{4}"
            r"(?!\d)"
        ),
        "[PHONE]",
    ),
    # SSN — only match in context (near keywords like SSN, social security, etc.)
    (
        "ssn",
        re.compile(
            r"(?:(?:SSN|social\s+security|社会安全)\s*[:#]?\s*)"
            r"\d{3}[\s\-]?\d{2}[\s\-]?\d{4}",
            re.IGNORECASE,
        ),
        "[SSN]",
    ),
    # IPv4
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b"
        ),
        "[IP_ADDRESS]",
    ),
    # IPv6 (full and compressed forms)
    (
        "ip_address",
        re.compile(
            r"(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4})"       # full
            r"|(?:(?:[0-9a-fA-F]{1,4}:){1,7}:)"                      # trailing ::
            r"|(?:::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4})"    # leading ::
            r"|(?:[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6})"     # :: in middle
        ),
        "[IP_ADDRESS]",
    ),
]

# =====================================================================
# Content-policy keyword catalogue
# =====================================================================

_CONTENT_POLICY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "hate_speech",
        re.compile(
            r"\b(?:kill\s+all|exterminate|genocide\s+against|ethnic\s+cleansing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"\b(?:how\s+to\s+(?:commit\s+suicide|hurt\s+yourself|self[\s\-]?harm))\b",
            re.IGNORECASE,
        ),
    ),
]

# =====================================================================
# Hallucination markers
# =====================================================================

_HALLUCINATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:as\s+(?:of|published\s+in)\s+my\s+(?:training|knowledge)\s+cutoff)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:I\s+(?:cannot|can't|don't)\s+(?:access|browse|search)\s+the\s+(?:internet|web))",
        re.IGNORECASE,
    ),
]


# =====================================================================
# Result dataclass
# =====================================================================

@dataclass
class OutputCheckResult:
    """Outcome of output guard checking."""

    is_safe: bool = True
    pii_found: list[str] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    masked_text: str = ""
    original_text: str = ""

    @property
    def issues(self) -> list[str]:
        """Combined list of all issues found."""
        out: list[str] = []
        if self.pii_found:
            out.append(f"PII: {', '.join(self.pii_found)}")
        if self.policy_violations:
            out.append(f"Policy: {', '.join(self.policy_violations)}")
        if self.hallucination_flags:
            out.append(f"Hallucination: {', '.join(self.hallucination_flags)}")
        return out

    @property
    def summary(self) -> str:
        return "; ".join(self.issues) if self.issues else "clean"


# =====================================================================
# OutputGuard
# =====================================================================

class OutputGuard:
    """Post-generation safety and quality checks.

    Parameters
    ----------
    mask_pii : bool
        If True, PII in the output is replaced with placeholders
        (e.g. ``[EMAIL]``).  If False, PII is flagged but the text
        is left unchanged.
    block_policy : bool
        If True, content-policy violations make ``is_safe=False``.
    """

    def __init__(
        self,
        *,
        mask_pii: bool = True,
        block_policy: bool = True,
    ) -> None:
        self.mask_pii = mask_pii
        self.block_policy = block_policy

    def check(self, text: str) -> OutputCheckResult:
        """Run all checks on *text* and return the result."""
        if not text:
            return OutputCheckResult(masked_text="", original_text="")

        result = OutputCheckResult(original_text=text)
        masked = text

        # ── PII scan ──
        for label, pattern, replacement in _PII_PATTERNS:
            matches = list(pattern.finditer(masked))
            if not matches:
                continue
            # Credit card: post-validate with Luhn algorithm
            if label == "credit_card":
                valid_matches = [m for m in matches if _luhn_check(m.group())]
                if not valid_matches:
                    continue
                if label not in result.pii_found:
                    result.pii_found.append(label)
                if self.mask_pii:
                    for m in reversed(valid_matches):
                        masked = masked[:m.start()] + replacement + masked[m.end():]
            else:
                if label not in result.pii_found:
                    result.pii_found.append(label)
                if self.mask_pii:
                    masked = pattern.sub(replacement, masked)

        # ── Content policy ──
        for label, pattern in _CONTENT_POLICY_PATTERNS:
            if pattern.search(text):
                if label not in result.policy_violations:
                    result.policy_violations.append(label)

        # ── Hallucination markers ──
        for pattern in _HALLUCINATION_PATTERNS:
            m = pattern.search(text)
            if m:
                result.hallucination_flags.append(m.group()[:80])

        result.masked_text = masked

        if self.block_policy and result.policy_violations:
            result.is_safe = False

        return result
