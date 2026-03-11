"""Tests for OutputGuard — PII detection and content policy enforcement."""

from __future__ import annotations

import pytest

from pinocchio.utils.output_guard import OutputGuard, _luhn_check


# =====================================================================
# Luhn validation
# =====================================================================

class TestLuhnCheck:
    """Direct tests for the Luhn check function."""

    def test_valid_visa(self):
        assert _luhn_check("4111111111111111") is True

    def test_valid_mastercard(self):
        assert _luhn_check("5500000000000004") is True

    def test_invalid_number(self):
        assert _luhn_check("5111111111111111") is False

    def test_too_short(self):
        assert _luhn_check("1234") is False


# =====================================================================
# Credit card detection (Luhn + format variants)
# =====================================================================

class TestCreditCardDetection:
    """Credit card detection should use Luhn validation and accept
    numbers with spaces or dashes."""

    def test_rejects_invalid_luhn(self):
        guard = OutputGuard()
        result = guard.check("Card: 5111111111111111")
        assert "credit_card" not in result.pii_found

    def test_accepts_valid_luhn(self):
        guard = OutputGuard()
        result = guard.check("Card: 4111111111111111")
        assert "credit_card" in result.pii_found
        assert "[CREDIT_CARD]" in result.masked_text

    def test_with_spaces(self):
        guard = OutputGuard()
        result = guard.check("Card: 4111 1111 1111 1111")
        assert "credit_card" in result.pii_found

    def test_with_dashes(self):
        guard = OutputGuard()
        result = guard.check("Card: 4111-1111-1111-1111")
        assert "credit_card" in result.pii_found


# =====================================================================
# IPv6 detection
# =====================================================================

class TestIPv6Detection:
    """IPv6 addresses (full and compressed) should be detected."""

    def test_full_ipv6(self):
        guard = OutputGuard()
        result = guard.check("Server at 2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert "ip_address" in result.pii_found
        assert "[IP_ADDRESS]" in result.masked_text

    def test_compressed_ipv6(self):
        guard = OutputGuard()
        result = guard.check("Connect to ::1 for localhost")
        assert "ip_address" in result.pii_found


# =====================================================================
# Obfuscated email & SSN
# =====================================================================

class TestObfuscatedPII:
    """Obfuscated emails and context-dependent SSN detection."""

    def test_obfuscated_email(self):
        guard = OutputGuard()
        result = guard.check("Contact john at example dot com")
        assert "email" in result.pii_found

    def test_ssn_requires_context(self):
        guard = OutputGuard()
        result = guard.check("The code is 123-45-6789.")
        assert "ssn" not in result.pii_found

    def test_ssn_with_context(self):
        guard = OutputGuard()
        result = guard.check("SSN: 123-45-6789")
        assert "ssn" in result.pii_found
        assert "[SSN]" in result.masked_text


# =====================================================================
# End-to-end mixed PII masking
# =====================================================================

class TestOutputGuardEndToEnd:
    """End-to-end output guard with mixed PII types."""

    def test_mixed_pii_masking(self):
        guard = OutputGuard()
        text = (
            "Contact john@example.com, call +1-555-123-4567, "
            "card 4111111111111111, server at 192.168.1.100"
        )
        result = guard.check(text)
        assert "email" in result.pii_found
        assert "phone" in result.pii_found
        assert "credit_card" in result.pii_found
        assert "ip_address" in result.pii_found
        assert "[EMAIL]" in result.masked_text
        assert "[PHONE]" in result.masked_text
        assert "[CREDIT_CARD]" in result.masked_text
        assert "[IP_ADDRESS]" in result.masked_text

    def test_clean_text_no_false_positives(self):
        guard = OutputGuard()
        result = guard.check("The temperature is 72 degrees and the year is 2024.")
        assert result.pii_found == []
        assert result.is_safe is True
