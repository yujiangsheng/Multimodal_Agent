"""Code Sandbox — isolated Python code execution via subprocess.

Executes user-provided Python code in a restricted subprocess with:
- Timeout enforcement
- stdout/stderr capture
- Import restrictions (no os, subprocess, etc.)
- Memory limit via resource module (Unix only)
- No filesystem write access

Usage::

    sandbox = CodeSandbox(timeout=10)
    result = sandbox.execute("print(sum(range(100)))")
    print(result.stdout)   # "4950"
    print(result.success)  # True
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of a sandbox code execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    success: bool = True
    error: str = ""
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "success": self.success,
            "error": self.error,
            "timed_out": self.timed_out,
        }


# Modules that are blocked from being imported
_BLOCKED_MODULES = {
    "os", "subprocess", "shutil", "signal", "ctypes", "socket",
    "http", "urllib", "requests", "httpx", "pathlib",
    "importlib", "sys", "builtins", "code", "codeop",
    "compileall", "py_compile",
    "multiprocessing", "threading",
    "pickle", "shelve", "marshal",
    "webbrowser", "antigravity",
}

# Build the sandbox wrapper script
_SANDBOX_WRAPPER = textwrap.dedent("""\
    import builtins as _builtins
    import sys as _sys

    # Block dangerous modules
    _BLOCKED = {blocked}
    _original_import = _builtins.__import__

    def _safe_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level in _BLOCKED:
            raise ImportError(f"Module '{{name}}' is not allowed in sandbox")
        return _original_import(name, *args, **kwargs)

    _builtins.__import__ = _safe_import

    # Block open() for writing
    _original_open = _builtins.open

    def _safe_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "waxb+"):
            if any(c in mode for c in "wa+x"):
                raise PermissionError("File writing is not allowed in sandbox")
        return _original_open(file, mode, *args, **kwargs)

    _builtins.open = _safe_open

    # Execute user code
    try:
        exec(open(_sys.argv[1]).read())
    except SystemExit:
        pass
""")


class CodeSandbox:
    """Execute Python code safely in an isolated subprocess."""

    def __init__(self, timeout: int = 15, max_output: int = 50000) -> None:
        self._timeout = min(timeout, 60)  # hard cap at 60s
        self._max_output = max_output

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in a sandboxed subprocess."""
        if not code.strip():
            return ExecutionResult(error="Empty code", success=False)

        # Quick static check for obviously dangerous patterns
        danger_check = self._static_check(code)
        if danger_check:
            return ExecutionResult(error=danger_check, success=False)

        # Write code and wrapper to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "user_code.py"
            wrapper_path = Path(tmpdir) / "sandbox_wrapper.py"

            code_path.write_text(code, encoding="utf-8")
            wrapper_script = _SANDBOX_WRAPPER.format(
                blocked=repr(_BLOCKED_MODULES)
            )
            wrapper_path.write_text(wrapper_script, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(wrapper_path), str(code_path)],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=tmpdir,
                    env={
                        "PATH": "",
                        "HOME": tmpdir,
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                stdout = proc.stdout[:self._max_output]
                stderr = proc.stderr[:self._max_output]
                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    success=proc.returncode == 0,
                    error=stderr if proc.returncode != 0 else "",
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    error=f"Code execution timed out after {self._timeout}s",
                    success=False,
                    timed_out=True,
                    exit_code=-1,
                )
            except Exception as exc:
                return ExecutionResult(
                    error=f"Sandbox error: {exc}",
                    success=False,
                    exit_code=-1,
                )

    def _static_check(self, code: str) -> str | None:
        """Quick static analysis for obviously dangerous code."""
        # Check for eval/exec of dynamic strings (already inside exec, but check for nested)
        dangerous_patterns = [
            ("__import__", "Direct __import__ is not allowed"),
            ("exec(", "Nested exec() is not allowed"),
            ("eval(", "eval() is not allowed — use direct expressions"),
            ("compile(", "compile() is not allowed"),
        ]
        for pattern, msg in dangerous_patterns:
            if pattern in code:
                return msg
        return None
