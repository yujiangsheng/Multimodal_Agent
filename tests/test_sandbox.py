"""Tests for Gap 2: Code sandbox (CodeSandbox)."""

from __future__ import annotations

import pytest

from pinocchio.sandbox.code_sandbox import CodeSandbox, ExecutionResult


class TestExecutionResult:
    def test_defaults(self):
        r = ExecutionResult()
        assert r.success is True  # default is True
        assert r.stdout == ""
        assert r.error == ""

    def test_to_dict(self):
        r = ExecutionResult(success=True, stdout="hello", exit_code=0)
        d = r.to_dict()
        assert d["success"] is True
        assert d["stdout"] == "hello"


class TestCodeSandbox:
    def test_simple_execution(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("print('hello world')")
        assert result.success is True
        assert "hello world" in result.stdout

    def test_return_value(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("x = 2 + 3\nprint(x)")
        assert result.success is True
        assert "5" in result.stdout

    def test_syntax_error(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("def f(\n")
        assert result.success is False

    def test_runtime_error(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("1 / 0")
        assert result.success is False

    def test_blocked_import_os(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import os\nos.listdir('.')")
        assert result.success is False

    def test_blocked_import_subprocess(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import subprocess")
        assert result.success is False

    def test_timeout(self):
        sandbox = CodeSandbox(timeout=2)
        result = sandbox.execute("import time\ntime.sleep(10)")
        assert result.success is False
        assert result.timed_out is True

    def test_timeout_capped(self):
        sandbox = CodeSandbox(timeout=100)
        assert sandbox._timeout == 60  # capped at 60

    def test_multiline_output(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("for i in range(3): print(i)")
        assert result.success is True
        assert "0" in result.stdout
        assert "1" in result.stdout
        assert "2" in result.stdout

    def test_math_computation(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import math\nprint(math.pi)")
        assert result.success is True
        assert "3.14" in result.stdout

    def test_empty_code(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("")
        assert result.success is False
        assert "Empty" in result.error

    def test_json_module_allowed(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import json\nprint(json.dumps({'a': 1}))")
        assert result.success is True
        assert '"a"' in result.stdout

    def test_static_check_blocks_eval(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("eval('1+1')")
        assert result.success is False

    def test_static_check_blocks_exec(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("exec('print(1)')")
        assert result.success is False

    def test_static_check_blocks_dunder_import(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("__import__('os')")
        assert result.success is False

    def test_list_comprehension(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("print([x**2 for x in range(5)])")
        assert result.success is True
        assert "[0, 1, 4, 9, 16]" in result.stdout
