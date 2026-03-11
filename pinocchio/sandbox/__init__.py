"""Code sandbox — safe Python execution in isolated subprocess.

Runs user-provided Python code in a restricted subprocess with:

* **Timeout enforcement** — hard cap at 60 seconds.
* **Import restrictions** — blocks ``os``, ``subprocess``, ``socket``, etc.
* **No filesystem writes** — ``open()`` is patched to read-only.
* **Static analysis** — rejects ``eval()``, ``exec()``, ``__import__``.

Quick start::

    from pinocchio.sandbox import CodeSandbox

    sandbox = CodeSandbox(timeout=10)
    result = sandbox.execute("print(2 ** 128)")
    print(result.stdout)   # "340282366920938463463374607431768211456"
    print(result.success)  # True
"""

from pinocchio.sandbox.code_sandbox import CodeSandbox, ExecutionResult

__all__ = ["CodeSandbox", "ExecutionResult"]
