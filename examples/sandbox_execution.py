"""代码沙箱安全执行示例。

演示在隔离子进程中安全执行 Python 代码。
# 离线可运行
"""

from pinocchio.sandbox import CodeSandbox


def basic_execution():
    """基本代码执行。"""
    sandbox = CodeSandbox(timeout=10)

    print("=== 基本执行 ===")
    result = sandbox.execute("print(sum(range(100)))")
    print(f"stdout: {result.stdout.strip()}")
    print(f"success: {result.success}")
    print()


def multi_line_code():
    """多行代码。"""
    sandbox = CodeSandbox(timeout=10)

    code = """
import math

primes = []
for n in range(2, 50):
    if all(n % d != 0 for d in range(2, int(math.sqrt(n)) + 1)):
        primes.append(n)

print(f"Primes under 50: {primes}")
print(f"Count: {len(primes)}")
"""

    print("=== 多行代码 ===")
    result = sandbox.execute(code)
    print(result.stdout)


def blocked_operations():
    """被阻止的危险操作。"""
    sandbox = CodeSandbox(timeout=5)

    print("=== 安全拦截 ===")

    # 1. 阻止危险模块导入
    result = sandbox.execute("import os; print(os.listdir('/'))")
    print(f"import os: success={result.success}, error={result.error[:80]}")

    # 2. 阻止 eval()
    result = sandbox.execute("eval('1+1')")
    print(f"eval(): success={result.success}, error={result.error[:80]}")

    # 3. 阻止文件写入
    result = sandbox.execute("open('/tmp/test.txt', 'w').write('hack')")
    print(f"file write: success={result.success}, error={result.error[:80]}")
    print()


def timeout_demo():
    """超时保护。"""
    sandbox = CodeSandbox(timeout=2)

    print("=== 超时保护 ===")
    result = sandbox.execute("import time; time.sleep(10)")
    print(f"timed_out: {result.timed_out}")
    print(f"error: {result.error}")
    print()


def result_serialization():
    """结果序列化。"""
    sandbox = CodeSandbox()
    result = sandbox.execute("print('hello')")

    print("=== 结果序列化 ===")
    import json
    print(json.dumps(result.to_dict(), indent=2))
    print()


if __name__ == "__main__":
    basic_execution()
    multi_line_code()
    blocked_operations()
    timeout_demo()
    result_serialization()
