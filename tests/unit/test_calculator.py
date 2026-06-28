"""
tests/unit/test_calculator.py
src/tools/calculator.py 的单元测试。

直接按文件路径加载模块，绕开 src/tools/__init__.py（它会 eager import
web_search→aiohttp 等可选依赖），使本测试无需安装第三方库即可运行。
"""
import asyncio
import importlib.util
import os

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "tools", "calculator.py",
)
_spec = importlib.util.spec_from_file_location("calculator_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CalculatorTool = _mod.CalculatorTool


def _calc(expr: str) -> str:
    return asyncio.run(CalculatorTool().execute(expr))


def test_basic_arithmetic():
    assert _calc("2+2") == "4"
    assert _calc("(150 + 230) * 0.15") == "57.0"


def test_functions_and_lists():
    assert _calc("mean([12, 15, 18, 21])") == "16.5"
    assert _calc("max([3, 7, 2])") == "7"
    assert _calc("sqrt(16)") == "4.0"
    assert _calc("factorial(5)") == "120"


def test_preprocess_percent_thousands_cjk_parens():
    assert _calc("15%") == "0.15"
    assert _calc("1,000 + 500") == "1500"
    assert _calc("（1 + 1）") == "2"  # 中文括号归一化


def test_runs_on_python_3_12_plus():
    # 回归：ast.Num / ast.Index 在 Python 3.12+ 被移除，
    # 旧实现 isinstance(node, ast.Num) 会对任意表达式抛 AttributeError。
    assert _calc("3 * 7") == "21"


def test_division_by_zero_handled():
    assert "Division by zero" in _calc("10 / 0")


def test_pow_dos_guard():
    assert "Exponent too large" in _calc("9 ** 9 ** 9")
    assert _calc("2 ** 10") == "1024"  # 正常幂不受影响


def test_unsafe_calls_rejected():
    # 非白名单函数/名称应被拒绝（沙箱安全）
    assert "Error" in _calc("__import__('os')")
    assert "Error" in _calc("open('x')")
