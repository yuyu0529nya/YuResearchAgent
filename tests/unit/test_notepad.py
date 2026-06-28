"""
tests/unit/test_notepad.py
src/tools/notepad.py 的单元测试。

按文件路径加载并注册到 sys.modules（@dataclass 在 Python 3.14 下需要模块
可在 sys.modules 中查到），从而无需安装第三方库即可运行。
"""
import asyncio
import importlib.util
import os
import sys

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "tools", "notepad.py",
)
_spec = importlib.util.spec_from_file_location("notepad_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
NotepadTool = _mod.NotepadTool


def _run(coro):
    return asyncio.run(coro)


def test_docstrings_present():
    # 回归：旧实现把 import asyncio 写在 docstring 前面，导致 __doc__ 为 None
    assert NotepadTool.write.__doc__ is not None
    assert NotepadTool.read.__doc__ is not None
    assert NotepadTool.search.__doc__ is not None


def test_write_then_read():
    np = NotepadTool()
    _run(np.write(content="结论 A", category="conclusion"))
    assert "结论 A" in _run(np.read())


def test_read_category_filter():
    np = NotepadTool()
    _run(np.write(content="待办 X", category="todo"))
    _run(np.write(content="结论 Y", category="conclusion"))
    todo_view = _run(np.read(category="todo"))
    assert "待办 X" in todo_view
    assert "结论 Y" not in todo_view


def test_search_hit_and_miss():
    np = NotepadTool()
    _run(np.write(content="营收 300 亿", category="source"))
    assert "营收" in _run(np.search(keyword="营收"))
    assert "No notes matching" in _run(np.search(keyword="不存在的词"))


def test_execute_unknown_action():
    assert "Unknown action" in _run(NotepadTool().execute("frobnicate"))


def test_execute_bad_args_returns_error_not_crash():
    # 回归：execute('write') 缺 content，旧实现会抛 TypeError
    assert "Invalid arguments" in _run(NotepadTool().execute("write"))


def test_clear_all():
    np = NotepadTool()
    _run(np.write(content="a"))
    _run(np.clear())
    assert "No notes" in _run(np.read())
