"""
tests/unit/test_file_reader.py
src/tools/file_reader.py 的单元测试，重点覆盖沙箱目录安全检查。

按文件路径加载；用显式参数实例化（env 已改懒加载，故无需 dotenv）。
"""
import asyncio
import importlib.util
import json
import os
import sys

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "tools", "file_reader.py",
)
_spec = importlib.util.spec_from_file_location("file_reader_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
FileReaderTool = _mod.FileReaderTool

_BIG = 10 ** 6


def _read(tool, path):
    return asyncio.run(tool.execute(str(path)))


def test_reads_file_inside_sandbox(tmp_path):
    base = tmp_path / "safe"
    base.mkdir()
    f = base / "doc.txt"
    f.write_text("hello inside", encoding="utf-8")
    tool = FileReaderTool(allowed_base_dir=str(base), max_file_size=_BIG)
    assert "hello inside" in _read(tool, f)


def test_blocks_sibling_prefix_directory_traversal(tmp_path):
    # 回归（安全）：/safe_evil 不应被 /safe 的 startswith 前缀误放行
    base = tmp_path / "safe"
    base.mkdir()
    evil = tmp_path / "safe_evil"
    evil.mkdir()
    secret = evil / "secret.txt"
    secret.write_text("LEAKED", encoding="utf-8")
    tool = FileReaderTool(allowed_base_dir=str(base), max_file_size=_BIG)
    out = _read(tool, secret)
    assert "Access denied" in out
    assert "LEAKED" not in out


def test_unsupported_extension(tmp_path):
    f = tmp_path / "a.exe"
    f.write_text("x", encoding="utf-8")
    tool = FileReaderTool(allowed_base_dir=str(tmp_path), max_file_size=_BIG)
    assert "Unsupported file type" in _read(tool, f)


def test_file_not_found(tmp_path):
    tool = FileReaderTool(allowed_base_dir=str(tmp_path), max_file_size=_BIG)
    assert "File not found" in _read(tool, tmp_path / "nope.txt")


def test_size_limit_enforced(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100, encoding="utf-8")
    tool = FileReaderTool(allowed_base_dir=str(tmp_path), max_file_size=10)
    assert "too large" in _read(tool, f).lower()


def test_json_summary(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"a": 1, "b": [1, 2, 3]}), encoding="utf-8")
    tool = FileReaderTool(allowed_base_dir=str(tmp_path), max_file_size=_BIG)
    out = _read(tool, f)
    assert "[Type: dict]" in out
