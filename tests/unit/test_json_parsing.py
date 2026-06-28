"""
tests/unit/test_json_parsing.py
================================================================================
src/utils/json_parsing.py 的单元测试。

覆盖统一 JSON 解析器的多层 fallback 策略，并量化其在畸形 LLM 输出上的
恢复率（rate），为"多层 fallback 鲁棒性"提供可复现的真实证据。
纯逻辑测试，无需 API Key 或重型依赖。
================================================================================
"""
import json

import pytest

from src.utils.json_parsing import (
    extract_json,
    extract_json_object,
    JSONExtractionError,
)


# ---------------------------------------------------------------------------
# 第 1 层：直接解析
# ---------------------------------------------------------------------------
def test_clean_object():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_clean_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_unicode_chinese():
    assert extract_json('{"标题": "深度研究", "n": 35}') == {"标题": "深度研究", "n": 35}


# ---------------------------------------------------------------------------
# 第 2 层：markdown 围栏代码块
# ---------------------------------------------------------------------------
def test_json_fenced_block():
    raw = '这是分析结果：\n```json\n{"score": 8.5}\n```\n以上。'
    assert extract_json(raw) == {"score": 8.5}


def test_plain_fenced_block():
    raw = "```\n{\"k\": [1, 2]}\n```"
    assert extract_json(raw) == {"k": [1, 2]}


# ---------------------------------------------------------------------------
# 第 3 层：平衡括号提取（前后有噪声文字）
# ---------------------------------------------------------------------------
def test_prose_around_object():
    raw = 'Sure! Here is the plan: {"sub_tasks": [{"id": "t1"}]} Hope it helps.'
    assert extract_json(raw) == {"sub_tasks": [{"id": "t1"}]}


def test_braces_inside_string_not_confused():
    # 字符串内的 } 不应提前结束对象
    raw = 'prefix {"expr": "f(x) = {a, b}", "ok": true} suffix'
    assert extract_json(raw) == {"expr": "f(x) = {a, b}", "ok": True}


def test_first_of_multiple_objects():
    raw = '{"first": 1}\n{"second": 2}'
    assert extract_json(raw) == {"first": 1}


def test_url_with_hash_not_stripped_as_comment():
    # # 出现在字符串内（URL fragment），不应被当作注释删除
    raw = '{"url": "http://example.com/page#section"}'
    assert extract_json(raw) == {"url": "http://example.com/page#section"}


# ---------------------------------------------------------------------------
# 第 4 层：轻量修复（尾随逗号 / 注释）
# ---------------------------------------------------------------------------
def test_trailing_comma_object():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_trailing_comma_array():
    assert extract_json('{"items": [1, 2, 3,]}') == {"items": [1, 2, 3]}


def test_line_comment_slash():
    raw = '{\n  "a": 1, // 第一个\n  "b": 2\n}'
    assert extract_json(raw) == {"a": 1, "b": 2}


def test_line_comment_hash():
    raw = '{\n  "a": 1,  # note\n  "b": 2\n}'
    assert extract_json(raw) == {"a": 1, "b": 2}


def test_fenced_with_trailing_comma():
    raw = '```json\n{"x": [1, 2,],}\n```'
    assert extract_json(raw) == {"x": [1, 2]}


# ---------------------------------------------------------------------------
# 失败与边界
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", None, "not json at all", "你好，世界"])
def test_unrecoverable_returns_none(bad):
    assert extract_json(bad) is None


def test_falsy_valid_json_distinguished_from_failure():
    # 合法但 falsy 的值应原样返回，而非被当作失败
    assert extract_json("false") is False
    assert extract_json("0") == 0
    assert extract_json('""') == ""


# ---------------------------------------------------------------------------
# extract_json_object：仅接受 dict
# ---------------------------------------------------------------------------
def test_object_helper_returns_dict():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_object_helper_rejects_array():
    assert extract_json_object("[1, 2, 3]") is None


def test_object_helper_required_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object("garbage", required=True)


# ---------------------------------------------------------------------------
# 鲁棒性指标：量化多层 fallback 在畸形样本上的恢复率
# ---------------------------------------------------------------------------
# 一批"语义合法、格式畸形"的样本，期望目标对象均为 {"ok": true}
_MALFORMED_RECOVERABLE = [
    '{"ok": true}',                                  # 干净
    '```json\n{"ok": true}\n```',                    # json 围栏
    '```\n{"ok": true}\n```',                        # 无语言围栏
    'Here you go: {"ok": true}',                     # 前缀噪声
    '{"ok": true}  // done',                         # 行尾注释
    '{"ok": true,}',                                 # 尾随逗号
    '答案如下：\n```json\n{"ok": true,}\n```\n完毕',   # 围栏+尾随逗号+中文噪声
    '{\n  "ok": true  # yes\n}',                      # # 注释
    'prefix {"ok": true} {"ok": false}',             # 取第一个
]


def test_fallback_recovery_rate():
    """多层 fallback 应恢复全部"语义合法、格式畸形"的样本。"""
    recovered = sum(1 for s in _MALFORMED_RECOVERABLE if extract_json_object(s) == {"ok": True})
    total = len(_MALFORMED_RECOVERABLE)
    rate = recovered / total
    # 报告恢复率，便于在 CI 输出中直接看到
    print(f"\n[robustness] fallback 恢复率: {recovered}/{total} = {rate:.0%}")
    assert recovered == total, f"仅恢复 {recovered}/{total}"


def test_direct_parse_baseline_is_lower():
    """对照：仅用 json.loads 的朴素基线恢复率应明显低于多层 fallback。

    这量化了多层 fallback 相对朴素解析的增益（简历中"鲁棒性提升"的依据）。
    """
    def naive(s):
        try:
            return json.loads(s)
        except Exception:
            return None

    naive_ok = sum(1 for s in _MALFORMED_RECOVERABLE if naive(s) == {"ok": True})
    robust_ok = sum(1 for s in _MALFORMED_RECOVERABLE if extract_json_object(s) == {"ok": True})
    print(f"\n[robustness] 朴素 json.loads: {naive_ok}/{len(_MALFORMED_RECOVERABLE)}"
          f" → 多层 fallback: {robust_ok}/{len(_MALFORMED_RECOVERABLE)}")
    assert robust_ok > naive_ok
