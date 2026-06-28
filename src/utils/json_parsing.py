"""
src/utils/json_parsing.py
================================================================================
统一的鲁棒 JSON 解析工具。

LLM 的结构化输出经常夹带噪声：markdown 代码围栏、解释性前后缀文字、
尾随逗号、`//` / `#` 注释等。本模块提供**单一入口** :func:`extract_json`，
集中实现多层 fallback 解析策略，替代此前散落在
``planner`` / ``red_agent`` / ``blue_agent`` / ``core.judge`` /
``evolution.*`` 等模块中的重复实现，消除配置漂移与维护负担。

解析策略（按顺序尝试，命中即返回）：
    1. 直接 ``json.loads``
    2. 提取 ```json ... ``` / ``` ... ``` 围栏代码块
    3. 提取第一个**平衡**的 ``{...}`` / ``[...]`` 片段（带引号/转义保护）
    4. 轻量修复（去行内注释、去尾随逗号）后重试 1 + 3

与旧实现相比的改进：
    - 平衡括号扫描取代贪婪正则 ``\\{.*\\}``，可正确处理字符串内含
      花括号、对象后存在尾随文本、多个并列对象等情形。
    - 去注释时保护字符串字面量，避免误删 URL（``http://``）或
      颜色值（``#fff``）中的字符。
================================================================================
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract_json", "extract_json_object", "JSONExtractionError"]


class JSONExtractionError(ValueError):
    """当 ``required=True`` 且无法解析出 JSON 对象时抛出。"""


# ```json ... ``` 或 ``` ... ``` 围栏（惰性匹配，支持多个）
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
# 尾随逗号：, 后紧跟 } 或 ]
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

# 用于区分"解析失败"与"解析出合法但 falsy 的值（如 null/false/0）"
_MISS = object()


def _try_loads(s: str) -> Any:
    """尝试 ``json.loads``；失败返回哨兵 ``_MISS`` 而非 None。

    这样可正确区分 ``json.loads("null")`` 得到的合法 ``None``
    与"解析失败"两种情况。
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _MISS


def _strip_line_comments(s: str) -> str:
    """去除行内 ``//`` 与 ``#`` 注释，但保护字符串字面量内的同名字符。"""
    out: list[str] = []
    for line in s.splitlines():
        in_str = False
        quote = ""
        esc = False
        cut = len(line)
        i = 0
        while i < len(line):
            ch = line[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in "\"'":
                    in_str = True
                    quote = ch
                elif ch == "#":
                    cut = i
                    break
                elif ch == "/" and line[i + 1 : i + 2] == "/":
                    cut = i
                    break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def _find_balanced(s: str) -> str | None:
    """返回 ``s`` 中第一个平衡的 ``{...}`` 或 ``[...]`` 片段。

    扫描时跟踪字符串状态与转义，因此字符串内的 ``{``/``}``/``[``/``]``
    不会影响深度计数。找不到完整片段时返回 None。
    """
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def extract_json(text: str | None) -> Any | None:
    """从噪声文本中鲁棒提取第一个合法 JSON 值。

    Args:
        text: LLM 原始输出（可能含 markdown、前后缀文字、尾随逗号、注释等）。

    Returns:
        解析得到的 Python 对象（``dict`` / ``list`` / 标量），全部策略失败返回 ``None``。
        注意：合法的 JSON ``null`` 会被解析为 ``None``，与失败的 ``None`` 不可区分；
        若需严格区分，请改用 :func:`extract_json_object`。
    """
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None

    # 1. 直接解析
    obj = _try_loads(raw)
    if obj is not _MISS:
        return obj

    # 2. 围栏代码块（可能有多个，逐个尝试）
    for block in _FENCE_RE.findall(raw):
        block = block.strip()
        obj = _try_loads(block)
        if obj is not _MISS:
            return obj
        bal = _find_balanced(block)
        if bal is not None:
            obj = _try_loads(bal)
            if obj is not _MISS:
                return obj

    # 3. 第一个平衡片段
    bal = _find_balanced(raw)
    if bal is not None:
        obj = _try_loads(bal)
        if obj is not _MISS:
            return obj

    # 4. 轻量修复后重试
    repaired = _TRAILING_COMMA_RE.sub(r"\1", _strip_line_comments(raw))
    obj = _try_loads(repaired)
    if obj is not _MISS:
        return obj
    bal = _find_balanced(repaired)
    if bal is not None:
        obj = _try_loads(_TRAILING_COMMA_RE.sub(r"\1", bal))
        if obj is not _MISS:
            return obj

    return None


def extract_json_object(text: str | None, *, required: bool = False) -> dict | None:
    """仅接受 JSON **对象**（``dict``）的便捷封装。

    Args:
        text: LLM 原始输出。
        required: 为 True 且解析结果不是 dict 时抛 :class:`JSONExtractionError`；
                  为 False（默认）时返回 None。

    Returns:
        解析得到的 ``dict``，或 None。
    """
    obj = extract_json(text)
    if isinstance(obj, dict):
        return obj
    if required:
        raise JSONExtractionError(
            f"无法从文本中解析出 JSON 对象: {str(text)[:200]!r}"
        )
    return None
