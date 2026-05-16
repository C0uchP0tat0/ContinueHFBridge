"""JSON repair & extraction helpers for unreliable model output."""
import json
import logging
import re
from typing import Optional

log = logging.getLogger("continue_adapter")


def unwrap_json_from_markdown(raw: str) -> str:
    """GLM часто оборачивает JSON в ```json ... ``` — вытаскиваем тело для парсера."""
    s = (raw or "").strip()
    if not s:
        return s
    fence = re.search(r"^```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
    if fence:
        return fence.group(1).strip()
    fence2 = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
    if fence2:
        return fence2.group(1).strip()
    return s


def extract_first_json_object(s: str) -> Optional[str]:
    """
    Берёт первый сбалансированный JSON-объект {...} из строки (учёт кавычек и escape).
    Нужен, когда модель после валидного JSON дописывает мусор или второй объект.
    """
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : j + 1]
    return None


def strip_concatenated_json_echoes(msg: str) -> str:
    """Убирает второй «приклеенный» JSON в конце assistant_message (частая ошибка GLM)."""
    if not msg or len(msg) < 24:
        return msg
    for pat in (
        r"\n\s*\{\s*\"assistant_message\"\s*:",
        r"\n\s*\{\s*\"tool_calls\"\s*:",
        r"\n\s*\{\s*\"final\"\s*:",
    ):
        m = re.search(pat, msg)
        if m and m.start() >= 1:
            return msg[: m.start()].rstrip()
    return msg


def repair_truncated_json(s: str) -> Optional[str]:
    """Try to close a truncated JSON string/object/array so it can be parsed.

    Works by scanning the string, tracking open structures, then appending
    the minimum closing characters needed.  Returns None if result doesn't parse.
    """
    if not s or not s.strip():
        return None
    s = s.strip()
    if s[0] not in ('{', '['):
        return None

    # Walk through the string tracking structure
    stack: list = []  # '{' or '[' or '"'
    esc = False
    for c in s:
        if stack and stack[-1] == '"':
            # Inside string
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                stack.pop()
        else:
            if c == '"':
                stack.append('"')
            elif c in ('{', '['):
                stack.append(c)
            elif c == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif c == ']' and stack and stack[-1] == '[':
                stack.pop()

    if not stack:
        return None  # Already balanced

    log.info("repair_truncated_json: stack=%s len=%d", stack, len(s))

    # Close unclosed structures in reverse
    suffix = ""
    for item in reversed(stack):
        if item == '"':
            suffix += '"'
        elif item == '{':
            suffix += '}'
        elif item == '[':
            suffix += ']'

    candidate = s + suffix
    try:
        json.loads(candidate)
        log.info("repair_truncated_json: success with suffix=%s", repr(suffix))
        return candidate
    except json.JSONDecodeError as e:
        log.debug("repair_truncated_json: first attempt failed: %s", e)
        # Try also adding null value if truncated mid-key
        # e.g. {"key": "val", "key2": → {"key": "val", "key2": null}
        for pad in ('null', '""'):
            try:
                json.loads(s + pad + suffix)
                log.info("repair_truncated_json: success with pad=%s suffix=%s", pad, repr(suffix))
                return s + pad + suffix
            except json.JSONDecodeError:
                pass
    return None


def repair_json_string(s: str) -> str:
    """Fix common JSON errors produced by GLM."""
    if not s:
        return s
    # Trailing commas before closing brackets
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Python-style booleans/None → JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # Unescaped newlines inside strings (common GLM bug): replace with \n
    # Only between unescaped quotes
    lines = s.split("\n")
    if len(lines) > 1:
        # Try to rejoin: if removing newlines makes it valid JSON, do it
        joined = " ".join(l.strip() for l in lines)
        try:
            json.loads(joined)
            return joined
        except json.JSONDecodeError:
            pass
    return s
