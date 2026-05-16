"""Parse tool calls from various model output formats (JSON, text patterns, Continue markup)."""
import json
import os
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple

from app.config import log
from app.parsing.clean import clean
from app.parsing.json_repair import (
    extract_first_json_object,
    repair_json_string,
    repair_truncated_json,
    strip_concatenated_json_echoes,
    unwrap_json_from_markdown,
)


# ── helpers ────────────────────────────────────────────────────────────

def openai_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def extract_path_hint(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"""['"]([^'"]+\.\w+)['"]|(?:^|\s)([\w./\\-]+\.\w{1,8})(?:\s|$)""", text)
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip()


# ── [tool_call:...] text patterns ─────────────────────────────────────

def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    pattern1 = re.compile(
        r"""\[tool_call:\s*(\w+)\s+for\s+(\w+)\s+'([^']+)'\s+with\s+(\w+)\s+'([^']*)'\s*\]""",
        re.DOTALL,
    )
    pattern2 = re.compile(
        r"""\[tool_call:\s*(\w+)\s+for\s+(\w+)\s+'([^']+)'\s+with\s+(\w+)\s*\]
        ((?:\s*\n(?:\s*\d*\s*.*\n?)*?))
        (?=\n\s*[A-ZА-Я]|\Z)""",
        re.DOTALL | re.VERBOSE,
    )
    tool_calls: List[Dict[str, Any]] = []
    for match in pattern1.findall(text):
        function_name, arg1_name, arg1_value, arg2_name, content = match
        arguments = {arg1_name: arg1_value, arg2_name: content}
        if arg1_name == "absolute_path":
            arguments["file_path"] = arguments.pop("absolute_path")
        try:
            arguments_json = json.dumps(arguments, ensure_ascii=False)
            json.loads(arguments_json)
        except Exception as e:
            log.error("Invalid tool_call arguments: %s — %s", arguments, e)
            continue
        tool_calls.append(openai_tool_call(function_name, arguments))

    for match in pattern2.findall(text):
        function_name, arg1_name, arg1_value, arg2_name, raw_content = match
        content_lines = []
        for line in raw_content.split("\n"):
            cleaned_line = re.sub(r"^\s*\d+\s*", "", line)
            if cleaned_line.strip():
                content_lines.append(cleaned_line)
        content = "\n".join(content_lines).strip()
        arguments = {arg1_name: arg1_value, arg2_name: content}
        if arg1_name == "absolute_path":
            arguments["file_path"] = arguments.pop("absolute_path")
        try:
            arguments_json = json.dumps(arguments, ensure_ascii=False)
            json.loads(arguments_json)
        except Exception as e:
            log.error("Invalid tool_call arguments: %s — %s", arguments, e)
            continue
        tool_calls.append(openai_tool_call(function_name, json.loads(arguments_json)))

    return tool_calls


# ── Continue TOOL_NAME / BEGIN_ARG / END_ARG markup ────────────────────

_DEFAULT_CONTINUE_ARG_KEYS = (
    "filepath",
    "path",
    "file_path",
    "target_file",
    "filename",
    "contents",
    "content",
    "body",
    "text",
    "command",
    "dirPath",
    "query",
    "pattern",
    "old_string",
    "new_string",
    "uri",
    "url",
)


def collect_tool_property_keys(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    if tools:
        for td in tools:
            props = (td.get("function") or {}).get("parameters", {})
            if not isinstance(props, dict):
                continue
            for k in (props.get("properties") or {}).keys():
                if isinstance(k, str) and k not in seen:
                    seen.add(k)
                    keys.append(k)
    for k in _DEFAULT_CONTINUE_ARG_KEYS:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return sorted(keys, key=len, reverse=True)


def _normalize_continue_tool_newlines(text: str) -> str:
    """Восстанавливает перевод строк после ```tool, если модель склеила токены."""
    s = text
    s = re.sub(r"(?i)(```\s*tool)\s*([^\n\r])", r"\1\n\2", s, count=1)
    s = re.sub(r"(?i)(TOOL_NAME\s*:\s*[^\n\r]+?)\s*(BEGIN_ARG)", r"\1\n\2", s)
    s = re.sub(r"(?i)(END_ARG)\s*(BEGIN_ARG)", r"\1\n\2", s)
    return s


def _split_continue_arg_blob(blob: str, known_keys: List[str]) -> Tuple[str, str]:
    """Имя аргумента и значение из куска между BEGIN_ARG: ... END_ARG (в т.ч. склеенного)."""
    blob = (blob or "").strip()
    if not blob:
        return "", ""
    blob_lower = blob.lower()
    for k in known_keys:
        if k.lower() == blob_lower:
            return k, ""
    if "\n" in blob or "\r" in blob:
        line1, _, rest = blob.partition("\n")
        line1 = line1.split("\r")[0].strip()
        if line1 and re.match(r"^[\w.-]+$", line1) and len(line1) < 128:
            return line1, rest.replace("\r\n", "\n").strip()
    parts = blob.split(None, 1)
    if len(parts) == 2 and re.match(r"^[\w.-]+$", parts[0]) and len(parts[0]) < 128:
        return parts[0], parts[1].strip()
    lower_blob = blob.lower()
    for k in known_keys:
        kl = k.lower()
        if lower_blob.startswith(kl) and len(blob) > len(k):
            rest = blob[len(k) :]
            if not rest or rest[0] in " :\t\n\r":
                val = rest.lstrip(" :\t\n\r").strip()
                # Strip extra surrounding quotes GLM often adds
                if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
                    val = val[1:-1]
                return k, val
            # склеено: filepathempty.txt
            val = rest.strip()
            if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            return k, val
    return "", blob


def parse_continue_system_tool_plaintext(
    text: str, tools: Optional[List[Dict[str, Any]]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Распознаёт текстовый формат Continue (TOOL_NAME / BEGIN_ARG / END_ARG), конвертирует в OpenAI tool_calls.
    Возвращает (остаточный текст без блока, tool_calls).
    """
    if not text or not (
        re.search(r"(?i)\b(?:TOOL_NAME|tool_name)\s*:", text)
        or "```tool" in text.lower()
        or "BEGIN_ARG" in text.upper()
    ):
        return text, []

    known = collect_tool_property_keys(tools)
    norm = _normalize_continue_tool_newlines(text)

    m_name = re.search(r"(?i)\b(?:TOOL_NAME|tool_name)\s*:\s*([^\n\r`]+)", norm)
    if not m_name:
        return text, []
    raw_name = m_name.group(1).strip()
    tool_name = re.split(r"(?i)BEGIN_ARG", raw_name)[0].strip()
    tool_name = re.sub(r"[`]+$", "", tool_name)
    if not tool_name:
        return text, []

    args: Dict[str, Any] = {}
    for am in re.finditer(r"(?i)BEGIN_ARG\s*:\s*([\s\S]*?)\s*END_ARG", norm):
        blob = am.group(1).strip()
        aname, aval = _split_continue_arg_blob(blob, known)
        if aname:
            args[aname] = aval
        elif blob and re.match(r"^[\w.-]+$", blob):
            for k in known:
                if k.lower() == blob.lower():
                    args[k] = ""
                    break
            else:
                if "contents" not in args and "content" not in args:
                    args["contents"] = blob
        elif blob:
            if "contents" not in args and "content" not in args:
                args["contents"] = blob

    if not args:
        return text, []

    first_tool = (tools or [{}])[0].get("function", {}).get("name") if tools else None
    if tools and first_tool:
        valid = {((t.get("function") or {}).get("name") or "") for t in tools}
        if tool_name not in valid and first_tool:
            log.warning(
                "continue_tool_text: unknown tool_name=%r valid=%s — keeping as returned",
                tool_name,
                valid,
            )

    span_start = norm.lower().find("```tool")
    if span_start < 0:
        span_start = m_name.start()
    last_end = None
    for am in re.finditer(r"(?i)END_ARG", norm):
        last_end = am.end()
    span_end = last_end if last_end is not None else m_name.end()
    tail = re.search(r"(?i)```\s*$", norm[span_end:])
    if tail:
        span_end += tail.end()

    remainder = (norm[:span_start] + norm[span_end:]).strip()

    log.info(
        "continue_tool_text -> native tool_calls name=%s args=%s remainder_len=%s",
        tool_name,
        args,
        len(remainder),
    )
    return remainder, [openai_tool_call(tool_name, args)]


def looks_like_continue_tool_markup(text: str) -> bool:
    if not text:
        return False
    t = text.upper()
    return "TOOL_NAME" in t or "TOOL_NAME:" in text or "BEGIN_ARG" in t or "```TOOL" in text.lower()


def strip_continue_tool_fences(text: str) -> str:
    """Убирает остатки ```tool ... ``` из текста, если блок уже преобразован в tool_calls."""
    out = re.sub(r"(?is)```\s*tool[\s\S]*?```", "", text)
    return out.strip()


def _convert_tool_markup_to_markdown(text: str) -> str:
    """Convert tool markup into readable markdown — extract code from BEGIN_ARG blocks.

    Instead of stripping tool blocks (losing code), this extracts the file content
    and presents it as markdown code blocks.
    """
    tool_block_pattern = re.compile(r"```\s*tool\s*\n([\s\S]*?)```", re.I)
    result = text

    def _replace_tool_block(m):
        block = m.group(1)
        tool_match = re.search(r"TOOL_NAME\s*:\s*(.+)", block)
        tn = tool_match.group(1).strip() if tool_match else ""

        fp_match = re.search(
            r"BEGIN_ARG\s*:\s*(?:filepath|path)\s*\n(.*?)\nEND_ARG", block, re.S
        )
        filepath = fp_match.group(1).strip() if fp_match else ""

        content_match = re.search(
            r"BEGIN_ARG\s*:\s*(?:content|contents|changes)\s*\n([\s\S]*?)(?:\nEND_ARG|$)",
            block,
        )
        code_content = content_match.group(1) if content_match else ""

        if not code_content.strip():
            if filepath:
                return f"\n*[{tn}: {filepath}]*\n"
            return ""

        ext = os.path.splitext(filepath)[1].lstrip(".") if filepath else ""
        lang = ext if ext in ("py", "js", "ts", "html", "css", "sh", "yaml", "json", "md") else ""

        header = f"\n**{filepath}**:\n" if filepath else "\n"
        return f"{header}```{lang}\n{code_content.strip()}\n```\n"

    result = tool_block_pattern.sub(_replace_tool_block, result)

    result = re.sub(r"(?m)^(TOOL_NAME\s*:.*|BEGIN_ARG\s*:.*|END_ARG\s*)$", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def stream_text_as_chunks_for_client(piece: str) -> List[str]:
    """
    Continue парсит system-tool построчно; нельзя резать одну строку на куски по 24 символа.
    """
    if not piece:
        return []
    if looks_like_continue_tool_markup(piece) or "```" in piece:
        if "\n" in piece:
            return piece.splitlines(keepends=True)
        return [piece]
    chunk_size = 24
    return [piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size)]


# ── JSON-based assistant response parsing ──────────────────────────────

def _load_assistant_json_dict(to_parse: str, text: str) -> Optional[Dict[str, Any]]:
    """Пытается получить dict из ответа модели; при обрезке хвоста — первый валидный объект."""
    for candidate in (to_parse, text):
        if not (candidate or "").strip():
            continue
        c = candidate.strip()
        try:
            o = json.loads(c)
            if isinstance(o, dict):
                return o
        except json.JSONDecodeError:
            pass
        repaired = repair_json_string(c)
        if repaired != c:
            try:
                o = json.loads(repaired)
                if isinstance(o, dict):
                    log.info("_load_assistant_json_dict: repaired JSON succeeded")
                    return o
            except json.JSONDecodeError:
                pass
        inner = extract_first_json_object(c)
        if inner:
            try:
                o = json.loads(inner)
                if isinstance(o, dict):
                    return o
            except json.JSONDecodeError:
                rep_inner = repair_json_string(inner)
                try:
                    o = json.loads(rep_inner)
                    if isinstance(o, dict):
                        return o
                except json.JSONDecodeError:
                    continue
        # Last resort: truncated JSON — close unclosed brackets/strings
        truncated = repair_truncated_json(c)
        if truncated:
            try:
                o = json.loads(truncated)
                if isinstance(o, dict):
                    log.info("_load_assistant_json_dict: repaired truncated JSON (%d -> %d chars)", len(c), len(truncated))
                    return o
            except json.JSONDecodeError:
                pass
        # Fallback: when JSON is truncated with large changes/contents, return assistant_message
        # as content (not as changes) so the model can retry with smaller edits instead of
        # writing the message literally into the file.
        if '"tool_calls"' in c or '"name"' in c:
            msg_match = re.search(r'"assistant_message"\s*:\s*"([^"]+)"', c)
            if msg_match:
                msg = msg_match.group(1)
                log.info("_load_assistant_json_dict: truncated JSON, returning assistant_message as content for retry")
                return {"assistant_message": msg, "tool_calls": []}
    return None


def parse_assistant_json(
    raw: str, tools: Optional[List[Dict[str, Any]]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse model JSON into (assistant_text, openai_style tool_calls)."""
    if not (raw or "").strip():
        return "", []

    text = clean(raw)
    to_parse = unwrap_json_from_markdown(text)

    obj = _load_assistant_json_dict(to_parse, text)
    if obj is None:
        extra = parse_tool_calls_from_text(text)
        if extra:
            return "", extra
        return text, []

    if "final" in obj and isinstance(obj["final"], str):
        return obj["final"], []

    if "assistant_message" in obj:
        msg = obj.get("assistant_message") or ""
        tcs = obj.get("tool_calls")
    elif "message" in obj and isinstance(obj["message"], str):
        msg = obj["message"]
        tcs = obj.get("tool_calls")
    elif "content" in obj:
        msg = str(obj.get("content") or "")
        tcs = obj.get("tool_calls")
    else:
        msg = ""
        tcs = obj.get("tool_calls")

    out_calls: List[Dict[str, Any]] = []
    if isinstance(tcs, list):
        for item in tcs:
            if not isinstance(item, dict):
                continue
            if "function" in item and isinstance(item["function"], dict):
                fn = item["function"]
                name = fn.get("name") or ""
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args_dict = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args_dict = {}
                elif isinstance(args, dict):
                    args_dict = args
                else:
                    args_dict = {}
                out_calls.append(
                    {
                        "id": item.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args_dict, ensure_ascii=False),
                        },
                    }
                )
                continue
            name = item.get("name") or item.get("tool") or ""
            args = item.get("arguments")
            if isinstance(args, dict):
                args_dict = args
            elif isinstance(args, str):
                try:
                    args_dict = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args_dict = {}
            else:
                args_dict = {}
            if name:
                out_calls.append(openai_tool_call(name, args_dict))

    if not out_calls and "action" in obj:
        from app.tools.registry import TOOLS
        action = obj.get("action")
        if isinstance(action, str) and action in TOOLS:
            args = {k: v for k, v in obj.items() if k != "action"}
            out_calls.append(openai_tool_call(action, args))

    msg = strip_concatenated_json_echoes(str(msg))
    if len(msg) > 200_000:
        msg = msg[:200_000] + "\n\n…(обрезано)"

    return msg, out_calls
