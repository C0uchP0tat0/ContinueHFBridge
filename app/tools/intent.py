"""Intent-based tool selection & synthetic tool call generation."""
import re
from typing import Dict, Any, List, Optional

from app.config import log
from app.parsing.tool_calls import openai_tool_call, extract_path_hint
from app.parsing.transcript import last_user_text


def select_tool_for_user_intent(
    user_text: str, tools: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Подбор инструмента Continue по последнему сообщению пользователя (без ответа модели)."""
    ut = user_text.lower()
    best: Optional[Dict[str, Any]] = None
    best_score = 0

    for td in tools:
        fn = td.get("function") or {}
        name_l = (fn.get("name") or "").lower()
        if not name_l:
            continue
        sc = 0
        if any(
            k in ut
            for k in (
                "read",
                "прочитай",
                "открой файл",
                "show file",
                "file content",
                "содержимое",
            )
        ) and any(x in name_l for x in ("read", "open", "file")):
            sc = max(sc, 8)
        if any(
            k in ut
            for k in (
                "grep",
                "search",
                "найди",
                "поиск",
                "find in code",
                "ищи",
            )
        ) and any(x in name_l for x in ("grep", "search")):
            sc = max(sc, 8)
        if any(k in ut for k in ("list", "ls", "директор", "папк", "перечисли")) and any(
            x in name_l for x in ("ls", "list", "dir")
        ):
            sc = max(sc, 8)
        if any(
            k in ut
            for k in ("run", "terminal", "command", "выполни", "shell", "cmd", "терминал")
        ) and any(x in name_l for x in ("terminal", "command", "run")):
            sc = max(sc, 8)
        if any(k in ut for k in ("edit", "change", "patch", "измени", "правк", "замени")) and (
            "edit" in name_l
        ):
            sc = max(sc, 8)
        if any(
            k in ut
            for k in (
                "create",
                "new file",
                "empty file",
                "пустой файл",
                "создай",
                "создать",
                "новый файл",
                "создай файл",
                "файл пуст",
            )
        ) and ("create" in name_l):
            sc = max(sc, 10)

        if sc > best_score:
            best_score = sc
            best = td

    return best


def default_args_from_schema(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool_def.get("function") or {}
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or list(props.keys())
    out: Dict[str, Any] = {}
    for key in required:
        spec = props.get(key, {}) if isinstance(props.get(key), dict) else {}
        t = spec.get("type")
        if t == "string":
            out[key] = spec.get("default", "")
        elif t == "integer":
            try:
                out[key] = int(spec.get("default", 0))
            except (TypeError, ValueError):
                out[key] = 0
        elif t == "number":
            try:
                out[key] = float(spec.get("default", 0))
            except (TypeError, ValueError):
                out[key] = 0.0
        elif t == "boolean":
            out[key] = bool(spec.get("default", False))
        elif t == "array":
            out[key] = spec.get("default", [])
        elif t == "object":
            out[key] = spec.get("default", {})
        else:
            out[key] = None
    return out


def build_tool_arguments(
    tool_def: Dict[str, Any], user_text: str, default_empty_filename: str = "empty.txt"
) -> Dict[str, Any]:
    """Заполняем аргументы по JSON Schema + эвристики из текста пользователя."""
    args = default_args_from_schema(tool_def)
    hint = extract_path_hint(user_text)
    ut = user_text.lower()

    if hint:
        for key in ("filepath", "path", "file_path", "target_file", "filename", "uri"):
            if key in args and not str(args.get(key) or "").strip():
                args[key] = hint

    # «Создай пустой файл» без имени — подставляем имя по умолчанию
    if any(
        k in ut
        for k in (
            "пустой файл",
            "пустой",
            "empty file",
            "новый файл",
            "new file",
            "создай файл",
            "создать файл",
        )
    ):
        if not hint:
            hint = default_empty_filename
        for key in ("filepath", "path", "file_path", "target_file", "filename"):
            if key in args and not str(args.get(key) or "").strip():
                args[key] = hint
        for key in ("contents", "content", "body", "text"):
            if key in args and args.get(key) is None:
                args[key] = ""

    # всё ещё пустые строковые обязательные поля — первое подставим как путь
    fn = tool_def.get("function") or {}
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    for key, spec in props.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("type") == "string" and key in args and not str(args.get(key) or "").strip():
            if any(x in (fn.get("name") or "").lower() for x in ("create", "write", "new")):
                args[key] = hint or default_empty_filename

    return args


def synthetic_tool_calls_if_needed(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    model_content: str,
    parsed_tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Если модель вернула только текст (без tool_calls), но пользователь явно просит действие —
    синтезируем вызов по схеме Continue, чтобы агент реально что-то сделал в IDE.
    """
    if parsed_tool_calls or not tools:
        return parsed_tool_calls

    last = messages[-1] if messages else {}
    if (last.get("role") or "") != "user":
        return parsed_tool_calls

    user_text = last_user_text(messages)
    chosen = select_tool_for_user_intent(user_text, tools)
    if not chosen:
        return parsed_tool_calls

    fn = chosen.get("function") or {}
    name = fn.get("name") or ""
    if not name:
        return parsed_tool_calls

    args = build_tool_arguments(chosen, user_text)
    log.info(
        "synthetic_tool_calls: model had no tools; user intent -> %s args=%s (user snippet=%r)",
        name,
        args,
        user_text[:200],
    )
    return [openai_tool_call(name, args)]
