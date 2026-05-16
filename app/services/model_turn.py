"""Orchestration layer — one model step per Continue HTTP request."""
from __future__ import annotations

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple

from fastapi import Request

from app.adapters import get_adapter, get_adapter_for_model
from app.adapters.glm import AGENT_JSON_RETRY
from app.config import FALLBACK_MODE, log
from app.parsing.tool_calls import (
    looks_like_continue_tool_markup,
    parse_assistant_json,
    parse_continue_system_tool_plaintext,
    parse_tool_calls_from_text,
    strip_continue_tool_fences,
)
from app.parsing.transcript import build_transcript_for_llm, _message_content, last_user_text
from app.tools.intent import (
    build_tool_arguments,
    openai_tool_call,
    select_tool_for_user_intent,
    synthetic_tool_calls_if_needed,
)
from app.tools.continue_defaults import DEFAULT_CONTINUE_TOOLS
from app.tools.validation import validate_and_fix_tool_calls


# ── Parse helpers ──────────────────────────────────────────────────────

def _try_parse_response(
    raw: str, tools: Optional[List[Dict[str, Any]]], messages: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Attempt to parse GLM raw response into (content, tool_calls, success).
    success=True means we got something usable (text or tool_calls).
    """
    if not (raw or "").strip():
        return "", [], False

    content, tool_calls = parse_assistant_json(raw, tools)

    # Fallback: try [tool_call:...] text patterns
    if not content.strip() and not tool_calls:
        extra = parse_tool_calls_from_text(raw)
        if extra:
            log.info("_try_parse: recovered tool_calls from [tool_call:...] text")
            return "", extra, True

    # Fallback: Continue TOOL_NAME/BEGIN_ARG markup in parsed content
    if not tool_calls and (content or "").strip():
        c2, tc2 = parse_continue_system_tool_plaintext(content, tools)
        if tc2:
            log.info("_try_parse: Continue TOOL_NAME/BEGIN_ARG markup -> tool_calls")
            content, tool_calls = c2, tc2

    # Fallback: Continue tool markup in raw output
    if not tool_calls and (raw or "").strip() and looks_like_continue_tool_markup(raw):
        c3, tc3 = parse_continue_system_tool_plaintext(raw, tools)
        if tc3:
            log.info("_try_parse: Continue tool markup in raw -> tool_calls")
            content, tool_calls = c3, tc3

    # Clean tool markup from content if we have tool_calls
    if tool_calls and (content or "").strip() and looks_like_continue_tool_markup(content):
        content = strip_continue_tool_fences(content).strip()

    success = bool(content.strip()) or bool(tool_calls)
    return content, tool_calls, success


# ── Fallback (FALLBACK_MODE=true) ─────────────────────────────────────

async def fallback_model_turn(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Заглушка при FALLBACK_MODE — та же логика намерений, что и repair-слой."""
    last = messages[-1] if messages else {}
    role = last.get("role") or "user"

    if role == "tool":
        return (
            "Результат инструмента учтён. Кратко опишите следующий шаг или я продолжу по задаче.",
            [],
        )

    if tools and role == "user":
        user_text = _message_content(last)
        chosen = select_tool_for_user_intent(user_text, tools) or tools[0]
        args = build_tool_arguments(chosen, user_text)
        fn = chosen.get("function") or {}
        name = fn.get("name") or "read_file"
        log.info("fallback_model_turn -> tool=%s args=%s", name, args)
        return "", [openai_tool_call(name, args)]

    return (
        "Я подключён как бэкенд для Continue (режим заглушки FALLBACK_MODE=true). "
        "Для реальных ответов выключите FALLBACK_MODE и настройте вызов модели.",
        [],
    )


# ── Main orchestration ────────────────────────────────────────────────

async def model_one_turn(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    cancel_event: Optional[asyncio.Event] = None,
    model_name: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Single model step — what Continue expects per HTTP request.
    Continue runs tools in the IDE and sends tool results in the next request.
    Includes retry on parse failure, tool_call validation, and cancellation support.
    """
    log.info(
        "model_one_turn start messages=%s last_role=%s tools=%s",
        len(messages),
        (messages[-1].get("role") if messages else None),
        len(tools or []),
    )

    if FALLBACK_MODE:
        return await fallback_model_turn(messages, tools)

    adapter = get_adapter_for_model(model_name) if model_name else get_adapter()

    # When Continue doesn't send tool definitions (agent mode without explicit
    # tools list), inject our defaults so the model knows exact tool names.
    effective_tools = tools if tools else DEFAULT_CONTINUE_TOOLS
    log.info(
        "model_one_turn: tools_from_continue=%s using_defaults=%s effective=%s",
        len(tools or []), not tools, len(effective_tools),
    )

    prompt = build_transcript_for_llm(messages, effective_tools)
    raw = await adapter.call(prompt, cancel_event=cancel_event)
    if cancel_event and cancel_event.is_set():
        log.info("model_one_turn: cancelled by client before parsing")
        return "", []
    if not raw:
        log.warning("model_one_turn: empty raw from model")
        return "Извините, сервис модели временно недоступен. Попробуйте позже.", []

    log.info("model_one_turn raw_len=%s head=%r", len(raw), raw[:800])

    content, tool_calls, success = _try_parse_response(raw, effective_tools, messages)

    # --- Retry on failure (if not cancelled) ---
    if not success and not (cancel_event and cancel_event.is_set()):
        log.warning("model_one_turn: first attempt failed to parse, retrying with hint")
        raw2 = await adapter.call(prompt, retry_hint=AGENT_JSON_RETRY, cancel_event=cancel_event)
        if cancel_event and cancel_event.is_set():
            log.info("model_one_turn: cancelled during retry")
            return "", []
        if raw2:
            log.info("model_one_turn retry raw_len=%s head=%r", len(raw2), raw2[:800])
            content2, tool_calls2, success2 = _try_parse_response(raw2, effective_tools, messages)
            if success2:
                content, tool_calls = content2, tool_calls2
                log.info("model_one_turn: retry succeeded")
            else:
                log.warning("model_one_turn: retry also failed, using raw as text")
                content = raw.strip()

    # --- Synthetic tool_calls (intent-based fallback) ---
    # Skip synthetic if the raw response clearly contained tool call patterns
    # (means the model DID try to call a tool, but JSON was truncated/malformed)
    raw_has_tool_pattern = bool(
        raw and ('"tool_calls"' in raw or "TOOL_NAME:" in raw or '"name":' in raw)
    )
    if not tool_calls and raw_has_tool_pattern:
        log.warning("model_one_turn: raw has tool patterns but parsing failed; NOT applying synthetic")
    parsed_before_synth = list(tool_calls)
    if not raw_has_tool_pattern:
        tool_calls = synthetic_tool_calls_if_needed(
            messages, effective_tools, content, tool_calls
        )
    if tool_calls and not parsed_before_synth:
        log.info("model_one_turn: applied synthetic tool_calls, clearing assistant text")
        content = ""

    # --- Validate tool_calls against schema ---
    if tool_calls:
        tool_calls = validate_and_fix_tool_calls(tool_calls, effective_tools)

    if not content.strip() and not tool_calls:
        log.warning(
            "model_one_turn: still empty after all fallbacks; raw_tail=%r",
            raw[-500:],
        )
        return (raw.strip() or "Пустой ответ модели."), []

    log.info(
        "model_one_turn result content_len=%s tool_calls=%s",
        len(content or ""),
        [(tc.get("function") or {}).get("name") for tc in tool_calls],
    )

    return content, tool_calls


# ── Disconnect monitor ─────────────────────────────────────────────────

async def monitor_disconnect(request: Request, cancel_event: asyncio.Event):
    """Background task: polls client disconnect and sets cancel_event."""
    try:
        while not cancel_event.is_set():
            if await request.is_disconnected():
                cancel_event.set()
                log.info("Client disconnected — cancelling model call")
                return
            await asyncio.sleep(0.5)
    except Exception:
        pass
