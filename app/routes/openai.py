"""OpenAI-compatible API routes: /v1/chat/completions, /v1/completions, /v1/models."""
import asyncio
import json
import re
import time
import uuid
from typing import AsyncIterator, Dict, Any, List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.adapters import get_adapter, get_adapter_for_model
from app.adapters.glm import FIM_SYSTEM
from app.config import MODEL_NAME, log
from app.parsing.tool_calls import stream_text_as_chunks_for_client
from app.schemas.requests import CompletionRequest, OpenAIChatRequest
from app.services.model_turn import model_one_turn, monitor_disconnect

router = APIRouter()


# ── /v1/chat/completions ──────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest, request: Request):
    log.info(
        "/v1/chat/completions stream=%s messages=%s tools=%s",
        req.stream,
        len(req.messages),
        bool(req.tools),
    )
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"

    # --- NON-STREAMING ---
    if not req.stream:
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            content, tool_calls = await model_one_turn(req.messages, req.tools, cancel_event=cancel_event, model_name=req.model)
        except asyncio.CancelledError:
            log.info("/v1/chat/completions: request cancelled")
            content, tool_calls = "", []
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass

        log.info(
            "/v1/chat/completions -> finish_reason=%s content_len=%s tools=%s",
            "tool_calls" if tool_calls else "stop",
            len(content or ""),
            [(t.get("function") or {}).get("name") for t in tool_calls],
        )
        msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
            if not (content or "").strip():
                msg["content"] = None
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        }

    # --- STREAMING ---
    async def stream_openai() -> AsyncIterator[str]:
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        ts = int(time.time())

        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': ts, 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

        content = ""
        tool_calls: List[Dict[str, Any]] = []
        try:
            content, tool_calls = await model_one_turn(
                req.messages, req.tools, cancel_event=cancel_event, model_name=req.model
            )
        except asyncio.CancelledError:
            log.info("/v1/chat/completions stream: cancelled during model call")
        except Exception as e:
            log.error("/v1/chat/completions stream error: %s", e)
            content = f"Ошибка при обращении к модели: {e}"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass

        log.info(
            "/v1/chat/completions stream -> content_len=%s tools=%s",
            len(content or ""),
            [(t.get("function") or {}).get("name") for t in tool_calls],
        )

        piece = content or ""
        # Ensure there's always some content when tool calls are present
        if not piece and tool_calls:
            piece = " "
        if piece:
            for part in stream_text_as_chunks_for_client(piece):
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': ts, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': part}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)

        if tool_calls:
            for idx, tc in enumerate(tool_calls):
                fn = tc.get("function") or {}
                tc_delta = {
                    "index": idx,
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": ""},
                }
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': ts, 'model': req.model, 'choices': [{'index': 0, 'delta': {'tool_calls': [tc_delta]}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)
                # Stream arguments in small chunks to avoid "Premature Close"
                args_str = fn.get("arguments", "{}")
                chunk_size = 512
                for i in range(0, len(args_str), chunk_size):
                    chunk = args_str[i:i + chunk_size]
                    tc_args_delta = {"index": idx, "function": {"arguments": chunk}}
                    yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': ts, 'model': req.model, 'choices': [{'index': 0, 'delta': {'tool_calls': [tc_args_delta]}, 'finish_reason': None}]})}\n\n"
                    await asyncio.sleep(0.01)

        finish = "tool_calls" if tool_calls else "stop"
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': ts, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_openai(), media_type="text/event-stream")


@router.post("/chat/completions")
async def chat_completions_no_v1(req: OpenAIChatRequest, request: Request):
    return await chat_completions(req, request)


# ── /v1/completions (FIM / tab autocomplete) ──────────────────────────

@router.post("/v1/completions")
async def completions(req: CompletionRequest):
    log.info("/v1/completions prompt_len=%s stream=%s", len(req.prompt), req.stream)

    cancel_event = asyncio.Event()
    adapter = get_adapter_for_model(req.model) if req.model else get_adapter()

    fim_prompt = req.prompt
    if req.suffix:
        fim_prompt = f"{req.prompt}[FILL_HERE]{req.suffix}"

    raw = await adapter.call(fim_prompt, cancel_event=cancel_event, system_prompt=FIM_SYSTEM)
    if not raw:
        raw = ""

    raw = re.sub(r"^```\w*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        async def stream_completion():
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "text": raw, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_completion(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "text": raw, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(req.prompt) // 4,
            "completion_tokens": len(raw) // 4,
            "total_tokens": (len(req.prompt) + len(raw)) // 4,
        },
    }


# ── /v1/models ─────────────────────────────────────────────────────────

@router.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "adapter",
            }
        ],
    }
