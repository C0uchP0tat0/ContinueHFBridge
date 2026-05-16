"""Ollama-compatible API routes: /api/chat, /api/generate, /api/tags, /api/show."""
import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Any, List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import MODEL_NAME, log
from app.parsing.tool_calls import stream_text_as_chunks_for_client
from app.schemas.requests import OllamaChatRequest, OllamaGenerateRequest
from app.services.model_turn import model_one_turn, monitor_disconnect
from app.tools.validation import ollama_done_metadata, ollama_tool_calls_from_openai

router = APIRouter()


# ── /api/chat ──────────────────────────────────────────────────────────

@router.post("/api/chat")
async def api_chat(req: OllamaChatRequest, request: Request):
    log.info(
        "/api/chat stream=%s messages=%s tools=%s",
        req.stream,
        len(req.messages),
        bool(req.tools),
    )

    # --- NON-STREAMING ---
    if not req.stream:
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            content, tool_calls = await model_one_turn(req.messages, req.tools, cancel_event=cancel_event, model_name=req.model or "")
        except asyncio.CancelledError:
            log.info("/api/chat: request cancelled")
            content, tool_calls = "", []
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info(
            "/api/chat -> content_len=%s tool_calls=%s",
            len(content or ""),
            [(t.get("function") or {}).get("name") for t in tool_calls],
        )
        ollama_calls = ollama_tool_calls_from_openai(tool_calls) if tool_calls else None
        body: Dict[str, Any] = {
            "model": req.model or MODEL_NAME,
            "message": {"role": "assistant", "content": content},
            **ollama_done_metadata(),
        }
        if ollama_calls:
            body["message"]["tool_calls"] = ollama_calls
        return body

    # --- STREAMING ---
    async def stream_ollama() -> AsyncIterator[str]:
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        mid = req.model or MODEL_NAME
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        yield json.dumps(
            {"model": mid, "created_at": created, "message": {"role": "assistant", "content": ""}, "done": False},
            ensure_ascii=False,
        ) + "\n"

        content = ""
        tool_calls: List[Dict[str, Any]] = []
        try:
            content, tool_calls = await model_one_turn(
                req.messages, req.tools, cancel_event=cancel_event, model_name=req.model or ""
            )
        except asyncio.CancelledError:
            log.info("/api/chat stream: cancelled during model call")
        except Exception as e:
            log.error("/api/chat stream error: %s", e)
            content = f"Ошибка: {e}"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass

        log.info(
            "/api/chat stream -> content_len=%s tools=%s",
            len(content or ""),
            [(t.get("function") or {}).get("name") for t in tool_calls],
        )
        ollama_calls = ollama_tool_calls_from_openai(tool_calls) if tool_calls else None

        piece = content or ""
        if piece:
            for chunk_text in stream_text_as_chunks_for_client(piece):
                yield json.dumps(
                    {"model": mid, "created_at": created, "message": {"role": "assistant", "content": chunk_text}, "done": False},
                    ensure_ascii=False,
                ) + "\n"
                await asyncio.sleep(0.01)

        final_msg: Dict[str, Any] = {"role": "assistant", "content": piece}
        if ollama_calls:
            final_msg["tool_calls"] = ollama_calls
        yield json.dumps(
            {"model": mid, "created_at": created, "message": final_msg, **ollama_done_metadata()},
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(stream_ollama(), media_type="application/x-ndjson")


# ── /api/generate ──────────────────────────────────────────────────────

@router.post("/api/generate")
async def api_generate(req: OllamaGenerateRequest, request: Request):
    log.info("/api/generate stream=%s", req.stream)
    fake_messages: List[Dict[str, Any]] = [{"role": "user", "content": req.prompt}]

    # --- NON-STREAMING ---
    if not req.stream:
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            content, _ = await model_one_turn(fake_messages, None, cancel_event=cancel_event)
        except asyncio.CancelledError:
            log.info("/api/generate: request cancelled")
            content = ""
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass
        return {"model": req.model, "response": content, "done": True}

    # --- STREAMING ---
    async def stream_gen() -> AsyncIterator[str]:
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        mid = req.model or MODEL_NAME
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        yield json.dumps(
            {"model": mid, "created_at": created, "response": "", "done": False},
            ensure_ascii=False,
        ) + "\n"

        content = ""
        try:
            content, _ = await model_one_turn(fake_messages, None, cancel_event=cancel_event)
        except asyncio.CancelledError:
            log.info("/api/generate stream: cancelled")
        except Exception as e:
            log.error("/api/generate stream error: %s", e)
            content = f"Ошибка: {e}"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass

        for i in range(0, len(content), 40):
            part = content[i : i + 40]
            yield json.dumps(
                {"model": mid, "created_at": created, "response": part, "done": False},
                ensure_ascii=False,
            ) + "\n"
            await asyncio.sleep(0.01)
        yield json.dumps(
            {"model": mid, "created_at": created, "response": "", "done": True},
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(stream_gen(), media_type="application/x-ndjson")


# ── Metadata ───────────────────────────────────────────────────────────

@router.get("/api/tags")
async def tags():
    return {"models": [{"name": MODEL_NAME, "model": MODEL_NAME}]}


@router.post("/api/show")
async def show():
    return {
        "license": "",
        "modelfile": f"FROM {MODEL_NAME}",
        "parameters": "",
        "template": "",
    }
