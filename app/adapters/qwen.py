"""Qwen3-VL-235B adapter — talks to a Gradio Space via SSE (ZeroGPU).

The Qwen demo Space uses a two-step Gradio API:
  Step 1  /add_text  (fn_index=0): appends user message to chat history
  Step 2  /predict   (fn_index=1): generates model response (streaming)

API schema discovered via /gradio_api/info endpoint.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.adapters.base import ModelAdapter
from app.adapters.glm import AGENT_JSON_SYSTEM
from app.config import (
    FALLBACK_MODE,
    QWEN_ENDPOINT,
    QWEN_TIMEOUT_PER_LINE,
    QWEN_TIMEOUT_TOTAL,
    QWEN_ZEROGPU_TOKEN,
    QWEN_ZEROGPU_UUID,
    log,
)
from app.parsing.clean import clean


class QwenAdapter(ModelAdapter):
    """Qwen3-VL-235B via Gradio SSE on HuggingFace Spaces (ZeroGPU)."""

    def __init__(self) -> None:
        self.endpoint = QWEN_ENDPOINT

    # ── public interface ───────────────────────────────────────────────

    async def call(
        self,
        prompt: str,
        *,
        retry_hint: str = "",
        cancel_event: Optional[asyncio.Event] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if FALLBACK_MODE:
            log.warning("FALLBACK_MODE: skipping remote Qwen")
            return ""

        sys_p = system_prompt if system_prompt is not None else AGENT_JSON_SYSTEM
        session_hash = uuid.uuid4().hex[:11]

        user_prompt = (retry_hint + "\n\n" + prompt) if retry_hint else prompt

        # Prepend system prompt to user message (Qwen has no separate system field)
        full_text = f"[SYSTEM]\n{sys_p}\n[/SYSTEM]\n\n{user_prompt}"

        log.info(
            "Calling Qwen session_hash=%s system_len=%s prompt_len=%s",
            session_hash, len(sys_p), len(user_prompt),
        )

        try:
            result = await asyncio.wait_for(
                self._call_inner(full_text, session_hash, cancel_event),
                timeout=QWEN_TIMEOUT_TOTAL,
            )
        except asyncio.TimeoutError:
            log.error("Qwen total timeout after %ss (session=%s)", QWEN_TIMEOUT_TOTAL, session_hash)
            return ""
        except asyncio.CancelledError:
            log.info("Qwen call task cancelled (session=%s)", session_hash)
            return ""

        cleaned = clean(result or "")
        log.info(
            "Qwen done raw_len=%s cleaned_len=%s preview=%r",
            len(result or ""),
            len(cleaned),
            (cleaned or result or "")[:1500],
        )
        return cleaned

    # ── internal ───────────────────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers, including ZeroGPU tokens when available."""
        origin = self.endpoint.rsplit("/gradio_api", 1)[0]
        headers: Dict[str, str] = {
            "content-type": "application/json",
            "origin": origin,
            "referer": f"{origin}/?__theme=system",
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
            ),
        }
        if QWEN_ZEROGPU_TOKEN:
            headers["x-zerogpu-token"] = QWEN_ZEROGPU_TOKEN
        if QWEN_ZEROGPU_UUID:
            headers["x-zerogpu-uuid"] = QWEN_ZEROGPU_UUID
        return headers

    async def _queue_call(
        self,
        client: httpx.AsyncClient,
        fn_index: int,
        data: List[Any],
        session_hash: str,
        cancel_event: Optional[asyncio.Event],
    ) -> Any:
        """Post to /queue/join and read SSE until process_completed.

        Returns raw output dict from Gradio.
        """
        headers = self._build_headers()

        # ── Join queue ─────────────────────────────────────────────────
        try:
            resp = await asyncio.wait_for(
                client.post(
                    f"{self.endpoint}/queue/join",
                    headers=headers,
                    json={
                        "data": data,
                        "event_data": None,
                        "fn_index": fn_index,
                        "trigger_id": 7,
                        "session_hash": session_hash,
                    },
                ),
                timeout=30,
            )
            if resp.status_code != 200:
                log.error("Qwen fn=%s join HTTP %s: %s", fn_index, resp.status_code, resp.text[:500])
                return None
            log.info("Qwen fn=%s join ok", fn_index)
        except asyncio.TimeoutError:
            log.error("Qwen fn=%s join timeout", fn_index)
            return None
        except asyncio.CancelledError:
            return None
        except Exception as e:
            log.error("Qwen fn=%s join error: %s", fn_index, e)
            return None

        # ── Read SSE ───────────────────────────────────────────────────
        last_output = None
        try:
            async with client.stream(
                "GET",
                f"{self.endpoint}/queue/data",
                params={"session_hash": session_hash},
                headers={"accept": "text/event-stream"},
            ) as sse:
                line_iter = sse.aiter_lines().__aiter__()
                while True:
                    if cancel_event and cancel_event.is_set():
                        return last_output
                    try:
                        line = await asyncio.wait_for(
                            line_iter.__anext__(), timeout=QWEN_TIMEOUT_PER_LINE
                        )
                    except asyncio.TimeoutError:
                        log.error("Qwen fn=%s SSE timeout", fn_index)
                        return last_output
                    except StopAsyncIteration:
                        break
                    except asyncio.CancelledError:
                        return last_output

                    if not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue

                    msg = evt.get("msg", "")
                    if msg == "process_starts":
                        log.info("Qwen fn=%s process started (session=%s)", fn_index, session_hash)
                    elif msg in ("process_generating", "process_completed"):
                        out = evt.get("output")
                        if out:
                            last_output = out
                        if msg == "process_completed":
                            break
                    elif msg == "queue_full":
                        log.warning("Qwen fn=%s queue full", fn_index)
                        return None
        except httpx.ReadTimeout:
            log.error("Qwen fn=%s stream timeout", fn_index)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Qwen fn=%s stream error: %s", fn_index, e)

        return last_output

    @staticmethod
    def _extract_bot_text(output: Any) -> str:
        """Extract the last bot message from Gradio output.

        output.data[0] = chatbot history = [[user_msg, bot_msg], ...]
        """
        if not output or not isinstance(output, dict):
            return ""
        data = output.get("data", [])
        if not data:
            return ""
        # First element is the chatbot history
        history = data[0] if isinstance(data[0], list) else None
        if not history:
            return ""
        for pair in reversed(history):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                bot = pair[1]
                if bot is None:
                    continue
                if isinstance(bot, str) and bot.strip():
                    return bot.strip()
                if isinstance(bot, dict):
                    return str(bot.get("content") or bot.get("text") or bot.get("value") or "")
        return ""

    async def _call_inner(
        self,
        full_text: str,
        session_hash: str,
        cancel_event: Optional[asyncio.Event],
    ) -> str:
        async with httpx.AsyncClient(timeout=QWEN_TIMEOUT_TOTAL + 30) as client:

            # ── Step 1: /add_text (fn_index=0) ─────────────────────────
            # Config: inputs=[chatbot(2), state(4), textbox(3)]
            # outputs=[chatbot(2), state(4)]
            # data = [chatbot_history, state, user_text]
            log.info("Qwen step 1: add_text (session=%s)", session_hash)
            add_out = await self._queue_call(
                client, 0,
                data=[[], None, full_text],
                session_hash=session_hash,
                cancel_event=cancel_event,
            )

            if cancel_event and cancel_event.is_set():
                return ""

            # Extract history and state from add_text output
            history: List[Any] = []
            state = None
            if add_out and isinstance(add_out, dict):
                out_data = add_out.get("data", [])
                if len(out_data) >= 1 and isinstance(out_data[0], list):
                    history = out_data[0]
                if len(out_data) >= 2:
                    state = out_data[1]
            if not history:
                history = [[full_text, None]]
            log.info(
                "Qwen step 1 done: history=%s entries, state=%s",
                len(history), "present" if state is not None else "null",
            )

            # ── Step 2: /predict (fn_index=1) ──────────────────────────
            # Config: inputs=[chatbot(2), state(4)], outputs=[chatbot(2)]
            # trigger_after=0 (auto-chains after add_text)
            # data = [chatbot_history, state]
            log.info("Qwen step 2: predict (session=%s)", session_hash)
            pred_out = await self._queue_call(
                client, 1,
                data=[history, state],
                session_hash=session_hash,
                cancel_event=cancel_event,
            )

            result = self._extract_bot_text(pred_out)
            log.info("Qwen predict result_len=%s", len(result))
            return result
