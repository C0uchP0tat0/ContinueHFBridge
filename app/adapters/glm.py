"""GLM-4 adapter — talks to a Gradio Space via SSE."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

import httpx

from app.adapters.base import ModelAdapter
from app.config import (
    GLM_ENDPOINT,
    GLM_TIMEOUT_PER_LINE,
    GLM_TIMEOUT_TOTAL,
    FALLBACK_MODE,
    log,
)
from app.parsing.clean import clean


# ── System prompts (used as defaults) ──────────────────────────────────

AGENT_JSON_SYSTEM = """
You are an expert AI programming assistant integrated into VS Code via the Continue extension.
You help the user write, debug, refactor, and understand code across any language or framework.
You ONLY work on the current task. Never generate unrelated content.

Your workflow:
- The user sends messages and you respond with actions or explanations.
- You have access to tools (listed in AVAILABLE_TOOLS_JSON in the conversation).
- Each response = exactly ONE step. After you call a tool, the IDE executes it and sends the result back. Then you decide the next step.

OUTPUT FORMAT — always reply with EXACTLY ONE valid JSON object, nothing else:
  Action: {"assistant_message": "<explanation>", "tool_calls": [{"name": "<TOOL_NAME>", "arguments": {"<arg>": "<value>"}}]}
  Text only: {"assistant_message": "<your answer>", "tool_calls": []}

AVAILABLE TOOLS (use EXACT names from AVAILABLE_TOOLS_JSON):
- read_file: read file contents (arg: filepath, e.g. "main.py" or "src/utils.py")
- edit_existing_file: edit existing file (args: filepath, changes)
- create_new_file: create NEW file (args: filepath, contents) — ONLY if file does NOT exist
- run_terminal_command: run shell command (arg: command)
- ls: list directory (arg: dirPath — use "./" for root)
- grep_search: search in files (arg: query)

STRICT RULES:
1. Output ONLY the JSON object. No markdown fences, no text before or after, no duplicates.
2. Use EXACT tool names and argument names. filepath must be a bare relative path (e.g. "file.py", NOT "./file.py").
3. Fill ALL required arguments from the conversation context.
4. ONE tool_call per response. Multi-step tasks = multiple request-response cycles.
5. After a TOOL_RESULT, analyze it and respond or call the next tool. Never ignore results.
6. If a tool call FAILS, do NOT repeat the same call. Change your approach or ask the user.
7. BEFORE creating a file: ALWAYS first use ls to check if it exists. If it exists, use edit_existing_file.
8. WORKFLOW ORDER: ls → read_file → then edit_existing_file or create_new_file.
9. When writing code in tool arguments: output ONLY valid code. No unrelated text mixed in.
10. assistant_message must contain your own meaningful text about what you are doing.
11. CRITICAL: For edit_existing_file, use old_string/new_string format for targeted edits. DO NOT send the entire file content as "changes" — it will be truncated. Make small incremental changes instead of full file replacements.
12. DO NOT create new files with different names when the task is to edit an existing file. Always use the exact filepath that was read or mentioned in the conversation.
13. Make VERY small edits: change 1-2 lines at a time. If you need to add multiple imports or change a large function, do it in multiple separate edits to avoid truncation.
14. CRITICAL: Change ONLY the minimal needed part. Example: to add an import, replace just "from selenium.common.exceptions import TimeoutException" with "from selenium.common.exceptions import TimeoutException, NoSuchElementException" — do NOT replace the entire import block.
""".strip()

AGENT_JSON_RETRY = """
Your previous response was not valid JSON.
Reply with EXACTLY ONE JSON object — no markdown fences, no extra text.
Format: {"assistant_message": "your text", "tool_calls": []}
""".strip()

PLAIN_SYSTEM = """
You are an expert AI programming assistant integrated into VS Code via the Continue extension.
You help the user write, debug, refactor, and understand code across any language or framework.

Rules:
- Respond in plain text or markdown. Be concise, precise, and helpful.
- When showing code, use fenced code blocks with the correct language tag.
- Explain your reasoning step by step when the problem is complex.
- If you are unsure, say so honestly rather than guessing.
- Do NOT output JSON. Do NOT invent tool calls. Just answer directly.
""".strip()

FIM_SYSTEM = """
You are a code completion engine. Output ONLY the code that should be inserted at the cursor position.
No explanations, no markdown fences, no comments, no extra text. Just the raw code to insert.
""".strip()


# ── Gradio helpers ─────────────────────────────────────────────────────

def _extract_gradio_result(out: Dict[str, Any]) -> str:
    """Robustly extract text from Gradio process_completed output."""
    data_items = out.get("data", [])
    if not data_items:
        return ""
    for item in data_items:
        # Chatbot format: [[user_msg, bot_msg], ...]
        if isinstance(item, list):
            for pair in reversed(item):
                if isinstance(pair, list) and len(pair) >= 2:
                    bot = pair[-1]
                    if isinstance(bot, dict):
                        text = bot.get("content") or bot.get("text") or bot.get("value") or ""
                        if text:
                            return str(text)
                    elif isinstance(bot, str) and bot.strip():
                        return bot
                elif isinstance(pair, dict):
                    text = pair.get("content") or pair.get("text") or pair.get("value") or ""
                    if text:
                        return str(text)
                elif isinstance(pair, str) and pair.strip():
                    return pair
        elif isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("value") or ""
            if text:
                return str(text)
        elif isinstance(item, str) and item.strip():
            return item
    # Last resort
    try:
        return str(data_items[0][1].get("content", ""))
    except (IndexError, TypeError, AttributeError):
        return ""


# ── Adapter ────────────────────────────────────────────────────────────

class GLMAdapter(ModelAdapter):
    """GLM-4 via Gradio SSE on HuggingFace Spaces."""

    def __init__(self) -> None:
        self.endpoint = GLM_ENDPOINT

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
            log.warning("FALLBACK_MODE: skipping remote GLM")
            return ""

        sys_p = system_prompt if system_prompt is not None else AGENT_JSON_SYSTEM
        session_hash = str(uuid.uuid4())[:12]

        # User prompt — retry hint is prepended to help the model recover
        user_prompt = (retry_hint + "\n\n" + prompt) if retry_hint else prompt

        log.info(
            "Calling GLM session_hash=%s system_len=%s prompt_len=%s",
            session_hash, len(sys_p), len(user_prompt),
        )

        try:
            result = await asyncio.wait_for(
                self._call_inner(user_prompt, sys_p, session_hash, cancel_event),
                timeout=GLM_TIMEOUT_TOTAL,
            )
        except asyncio.TimeoutError:
            log.error("GLM total timeout after %ss (session=%s)", GLM_TIMEOUT_TOTAL, session_hash)
            return ""
        except asyncio.CancelledError:
            log.info("GLM call task cancelled (session=%s)", session_hash)
            return ""

        cleaned = clean(result or "")
        log.info(
            "GLM done raw_len=%s cleaned_len=%s preview=%r",
            len(result or ""),
            len(cleaned),
            (cleaned or result or "")[:1500],
        )
        return cleaned

    # ── internal ───────────────────────────────────────────────────────

    async def _call_inner(
        self,
        user_prompt: str,
        system_prompt: str,
        session_hash: str,
        cancel_event: Optional[asyncio.Event],
    ) -> str:
        async with httpx.AsyncClient(timeout=GLM_TIMEOUT_TOTAL + 30) as client:
            # Queue join — Gradio format:
            #   data[0] = user message
            #   data[1] = None (history / image)
            #   data[2] = system prompt
            #   data[3] = True  (web search flag)
            #   data[4] = 1     (temperature / mode)
            try:
                join_response = await asyncio.wait_for(
                    client.post(
                        f"{self.endpoint}/queue/join",
                        json={
                            "data": [user_prompt, None, system_prompt, True, 1],
                            "fn_index": 0,
                            "trigger_id": 8,
                            "session_hash": session_hash,
                        },
                    ),
                    timeout=30,
                )
                if join_response.status_code != 200:
                    log.error("Queue join HTTP %s: %s", join_response.status_code, join_response.text[:500])
                    return ""
                log.info("Queue join status: %s", join_response.status_code)
            except asyncio.TimeoutError:
                log.error("Queue join timeout after 30s")
                return ""
            except asyncio.CancelledError:
                log.info("Queue join cancelled")
                return ""
            except Exception as e:
                log.error("Queue join error: %s", e)
                return ""

            # SSE stream
            result = ""
            try:
                async with client.stream(
                    "GET",
                    f"{self.endpoint}/queue/data",
                    params={"session_hash": session_hash},
                ) as stream:
                    line_iter = stream.aiter_lines().__aiter__()
                    while True:
                        if cancel_event and cancel_event.is_set():
                            log.info("GLM call cancelled by client (session=%s)", session_hash)
                            return result or ""
                        try:
                            line = await asyncio.wait_for(
                                line_iter.__anext__(), timeout=GLM_TIMEOUT_PER_LINE
                            )
                        except asyncio.TimeoutError:
                            log.error(
                                "GLM SSE line timeout after %ss (session=%s), returning partial",
                                GLM_TIMEOUT_PER_LINE,
                                session_hash,
                            )
                            return result or ""
                        except StopAsyncIteration:
                            break
                        except asyncio.CancelledError:
                            log.info("GLM SSE read cancelled (session=%s)", session_hash)
                            return result or ""

                        if not line.startswith("data:"):
                            continue
                        try:
                            data = json.loads(line[5:])
                            msg_type = data.get("msg", "")
                            if msg_type == "process_completed":
                                out = data.get("output", {})
                                if out:
                                    result = _extract_gradio_result(out)
                                break
                            elif msg_type == "process_generating":
                                out = data.get("output", {})
                                if out:
                                    partial = _extract_gradio_result(out)
                                    if partial:
                                        result = partial
                            elif msg_type == "queue_full":
                                log.warning("Gradio queue full (session=%s)", session_hash)
                                return ""
                            elif msg_type == "process_starts":
                                log.info("Gradio process started (session=%s)", session_hash)
                        except json.JSONDecodeError as e:
                            log.warning("Gradio SSE parse error: %s line=%r", e, line[:200])
                            continue
                        except Exception as e:
                            log.error("Gradio SSE error: %s", e)
                            continue
            except httpx.ReadTimeout:
                log.error("GLM stream read timeout (session=%s)", session_hash)
                return result or ""
            except asyncio.CancelledError:
                log.info("GLM stream cancelled (session=%s)", session_hash)
                return result or ""
            except Exception as e:
                log.error("Stream error (session=%s): %s", session_hash, e)
                return result or ""

        return result
