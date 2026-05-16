"""Transcript builder — converts OpenAI/Ollama message lists into a flat prompt."""
import json
from typing import List, Optional, Dict, Any

from app.config import MAX_TRANSCRIPT_CHARS


def _message_content(m: Dict[str, Any]) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


def build_transcript_for_llm(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
) -> str:
    lines: List[str] = []
    if tools:
        lines.append(
            "AVAILABLE_TOOLS_JSON:\n"
            + json.dumps(tools, ensure_ascii=False, indent=2)
        )
    for m in messages:
        role = m.get("role") or "user"
        if role == "tool":
            name = m.get("name") or "tool"
            lines.append(f"TOOL_RESULT name={name}:\n{_message_content(m)}")
        elif role == "assistant" and m.get("tool_calls"):
            lines.append(
                f"assistant (previous tool_calls JSON):\n{json.dumps(m.get('tool_calls'), ensure_ascii=False)}"
            )
            if _message_content(m).strip():
                lines.append(f"assistant (text):\n{_message_content(m)}")
        else:
            lines.append(f"{role}:\n{_message_content(m)}")
    result = "\n\n".join(lines)
    # Truncate from the beginning if too long, keeping recent messages
    if len(result) > MAX_TRANSCRIPT_CHARS:
        result = result[-MAX_TRANSCRIPT_CHARS:]
        idx = result.find("\n\n")
        if idx > 0:
            result = "...(earlier messages truncated)\n\n" + result[idx + 2 :]
    return result


def last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if (m.get("role") or "") == "user":
            return _message_content(m)
    return ""
