"""HTML / Gradio output cleanup utilities."""
import re
import html


def clean(text: str) -> str:
    if not text:
        return ""
    # Step 1: Gradio wraps the ENTIRE output in HTML —
    # convert <br> → \n and unescape entities GLOBALLY first
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = html.unescape(text)
    # Step 2: Strip structural HTML outside code fences
    parts = re.split(r"(```[\s\S]*?```)", text)
    cleaned_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside code fence — keep as-is
            cleaned_parts.append(part)
        else:
            # Outside code fence — strip remaining HTML tags from Gradio
            p = re.sub(r"<details.*?</details>", "", part, flags=re.S)
            p = re.sub(r"</?(?:div|span|p|hr|ul|ol|li|table|tr|td|th|thead|tbody|a|img|b|i|em|strong|pre|code|blockquote|h[1-6])\b[^>]*>", "", p, flags=re.I)
            cleaned_parts.append(p)
    return "".join(cleaned_parts).strip()
