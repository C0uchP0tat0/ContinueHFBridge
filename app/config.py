import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("continue_adapter")

# ── Model / adapter selection ──────────────────────────────────────────
# Set ADAPTER env-var to switch backends: "glm" (default), "openai", …
ADAPTER_NAME: str = os.getenv("ADAPTER", "glm")
MODEL_NAME: str = os.getenv("MODEL_NAME", "glm-4-5")

# ── Transcript limits ─────────────────────────────────────────────────
MAX_TRANSCRIPT_CHARS: int = int(os.getenv("MAX_TRANSCRIPT_CHARS", "14000"))

# ── GLM-specific (used only by app.adapters.glm) ─────────────────────
GLM_TIMEOUT_TOTAL: int = int(os.getenv("GLM_TIMEOUT_TOTAL", "150"))
GLM_TIMEOUT_PER_LINE: int = int(os.getenv("GLM_TIMEOUT_PER_LINE", "45"))
GLM_MAX_RETRIES: int = int(os.getenv("GLM_MAX_RETRIES", "1"))
GLM_ENDPOINT: str = os.getenv(
    "GLM_ENDPOINT",
    "https://zai-org-glm-4-5-space.hf.space/gradio_api",
)

# ── Qwen-specific (used only by app.adapters.qwen) ───────────────────
QWEN_ENDPOINT: str = os.getenv(
    "QWEN_ENDPOINT",
    "https://qwen-qwen3-vl-235b-a22b-instruct-demo.hf.space/gradio_api",
)
QWEN_TIMEOUT_TOTAL: int = int(os.getenv("QWEN_TIMEOUT_TOTAL", "180"))
QWEN_TIMEOUT_PER_LINE: int = int(os.getenv("QWEN_TIMEOUT_PER_LINE", "60"))
QWEN_ZEROGPU_TOKEN: str = os.getenv("QWEN_ZEROGPU_TOKEN", "")
QWEN_ZEROGPU_UUID: str = os.getenv("QWEN_ZEROGPU_UUID", "")

# ── Fallback mode (skip remote model entirely) ────────────────────────
FALLBACK_MODE: bool = os.getenv("FALLBACK_MODE", "false").lower() in ("1", "true", "yes")

# ── Server ─────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "11434"))
