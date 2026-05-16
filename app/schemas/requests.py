from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict

from app.config import MODEL_NAME


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completions (VS Code Continue with provider: openai)."""

    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_NAME
    messages: List[Dict[str, Any]]
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Any = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class OllamaChatRequest(BaseModel):
    """Ollama /api/chat body (Continue with provider: ollama)."""

    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_NAME
    messages: List[Dict[str, Any]]
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Any = None
    think: Optional[bool] = None


class OllamaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_NAME
    prompt: str = ""
    stream: bool = False
    suffix: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str = ""
    suffix: Optional[str] = ""
    max_tokens: int = 256
    temperature: float = 0.2
    stop: Optional[Any] = None
    stream: bool = False
