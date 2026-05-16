"""Adapter registry — maps ADAPTER_NAME → concrete ModelAdapter."""
from __future__ import annotations

import importlib
from typing import Dict, Type

from app.adapters.base import ModelAdapter
from app.config import ADAPTER_NAME, log

# ── Adapter class registry (lazy import paths) ────────────────────────

_REGISTRY: Dict[str, str] = {
    "glm": "app.adapters.glm:GLMAdapter",
    "qwen": "app.adapters.qwen:QwenAdapter",
    # Add new adapters here:
    # "openai": "app.adapters.openai_adapter:OpenAIAdapter",
}

# ── Model name → adapter key mapping ──────────────────────────────────
# Continue sends model names like "glm-4-5" or "qwen3-vl-235b" in requests.
# This map resolves them to the correct adapter key.
_MODEL_TO_ADAPTER: Dict[str, str] = {
    "glm-4-5": "glm",
    "glm-4": "glm",
    "glm": "glm",
    "qwen3-vl-235b": "qwen",
    "qwen3-vl": "qwen",
    "qwen": "qwen",
}

# ── Singleton cache (one instance per adapter key) ────────────────────
_instances: Dict[str, ModelAdapter] = {}


def _load_adapter(key: str) -> ModelAdapter:
    """Instantiate and cache an adapter by registry key."""
    if key in _instances:
        return _instances[key]

    entry = _REGISTRY.get(key)
    if entry is None:
        raise RuntimeError(
            f"Unknown adapter '{key}'. "
            f"Available: {', '.join(_REGISTRY.keys())}"
        )

    module_path, class_name = entry.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls: Type[ModelAdapter] = getattr(module, class_name)
    inst = cls()
    _instances[key] = inst
    log.info("Loaded adapter: %s (%s)", key, cls.__name__)
    return inst


def get_adapter() -> ModelAdapter:
    """Return the default adapter (from ADAPTER env-var)."""
    return _load_adapter(ADAPTER_NAME)


def get_adapter_for_model(model_name: str) -> ModelAdapter:
    """Return the adapter matching a model name from the request.

    Falls back to the default adapter if the model name is unknown.
    """
    log.info("get_adapter_for_model: received model_name=%r", model_name)
    key = _MODEL_TO_ADAPTER.get(model_name)
    if key is None:
        # Try prefix match: "qwen3-vl-235b-instruct" → starts with "qwen3-vl-235b"
        mn_lower = model_name.lower()
        for prefix, adapter_key in _MODEL_TO_ADAPTER.items():
            if mn_lower.startswith(prefix):
                key = adapter_key
                log.info("get_adapter_for_model: prefix match %r -> %s", prefix, adapter_key)
                break
    if key is None:
        log.info("get_adapter_for_model: unknown model %r, using default %s", model_name, ADAPTER_NAME)
        key = ADAPTER_NAME
    log.info("get_adapter_for_model: selected adapter key=%s for model=%r", key, model_name)
    return _load_adapter(key)
