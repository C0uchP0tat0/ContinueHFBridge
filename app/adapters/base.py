"""Abstract base class for model adapters.

To add a new model backend:
  1. Create a new file in app/adapters/ (e.g. my_model.py).
  2. Subclass ModelAdapter and implement `call()`.
  3. Register it in app/adapters/registry.py.
  4. Set ADAPTER=my_model in env.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class ModelAdapter(ABC):
    """Interface every model backend must implement."""

    @abstractmethod
    async def call(
        self,
        prompt: str,
        *,
        retry_hint: str = "",
        cancel_event: Optional[asyncio.Event] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send *prompt* to the model and return the raw text response.

        Parameters
        ----------
        prompt:
            The fully-assembled prompt (transcript + system).
        retry_hint:
            Extra instruction prepended on retry attempts.
        cancel_event:
            When set, the adapter should abort ASAP and return whatever
            partial result it has (or ``""``).
        system_prompt:
            Override the default system prompt for this call.

        Returns
        -------
        str  — raw model output (may contain HTML, JSON, markdown, etc.).
              Cleaning is done by the caller.
        """
        ...
