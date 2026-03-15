"""Multi-provider LLM client for the aawaaz fine-tuning pipeline.

Provides a uniform ``generate(messages, **kwargs) → str`` interface over:
- ``anthropic`` — native Anthropic SDK
- ``openai`` — native OpenAI SDK
- ``openai_compatible`` — OpenAI SDK with custom ``base_url``
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("aawaaz.llm_client")

# ── Provider implementations ───────────────────────────────────────────────


class LLMClient:
    """Unified LLM client wrapping Anthropic and OpenAI SDKs."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env: str,
        base_url: str | None = None,
        max_retries: int = 3,
        initial_backoff: float = 2.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key environment variable '{api_key_env}' is not set. "
                f"Export it before running: export {api_key_env}=<your-key>"
            )

        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: uv pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key)
            self._generate = self._generate_anthropic
        elif provider in ("openai", "openai_compatible"):
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed. "
                    "Run: uv pip install openai"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": api_key}
            if provider == "openai_compatible" and base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
            self._generate = self._generate_openai
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                "Must be 'anthropic', 'openai', or 'openai_compatible'."
            )

        logger.info(
            "LLM client ready: provider=%s model=%s", provider, model
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> str:
        """Send a chat completion request with exponential backoff retry.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
        max_tokens:
            Maximum tokens in the response.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            The assistant's response text.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait = self.initial_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"API call failed after {self.max_retries} retries: {last_exc}"
        ) from last_exc

    # ── Provider-specific implementations ──────────────────────────────────

    def _generate_anthropic(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call Anthropic's Messages API.

        Handles system messages by extracting them into the ``system``
        parameter, since Anthropic's API requires system content separately.
        """
        system_parts: list[str] = []
        non_system: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                non_system.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _generate_openai(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call OpenAI (or compatible) Chat Completions API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.choices[0].message.content


def create_client_from_config(
    provider: str,
    model: str,
    api_key_env: str,
    base_url: str | None = None,
    max_retries: int = 3,
) -> LLMClient:
    """Factory that creates an ``LLMClient`` from config values.

    This is the primary entry point — scripts pass the relevant config
    fields and get back a ready-to-use client.
    """
    return LLMClient(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        max_retries=max_retries,
    )
