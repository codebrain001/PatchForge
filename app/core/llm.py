"""
Unified LLM client for PatchForge.

Supports Gemini and OpenAI with automatic fallback.
Provider is selected via LLM_PROVIDER in .env:
  - "gemini"  : Gemini only
  - "openai"  : OpenAI only
  - "auto"    : try Gemini first, fall back to OpenAI on failure
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("patchforge.llm")

_gemini_client = None
_openai_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google.genai import Client
        _gemini_client = Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _available_providers() -> list[str]:
    """Return ordered list of providers to try based on config."""
    provider = settings.llm_provider.lower().strip()
    if provider == "gemini":
        return ["gemini"] if settings.gemini_api_key else []
    elif provider == "openai":
        return ["openai"] if settings.openai_api_key else []
    else:  # "auto"
        providers = []
        if settings.gemini_api_key:
            providers.append("gemini")
        if settings.openai_api_key:
            providers.append("openai")
        return providers


def is_llm_available() -> bool:
    return len(_available_providers()) > 0


def get_active_provider() -> str:
    """Return the name of the provider that will be tried first, or 'none'."""
    providers = _available_providers()
    return providers[0] if providers else "none"


def _call_gemini(
    system: str,
    prompt: str,
    json_schema: Optional[dict] = None,
) -> str:
    """Call Gemini and return raw text response."""
    from google.genai import types

    client = _get_gemini_client()
    config_kwargs: dict[str, Any] = {"system_instruction": system}

    if json_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = json_schema

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text


def _call_openai(
    system: str,
    prompt: str,
    json_schema: Optional[dict] = None,
) -> str:
    """Call OpenAI and return raw text response."""
    client = _get_openai_client()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "messages": messages,
        "temperature": 0.3,
    }

    if json_schema:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_gemini_vision(
    system: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> str:
    """Call Gemini with an image and return raw text response."""
    from google.genai import types

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Content(parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]),
        ],
    )
    return response.text


def _call_openai_vision(
    system: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> str:
    """Call OpenAI with an image and return raw text response."""
    client = _get_openai_client()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── Public API ───────────────────────────────────────────────────


def call_llm(
    system: str,
    prompt: str,
    json_schema: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Call the configured LLM provider(s) with a text prompt.

    Returns:
        (response_text, provider_used)

    Raises:
        RuntimeError if no provider is available or all providers fail.
    """
    providers = _available_providers()
    if not providers:
        raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

    last_error = None
    for provider in providers:
        try:
            if provider == "gemini":
                text = _call_gemini(system, prompt, json_schema)
            else:
                text = _call_openai(system, prompt, json_schema)
            logger.debug("LLM call succeeded via %s", provider)
            return text, provider
        except Exception as e:
            last_error = e
            logger.warning("LLM call failed via %s: %s", provider, e)
            if len(providers) > 1:
                logger.info("Falling back to next provider...")

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def call_llm_vision(
    system: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> tuple[str, str]:
    """
    Call the configured LLM provider(s) with a text + image prompt.

    Returns:
        (response_text, provider_used)

    Raises:
        RuntimeError if no provider is available or all providers fail.
    """
    providers = _available_providers()
    if not providers:
        raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

    last_error = None
    for provider in providers:
        try:
            if provider == "gemini":
                text = _call_gemini_vision(system, prompt, image_bytes, mime_type)
            else:
                text = _call_openai_vision(system, prompt, image_bytes, mime_type)
            logger.debug("LLM vision call succeeded via %s", provider)
            return text, provider
        except Exception as e:
            last_error = e
            logger.warning("LLM vision call failed via %s: %s", provider, e)
            if len(providers) > 1:
                logger.info("Falling back to next provider...")

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def parse_json_response(text: str | None) -> dict:
    """Parse a JSON response from an LLM, handling markdown code fences.

    Handles None, empty strings, markdown-wrapped JSON, and bare JSON.
    Raises ValueError with a clear message on unparseable input.
    """
    if not text:
        raise ValueError("LLM returned empty response — cannot parse JSON.")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("LLM returned whitespace-only response — cannot parse JSON.")

    if cleaned.startswith("```"):
        lines = cleaned.split("\n", 1)
        if len(lines) < 2:
            raise ValueError(f"LLM returned malformed code fence with no content: {cleaned[:100]}")
        cleaned = lines[1].rsplit("```", 1)[0].strip()

    if not cleaned:
        raise ValueError(f"LLM returned empty code fence: {text[:200]}")

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        preview = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
        raise ValueError(
            f"LLM returned invalid JSON: {e}. Response preview: {preview}"
        ) from e

    if not isinstance(result, dict):
        raise ValueError(f"LLM returned {type(result).__name__} instead of JSON object: {cleaned[:200]}")

    return result
