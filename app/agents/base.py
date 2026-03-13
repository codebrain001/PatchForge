from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, Field

from app.core.llm import call_llm, call_llm_vision, is_llm_available, parse_json_response

logger = logging.getLogger("patchforge.agents")

AGENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "number"},
        "should_proceed": {"type": "boolean"},
    },
    "required": ["reasoning", "should_proceed"],
}


class AgentResult(BaseModel):
    success: bool
    data: dict = Field(default_factory=dict)
    reasoning: str = ""
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class Agent:
    """
    Base class for PatchForge pipeline agents.

    Each agent wraps a domain pipeline function and uses an LLM (Gemini
    or OpenAI, with automatic fallback) as its core decision engine.
    The LLM is not optional — it IS the thinking layer that analyzes
    results, arbitrates between competing strategies, and decides
    whether the pipeline should proceed.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    async def analyze(self, context: dict) -> AgentResult:
        """Ask the LLM to reason about the pipeline stage output.

        Raises RuntimeError if no LLM is configured — the LLM is the
        decision engine, not an optional enhancement.
        """
        if not is_llm_available():
            raise RuntimeError(
                f"{self.name} requires an LLM to make decisions. "
                "Set GEMINI_API_KEY or OPENAI_API_KEY in .env — "
                "the LLM is the core thinking engine of this pipeline."
            )

        prompt = self._build_prompt(context)

        try:
            text, provider = await asyncio.to_thread(
                call_llm,
                self.role,
                prompt,
                AGENT_JSON_SCHEMA,
            )

            parsed = parse_json_response(text)

            raw_suggestions = parsed.get("suggestions", [])
            if isinstance(raw_suggestions, str):
                raw_suggestions = [raw_suggestions] if raw_suggestions else []

            return AgentResult(
                success=parsed.get("should_proceed", True),
                data=context,
                reasoning=parsed.get("reasoning", ""),
                suggestions=raw_suggestions,
                confidence=parsed.get("confidence", context.get("confidence", 1.0)),
            )

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("LLM analysis failed for %s: %s", self.name, e)
            raise RuntimeError(
                f"{self.name} LLM call failed: {e}. "
                "Check your API key and network connection."
            ) from e

    async def analyze_with_vision(
        self,
        context: dict,
        image_bytes: bytes,
        vision_prompt: str,
        mime_type: str = "image/jpeg",
    ) -> AgentResult:
        """Ask the LLM to reason about results while also looking at the image."""
        if not is_llm_available():
            raise RuntimeError(
                f"{self.name} requires an LLM for vision analysis. "
                "Set GEMINI_API_KEY or OPENAI_API_KEY in .env."
            )

        full_prompt = self._build_prompt(context) + "\n\n" + vision_prompt

        try:
            text, provider = await asyncio.to_thread(
                call_llm_vision,
                self.role,
                full_prompt,
                image_bytes,
                mime_type,
            )

            parsed = parse_json_response(text)

            raw_suggestions = parsed.get("suggestions", [])
            if isinstance(raw_suggestions, str):
                raw_suggestions = [raw_suggestions] if raw_suggestions else []

            return AgentResult(
                success=parsed.get("should_proceed", True),
                data=context,
                reasoning=parsed.get("reasoning", ""),
                suggestions=raw_suggestions,
                confidence=parsed.get("confidence", context.get("confidence", 1.0)),
            )

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("LLM vision analysis failed for %s: %s", self.name, e)
            return await self.analyze(context)

    def _build_prompt(self, context: dict) -> str:
        sanitized = {}
        for k, v in context.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                sanitized[k] = v
            elif isinstance(v, list) and len(v) < 20:
                sanitized[k] = v
            else:
                sanitized[k] = str(type(v).__name__)

        return (
            f"You are the {self.name} agent in a photo-to-3D-print repair pipeline.\n\n"
            f"Stage results:\n{json.dumps(sanitized, indent=2, default=str)}\n\n"
            "Analyze these results. Respond with a JSON object containing:\n"
            '- "reasoning": your analysis of the results (1-3 sentences)\n'
            '- "suggestions": array of actionable suggestions for the user (0-3 items)\n'
            '- "confidence": your confidence in the results (0.0-1.0)\n'
            '- "should_proceed": whether the pipeline should continue (true/false)\n'
        )
