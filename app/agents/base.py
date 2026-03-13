from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

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
    or OpenAI, with automatic fallback) to analyze results and produce
    reasoning + suggestions.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    async def analyze(self, context: dict) -> AgentResult:
        """Ask the configured LLM to reason about the pipeline stage output."""
        if not is_llm_available():
            return AgentResult(
                success=True,
                data=context,
                reasoning=f"[{self.name}] No LLM configured — skipping AI analysis.",
                confidence=context.get("confidence", 1.0),
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

            return AgentResult(
                success=parsed.get("should_proceed", True),
                data=context,
                reasoning=parsed.get("reasoning", ""),
                suggestions=parsed.get("suggestions", []),
                confidence=parsed.get("confidence", context.get("confidence", 1.0)),
            )

        except Exception as e:
            logger.warning("LLM analysis failed for %s: %s", self.name, e)
            return AgentResult(
                success=True,
                data=context,
                reasoning=f"[{self.name}] AI analysis unavailable: {e}",
                confidence=context.get("confidence", 1.0),
            )

    async def analyze_with_vision(
        self,
        context: dict,
        image_bytes: bytes,
        vision_prompt: str,
        mime_type: str = "image/jpeg",
    ) -> AgentResult:
        """Ask the LLM to reason about results while also looking at the image."""
        if not is_llm_available():
            return AgentResult(
                success=True,
                data=context,
                reasoning=f"[{self.name}] No LLM configured — skipping vision analysis.",
                confidence=context.get("confidence", 1.0),
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

            return AgentResult(
                success=parsed.get("should_proceed", True),
                data=context,
                reasoning=parsed.get("reasoning", ""),
                suggestions=parsed.get("suggestions", []),
                confidence=parsed.get("confidence", context.get("confidence", 1.0)),
            )

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
            "Analyze these results. Respond with JSON containing:\n"
            '- "reasoning": your analysis of the results (1-3 sentences)\n'
            '- "suggestions": list of actionable suggestions for the user (0-3 items)\n'
            '- "confidence": your confidence in the results (0.0-1.0)\n'
            '- "should_proceed": whether the pipeline should continue (true/false)\n'
        )
