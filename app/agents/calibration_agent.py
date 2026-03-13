from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import numpy as np

from app.agents.base import Agent, AgentResult
from app.core.llm import call_llm, parse_json_response
from app.models.job import CalibrationResult

logger = logging.getLogger("patchforge.agents.calibration")

ROLE = (
    "You are the calibration decision engine in a photo-to-3D-print patch pipeline. "
    "The target printer is a Bambu Lab A1 (build volume: 256 x 256 x 256 mm). "
    "Calibration determines the mm-per-pixel scale so we can convert the gap/void "
    "dimensions from pixels to real-world mm for the replacement patch. "
    "Multiple calibration strategies may produce independent scale estimates: "
    "HEIF depth extraction, ArUco marker detection, WebXR AR measurement, and/or "
    "user reference line. YOUR JOB is to decide which estimate to trust, whether "
    "to blend them, and what the final authoritative scale factor should be. "
    "A physically plausible scale is typically 0.01-1.0 mm/px for phone photos. "
    "Values outside 0.001-10 mm/px are almost certainly wrong. "
    "You MUST pick one final answer."
)

CONSENSUS_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_method": {"type": "string"},
        "final_scale_factor": {"type": "number"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "should_proceed": {"type": "boolean"},
    },
    "required": ["chosen_method", "final_scale_factor", "confidence", "reasoning", "should_proceed"],
}


class CalibrationAgent(Agent):
    def __init__(self):
        super().__init__("CalibrationAgent", ROLE)

    async def run(
        self,
        image: np.ndarray,
        marker_size_mm: float = 40.0,
        original_upload_path: Optional[str] = None,
        ref_line_start: Optional[tuple[int, int]] = None,
        ref_line_end: Optional[tuple[int, int]] = None,
        ref_line_mm: Optional[float] = None,
        webxr_scale: Optional[float] = None,
    ) -> tuple[CalibrationResult, AgentResult, Optional[np.ndarray]]:
        """
        Run ALL calibration strategies, then let the LLM arbitrate.

        Returns:
            (chosen CalibrationResult, AgentResult with reasoning, depth_map or None)
        """
        from app.pipeline.calibration import calibrate_all

        all_results, depth_map = calibrate_all(
            image,
            marker_size_mm=marker_size_mm,
            original_upload_path=original_upload_path,
            ref_line_start=ref_line_start,
            ref_line_end=ref_line_end,
            ref_line_mm=ref_line_mm,
            webxr_scale=webxr_scale,
        )

        if not all_results:
            # No algorithmic calibration succeeded — let the LLM try vision-based
            analysis = AgentResult(
                success=False,
                data={"candidates": [], "image_width": image.shape[1], "image_height": image.shape[0]},
                reasoning="No calibration strategy produced a result. Need vision-based estimation or user reference line.",
                suggestions=["Draw a reference line on a known object", "Place an ArUco marker in the scene"],
                confidence=0.0,
            )
            return CalibrationResult(scale_factor=0.0, method="none", confidence=0.0), analysis, depth_map

        if len(all_results) == 1:
            # Single result — still have LLM validate it
            result = all_results[0]
            analysis = await self._validate_single(result, image.shape)
            return result, analysis, depth_map

        # Multiple results — LLM consensus
        chosen, analysis = await self._consensus(all_results, image.shape)
        return chosen, analysis, depth_map

    async def _validate_single(self, result: CalibrationResult, img_shape: tuple) -> AgentResult:
        """Single calibration result — LLM validates plausibility."""
        context = {
            "candidates": [self._result_to_dict(result)],
            "num_strategies_succeeded": 1,
            "image_width": img_shape[1],
            "image_height": img_shape[0],
        }
        return await self.analyze(context)

    async def _consensus(
        self,
        results: list[CalibrationResult],
        img_shape: tuple,
    ) -> tuple[CalibrationResult, AgentResult]:
        """Multiple results — LLM picks the best or blends them."""
        candidates = [self._result_to_dict(r) for r in results]

        prompt = (
            f"CALIBRATION CONSENSUS REQUIRED.\n\n"
            f"Image dimensions: {img_shape[1]}x{img_shape[0]} pixels.\n\n"
            f"{len(results)} calibration strategies produced scale estimates:\n\n"
        )
        for i, c in enumerate(candidates, 1):
            prompt += (
                f"  Strategy {i}: {c['method']}\n"
                f"    scale_factor: {c['scale_factor']:.6f} mm/px\n"
                f"    confidence: {c['confidence']}\n"
                f"    marker_id: {c.get('marker_id', 'N/A')}\n"
                f"    depth_map: {c.get('depth_map_available', False)}\n\n"
            )

        # Check agreement
        scales = [r.scale_factor for r in results]
        max_s, min_s = max(scales), min(scales)
        agreement_ratio = min_s / max_s if max_s > 0 else 0
        prompt += f"Agreement ratio (min/max): {agreement_ratio:.3f}\n"
        if agreement_ratio > 0.9:
            prompt += "The strategies AGREE closely. Consider averaging.\n"
        else:
            prompt += "The strategies DISAGREE. You must choose which to trust and explain why.\n"

        prompt += (
            "\nRespond with a JSON object containing your decision:\n"
            '- "chosen_method": which method to trust (or "blended" if averaging)\n'
            '- "final_scale_factor": the authoritative mm-per-pixel value\n'
            '- "confidence": your confidence in this decision (0.0-1.0)\n'
            '- "reasoning": explain your decision process\n'
            '- "suggestions": actionable suggestions for the user\n'
            '- "should_proceed": whether the pipeline should continue\n'
        )

        try:
            text, provider = await asyncio.to_thread(
                call_llm,
                self.role,
                prompt,
                CONSENSUS_SCHEMA,
            )

            parsed = parse_json_response(text)
            final_scale = float(parsed.get("final_scale_factor", results[0].scale_factor))
            chosen_method = parsed.get("chosen_method", results[0].method)
            confidence = float(parsed.get("confidence", 0.5))

            # Sanity: LLM must return a physically plausible scale (0.001 to 10 mm/px)
            if final_scale <= 0 or final_scale > 10.0:
                logger.warning(
                    "LLM returned implausible scale_factor=%.6f — falling back to best candidate",
                    final_scale,
                )
                best = max(results, key=lambda r: r.confidence)
                final_scale = best.scale_factor
                chosen_method = best.method

            # Find the closest matching result or construct a blended one
            if chosen_method == "blended":
                chosen_result = CalibrationResult(
                    scale_factor=final_scale,
                    method="consensus_blended",
                    confidence=confidence,
                    depth_map_available=any(r.depth_map_available for r in results),
                )
            else:
                # Pick the result matching the chosen method
                chosen_result = next(
                    (r for r in results if r.method == chosen_method),
                    results[0],
                )
                chosen_result = CalibrationResult(
                    scale_factor=final_scale,
                    method=chosen_method,
                    marker_id=chosen_result.marker_id,
                    confidence=confidence,
                    depth_map_available=chosen_result.depth_map_available,
                )

            raw_suggestions = parsed.get("suggestions", [])
            if isinstance(raw_suggestions, str):
                raw_suggestions = [raw_suggestions] if raw_suggestions else []

            analysis = AgentResult(
                success=parsed.get("should_proceed", True),
                data={"candidates": [self._result_to_dict(r) for r in results], "chosen": chosen_method},
                reasoning=parsed.get("reasoning", ""),
                suggestions=raw_suggestions,
                confidence=confidence,
            )

            logger.info(
                "Calibration consensus: %s -> %.6f mm/px (conf=%.2f) from %d candidates",
                chosen_method, final_scale, confidence, len(results),
            )
            return chosen_result, analysis

        except Exception as e:
            logger.warning("LLM consensus failed: %s — falling back to highest-confidence result", e)
            best = max(results, key=lambda r: r.confidence)
            analysis = AgentResult(
                success=True,
                data={"candidates": [self._result_to_dict(r) for r in results]},
                reasoning=f"LLM consensus unavailable ({e}). Using highest-confidence result: {best.method}.",
                suggestions=[],
                confidence=best.confidence,
            )
            return best, analysis

    @staticmethod
    def _result_to_dict(r: CalibrationResult) -> dict:
        return {
            "method": r.method,
            "scale_factor": r.scale_factor,
            "confidence": r.confidence,
            "marker_id": r.marker_id,
            "depth_map_available": r.depth_map_available,
        }
