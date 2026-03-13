from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import numpy as np

from app.agents.base import Agent, AgentResult
from app.core.llm import call_llm, parse_json_response
from app.models.job import ThicknessMethod, ThicknessResult

logger = logging.getLogger("patchforge.agents.thickness")

ROLE = (
    "You are the thickness decision engine in a photo-to-3D-print patch pipeline. "
    "The target printer is a Bambu Lab A1 (build volume: 256 x 256 x 256 mm). "
    "Thickness means the DEPTH of the replacement patch — how thick the wall/shell "
    "was where it broke. This is NOT the width or height of the gap. "
    "Multiple thickness estimation strategies may produce independent estimates: "
    "LiDAR depth difference, video multi-view stereo, side-photo analysis, "
    "monocular depth inference, and/or vision-based proportional reasoning. "
    "YOUR JOB is to decide which estimate to trust, whether to blend them, and "
    "what the final authoritative patch thickness should be. Base your decision "
    "ONLY on evidence from the measurement methods. Do NOT infer thickness from "
    "material type or generic object assumptions. Physically plausible patch "
    "thickness is typically 1-30mm. The minimum printable wall is 0.8mm. "
    "You MUST pick one final answer."
)

CONSENSUS_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_method": {"type": "string"},
        "final_thickness_mm": {"type": "number"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "should_proceed": {"type": "boolean"},
    },
    "required": ["chosen_method", "final_thickness_mm", "confidence", "reasoning", "should_proceed"],
}


class ThicknessAgent(Agent):
    def __init__(self):
        super().__init__("ThicknessAgent", ROLE)

    async def run(
        self,
        mask: np.ndarray,
        scale_factor: float,
        measurement_width_mm: float,
        measurement_height_mm: float,
        original_upload_path: Optional[str] = None,
        key_frame_paths: Optional[list[str]] = None,
        side_image_path: Optional[str] = None,
        manual_hint_mm: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None,
        vision_thickness_mm: Optional[float] = None,
        calibration_method: str = "unknown",
        image_bgr: Optional[np.ndarray] = None,
    ) -> tuple[ThicknessResult, AgentResult]:
        """
        Run ALL thickness strategies, then let the LLM arbitrate.

        Accepts depth_map from calibration stage to avoid redundant extraction.
        Accepts vision_thickness_mm from the vision measurement pass.
        """
        from app.pipeline.thickness_estimation import estimate_thickness_all

        all_results = estimate_thickness_all(
            original_upload_path=original_upload_path,
            mask=mask,
            scale_factor=scale_factor,
            measurement_width_mm=measurement_width_mm,
            measurement_height_mm=measurement_height_mm,
            key_frame_paths=key_frame_paths,
            side_image_path=side_image_path,
            manual_hint_mm=manual_hint_mm,
            depth_map=depth_map,
            image_bgr=image_bgr,
        )

        # Include vision measurement thickness as an additional candidate
        if vision_thickness_mm is not None and vision_thickness_mm > 0:
            all_results.append(ThicknessResult(
                thickness_mm=round(vision_thickness_mm, 2),
                method=ThicknessMethod.VISION_ESTIMATE,
                confidence=0.55,
                depth_map_used=False,
                num_views_used=0,
            ))

        if not all_results:
            from app.config import settings
            default = manual_hint_mm if manual_hint_mm and manual_hint_mm > 0 else settings.default_thickness_mm
            fallback = ThicknessResult(
                thickness_mm=default,
                method=ThicknessMethod.MANUAL,
                confidence=0.0,
            )
            analysis = AgentResult(
                success=True,
                data={"candidates": []},
                reasoning=f"No thickness strategy produced a result. Using default {default}mm. Upload a side photo or video for better estimation.",
                suggestions=["Upload a side-angle photo of the break", "Record a short video walking around the object"],
                confidence=0.0,
            )
            return fallback, analysis

        if len(all_results) == 1:
            result = all_results[0]
            analysis = await self._validate_single(result, measurement_width_mm, measurement_height_mm)
            return result, analysis

        # Multiple results — LLM consensus
        chosen, analysis = await self._consensus(
            all_results, measurement_width_mm, measurement_height_mm, calibration_method,
        )
        return chosen, analysis

    async def _validate_single(
        self, result: ThicknessResult, width_mm: float, height_mm: float,
    ) -> AgentResult:
        context = {
            "candidates": [self._result_to_dict(result)],
            "num_strategies_succeeded": 1,
            "measurement_width_mm": width_mm,
            "measurement_height_mm": height_mm,
        }
        return await self.analyze(context)

    async def _consensus(
        self,
        results: list[ThicknessResult],
        width_mm: float,
        height_mm: float,
        calibration_method: str,
    ) -> tuple[ThicknessResult, AgentResult]:
        candidates = [self._result_to_dict(r) for r in results]

        prompt = (
            f"THICKNESS CONSENSUS REQUIRED.\n\n"
            f"Damage dimensions: {width_mm:.1f} x {height_mm:.1f} mm\n"
            f"Calibration method: {calibration_method}\n\n"
            f"{len(results)} thickness strategies produced estimates:\n\n"
        )
        for i, c in enumerate(candidates, 1):
            prompt += (
                f"  Strategy {i}: {c['method']}\n"
                f"    thickness_mm: {c['thickness_mm']:.2f}\n"
                f"    confidence: {c['confidence']}\n"
                f"    depth_map_used: {c['depth_map_used']}\n"
                f"    num_views: {c['num_views_used']}\n\n"
            )

        thicknesses = [r.thickness_mm for r in results]
        spread = max(thicknesses) - min(thicknesses) if len(thicknesses) > 1 else 0
        prompt += f"Spread: {spread:.2f} mm (max - min)\n"

        prompt += (
            f"\nDecision guidelines:\n"
            f"- The break is {width_mm:.1f}mm wide — thickness must be physically plausible relative to this\n"
            f"- Do NOT guess thickness from material type or generic assumptions\n"
            f"- Only trust methods that have direct physical evidence (side photo, LiDAR depth, "
            f"visible break edge with proportional reasoning against a reference object)\n"
            f"- Vision estimates based on proportional reasoning with a reference object are the most trustworthy\n"
            f"- Monocular depth estimates (confidence < 0.4) should only be used as a last resort\n"
            f"- If the highest confidence candidate is still below 0.3, report low confidence honestly\n"
            f"- When in doubt, prefer the estimate from the method with the most direct physical evidence\n\n"
            f"Respond with a JSON object containing your decision:\n"
            f'- "chosen_method": which method to trust (or "blended" if averaging)\n'
            f'- "final_thickness_mm": the authoritative thickness in mm\n'
            f'- "confidence": your confidence (0.0-1.0)\n'
            f'- "reasoning": explain your decision\n'
            f'- "suggestions": actionable suggestions\n'
            f'- "should_proceed": whether pipeline should continue\n'
        )

        try:
            text, provider = await asyncio.to_thread(
                call_llm,
                self.role,
                prompt,
                CONSENSUS_SCHEMA,
            )

            parsed = parse_json_response(text)
            final_thickness = float(parsed.get("final_thickness_mm", results[0].thickness_mm))
            chosen_method = parsed.get("chosen_method", results[0].method.value)
            confidence = float(parsed.get("confidence", 0.5))

            final_thickness = max(0.5, min(50.0, final_thickness))

            if chosen_method == "blended":
                method_enum = ThicknessMethod.VISION_ESTIMATE
            else:
                method_enum = next(
                    (r.method for r in results if r.method.value == chosen_method),
                    results[0].method,
                )

            chosen_result = ThicknessResult(
                thickness_mm=round(final_thickness, 2),
                method=method_enum,
                confidence=round(confidence, 2),
                depth_map_used=any(r.depth_map_used for r in results),
                num_views_used=max(r.num_views_used for r in results),
            )

            raw_suggestions = parsed.get("suggestions", [])
            if isinstance(raw_suggestions, str):
                raw_suggestions = [raw_suggestions] if raw_suggestions else []

            analysis = AgentResult(
                success=parsed.get("should_proceed", True),
                data={"candidates": candidates, "chosen": chosen_method},
                reasoning=parsed.get("reasoning", ""),
                suggestions=raw_suggestions,
                confidence=confidence,
            )

            logger.info(
                "Thickness consensus: %s -> %.2f mm (conf=%.2f) from %d candidates",
                chosen_method, final_thickness, confidence, len(results),
            )
            return chosen_result, analysis

        except Exception as e:
            logger.warning("LLM thickness consensus failed: %s — using highest-confidence result", e)
            best = max(results, key=lambda r: r.confidence)
            analysis = AgentResult(
                success=True,
                data={"candidates": candidates},
                reasoning=f"LLM consensus unavailable ({e}). Using highest-confidence result: {best.method.value}.",
                suggestions=[],
                confidence=best.confidence,
            )
            return best, analysis

    @staticmethod
    def _result_to_dict(r: ThicknessResult) -> dict:
        return {
            "method": r.method.value,
            "thickness_mm": r.thickness_mm,
            "confidence": r.confidence,
            "depth_map_used": r.depth_map_used,
            "num_views_used": r.num_views_used,
        }
