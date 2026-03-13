from __future__ import annotations

from typing import Optional

import numpy as np

from app.agents.base import Agent, AgentResult
from app.models.job import ThicknessResult

ROLE = (
    "You are a depth/thickness estimation expert for 3D-printable repair parts. "
    "You evaluate automatically inferred thickness values for plausibility. "
    "Consider the damage dimensions: a break that is 20mm wide is unlikely to be "
    "0.5mm deep. Similarly, thickness over 30mm is unusual for repair patches. "
    "If confidence is low, suggest the user upload a side photo or manually adjust. "
    "For video-based estimates, more views increase reliability."
)


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
    ) -> tuple[ThicknessResult, AgentResult]:
        from app.pipeline.thickness_estimation import estimate_thickness

        result = estimate_thickness(
            original_upload_path=original_upload_path,
            mask=mask,
            scale_factor=scale_factor,
            measurement_width_mm=measurement_width_mm,
            measurement_height_mm=measurement_height_mm,
            key_frame_paths=key_frame_paths,
            side_image_path=side_image_path,
            manual_hint_mm=manual_hint_mm,
        )

        context = {
            "thickness_mm": result.thickness_mm,
            "method": result.method.value,
            "confidence": result.confidence,
            "depth_map_used": result.depth_map_used,
            "num_views_used": result.num_views_used,
            "measurement_width_mm": measurement_width_mm,
            "measurement_height_mm": measurement_height_mm,
            "had_video_frames": bool(key_frame_paths and len(key_frame_paths) >= 2),
            "had_side_photo": bool(side_image_path),
            "had_lidar": bool(
                original_upload_path
                and original_upload_path.lower().endswith((".heic", ".heif"))
            ),
        }

        analysis = await self.analyze(context)
        return result, analysis
