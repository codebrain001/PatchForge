from __future__ import annotations

from typing import Optional

import numpy as np

from app.agents.base import Agent, AgentResult
from app.models.job import CalibrationResult

ROLE = (
    "You are an expert calibration agent specializing in scale estimation from images. "
    "You evaluate calibration results from multiple strategies (HEIF depth extraction, "
    "ArUco marker detection, WebXR AR measurement, user reference line) and assess "
    "accuracy and confidence. You understand camera optics, ArUco marker detection, "
    "and depth sensing. Flag any suspicious results like implausible scale factors."
)


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
    ) -> tuple[CalibrationResult, AgentResult]:
        from app.pipeline.calibration import calibrate

        cal_result = calibrate(
            image,
            marker_size_mm=marker_size_mm,
            original_upload_path=original_upload_path,
            ref_line_start=ref_line_start,
            ref_line_end=ref_line_end,
            ref_line_mm=ref_line_mm,
            webxr_scale=webxr_scale,
        )

        context = {
            "scale_factor_mm_per_px": cal_result.scale_factor,
            "method": cal_result.method,
            "confidence": cal_result.confidence,
            "marker_id": cal_result.marker_id,
            "depth_map_available": cal_result.depth_map_available,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
        }

        analysis = await self.analyze(context)
        return cal_result, analysis
