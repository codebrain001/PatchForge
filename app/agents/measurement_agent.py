from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.agents.base import Agent, AgentResult
from app.models.job import MeasurementResult

ROLE = (
    "You are a dimensional analysis expert for 3D-printable repair patches. "
    "The target printer is a Bambu Lab A1 (build volume: 256 x 256 x 256 mm). "
    "You evaluate the dimensions of the GAP/VOID left by a broken-off piece — "
    "these dimensions determine the size of the replacement PATCH we will print. "
    "You are NOT measuring the whole object, only the missing piece area. "
    "Common repair patches are typically 5-150mm. "
    "Flag measurements below 2mm or above 200mm as potential calibration errors, "
    "and flag any dimension exceeding 256mm as unprintable on this printer. "
    "When shown an image, verify the mask covers ONLY the gap, not the whole object."
)


class MeasurementAgent(Agent):
    def __init__(self):
        super().__init__("MeasurementAgent", ROLE)

    async def run(
        self,
        contours: list[np.ndarray],
        scale_factor: float,
        calibration_method: str,
        calibration_confidence: float = 1.0,
        image_bgr: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[MeasurementResult, AgentResult]:
        from app.pipeline.measurement import measure

        result = measure(contours, scale_factor, calibration_confidence)

        context = {
            "width_mm": result.width_mm,
            "height_mm": result.height_mm,
            "area_mm2": result.area_mm2,
            "perimeter_mm": result.perimeter_mm,
            "min_enclosing_radius_mm": result.min_enclosing_radius_mm,
            "calibration_method": calibration_method,
            "scale_factor": scale_factor,
        }

        if image_bgr is not None and mask is not None:
            analysis = await self._analyze_with_image(
                context, image_bgr, mask, contours,
            )
        else:
            analysis = await self.analyze(context)
        return result, analysis

    async def _analyze_with_image(
        self,
        context: dict,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        contours: list[np.ndarray],
    ) -> AgentResult:
        """Send the image with mask overlay to the LLM for visual verification."""
        overlay = image_bgr.copy()
        mask_color = np.zeros_like(overlay)
        mask_color[:, :, 1] = 255  # green
        overlay[mask > 127] = cv2.addWeighted(
            overlay[mask > 127], 0.5, mask_color[mask > 127], 0.5, 0,
        )
        cv2.drawContours(overlay, contours[:1], -1, (0, 0, 255), 2)

        # Draw the minimum area rectangle in blue
        if contours:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(overlay, [box], 0, (255, 0, 0), 2)

        h, w = overlay.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            overlay = cv2.resize(overlay, (int(w * s), int(h * s)))

        _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = buf.tobytes()

        vision_prompt = (
            "The attached image shows the broken object with the detected damage "
            "mask highlighted in GREEN, the measurement bounding box in BLUE, "
            "and the RED contour as the detected damage boundary.\n\n"
            "Visually verify:\n"
            "1. Does the green mask cover ONLY the broken/missing area, without "
            "including undamaged surfaces, shadows, or background?\n"
            "2. Does the blue rectangle tightly fit the actual break boundaries?\n"
            "3. Are there visible edges, break lines, or shadows that suggest the "
            "real damage is smaller or larger than what the mask shows?\n"
            "4. Given visible context (nearby objects, reference objects like coins), "
            f"do the computed dimensions ({context['width_mm']:.1f}mm x "
            f"{context['height_mm']:.1f}mm) seem physically plausible?\n\n"
            "If the mask appears too large or includes non-damage areas, lower "
            "your confidence and suggest the user manually click on the damage "
            "center for tighter segmentation."
        )

        return await self.analyze_with_vision(
            context, image_bytes, vision_prompt,
        )
