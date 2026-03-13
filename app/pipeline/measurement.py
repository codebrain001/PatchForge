from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import MeasurementError
from app.models.job import MeasurementResult


def measure(
    contours: list[np.ndarray],
    scale_factor: float,
    calibration_confidence: float = 1.0,
) -> MeasurementResult:
    """
    Measure dimensions of the largest contour in real-world mm.

    Args:
        contours: list of OpenCV contours (sorted largest-first)
        scale_factor: mm per pixel from calibration
        calibration_confidence: how reliable the scale factor is (0-1)
    """
    if not contours:
        raise MeasurementError("No contours provided for measurement.")

    if scale_factor <= 0:
        raise MeasurementError(
            f"Invalid scale factor: {scale_factor} mm/px. "
            "Calibration must produce a positive scale factor before measurement."
        )

    contour = contours[0]

    if cv2.contourArea(contour) < 100:
        raise MeasurementError(
            "Primary contour is too small (< 100 px^2). "
            "Click closer to the damaged area."
        )

    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, closed=True)

    x, y, w_bbox, h_bbox = cv2.boundingRect(contour)

    # Rotated bounding rectangle gives tighter width/height for non-axis-aligned damage
    rect = cv2.minAreaRect(contour)
    (_, _), (rect_w, rect_h), _ = rect
    w_tight = min(rect_w, rect_h)
    h_tight = max(rect_w, rect_h)

    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    sf = scale_factor
    sf2 = scale_factor ** 2

    return MeasurementResult(
        width_mm=round(w_tight * sf, 2),
        height_mm=round(h_tight * sf, 2),
        area_mm2=round(area_px * sf2, 2),
        perimeter_mm=round(perimeter_px * sf, 2),
        bounding_rect_mm=[
            round(x * sf, 2),
            round(y * sf, 2),
            round(w_bbox * sf, 2),
            round(h_bbox * sf, 2),
        ],
        min_enclosing_radius_mm=round(radius * sf, 2),
        confidence=round(calibration_confidence, 2),
    )
