from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.exceptions import CalibrationError
from app.models.job import CalibrationResult
from app.pipeline.depth_extraction import extract_depth_map, calibrate_from_depth


ARUCO_DICT = cv2.aruco.DICT_4X4_50


def detect_aruco_markers(
    image: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    return corners, ids


def _marker_side_length_px(corners: np.ndarray) -> float:
    pts = corners.reshape(4, 2)
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(sides))


def calibrate_aruco(
    image: np.ndarray,
    marker_size_mm: float = 40.0,
) -> Optional[CalibrationResult]:
    """Try ArUco marker detection. Returns None if no marker found."""
    corners, ids = detect_aruco_markers(image)

    if corners is None or ids is None or len(corners) == 0:
        return None

    best_idx = 0
    best_side_px = 0.0
    for i, c in enumerate(corners):
        side_px = _marker_side_length_px(c)
        if side_px > best_side_px:
            best_side_px = side_px
            best_idx = i

    if best_side_px < 10:
        return None

    scale_factor = marker_size_mm / best_side_px

    sharpness = cv2.Laplacian(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
        cv2.CV_64F,
    ).var()
    confidence = min(1.0, sharpness / 500.0) * min(1.0, best_side_px / 80.0)

    return CalibrationResult(
        scale_factor=scale_factor,
        method="aruco",
        marker_id=int(ids[best_idx][0]),
        confidence=round(confidence, 3),
    )


def calibrate_manual(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    real_mm: float,
) -> CalibrationResult:
    """Calibrate from a user-drawn reference line between two pixel coords."""
    pixel_dist = math.dist(point_a, point_b)
    if pixel_dist < 5:
        raise CalibrationError("Reference line is too short (< 5 px).")
    return CalibrationResult(
        scale_factor=real_mm / pixel_dist,
        method="reference_line",
        confidence=0.85,
    )


def calibrate_webxr(scale_mm_per_px: float) -> CalibrationResult:
    """Accept a pre-computed scale factor from WebXR AR measurement."""
    if scale_mm_per_px <= 0 or scale_mm_per_px > 10.0:
        raise CalibrationError(
            f"WebXR scale {scale_mm_per_px} mm/px is out of reasonable bounds (0–10 mm/px)."
        )
    return CalibrationResult(
        scale_factor=scale_mm_per_px,
        method="webxr",
        confidence=0.80,
    )


def calibrate_all(
    image: np.ndarray,
    marker_size_mm: float = 40.0,
    original_upload_path: Optional[str] = None,
    ref_line_start: Optional[tuple[int, int]] = None,
    ref_line_end: Optional[tuple[int, int]] = None,
    ref_line_mm: Optional[float] = None,
    webxr_scale: Optional[float] = None,
) -> tuple[list[CalibrationResult], Optional[np.ndarray]]:
    """
    Run ALL applicable calibration strategies and return every result.

    The LLM decides which result to trust — this function just collects evidence.
    Also returns the extracted depth map (if any) for downstream thickness estimation.

    Returns:
        (list of CalibrationResults, depth_map or None)
    """
    results: list[CalibrationResult] = []
    depth_map: Optional[np.ndarray] = None

    # Strategy 1: HEIF depth extraction
    if original_upload_path:
        ext = Path(original_upload_path).suffix.lower()
        if ext in (".heic", ".heif"):
            depth_map = extract_depth_map(original_upload_path)
            if depth_map is not None:
                h, w = image.shape[:2]
                result = calibrate_from_depth(original_upload_path, depth_map, w, h)
                if result is not None:
                    results.append(result)

    # Strategy 2: ArUco marker detection (always attempted)
    aruco_result = calibrate_aruco(image, marker_size_mm)
    if aruco_result is not None:
        results.append(aruco_result)

    # Strategy 3: WebXR pre-computed scale
    if webxr_scale is not None:
        try:
            results.append(calibrate_webxr(webxr_scale))
        except CalibrationError:
            pass

    # Strategy 4: User-drawn reference line
    if ref_line_start and ref_line_end and ref_line_mm:
        try:
            results.append(calibrate_manual(
                tuple(ref_line_start), tuple(ref_line_end), ref_line_mm
            ))
        except CalibrationError:
            pass

    return results, depth_map


def calibrate(
    image: np.ndarray,
    marker_size_mm: float = 40.0,
    original_upload_path: Optional[str] = None,
    ref_line_start: Optional[tuple[int, int]] = None,
    ref_line_end: Optional[tuple[int, int]] = None,
    ref_line_mm: Optional[float] = None,
    webxr_scale: Optional[float] = None,
) -> CalibrationResult:
    """Legacy single-result wrapper. Prefer calibrate_all() + LLM consensus."""
    results, _ = calibrate_all(
        image, marker_size_mm, original_upload_path,
        ref_line_start, ref_line_end, ref_line_mm, webxr_scale,
    )
    if not results:
        raise CalibrationError(
            "No calibration method succeeded. "
            "Please draw a reference line on a known object, or include an ArUco marker."
        )
    return max(results, key=lambda r: r.confidence)
