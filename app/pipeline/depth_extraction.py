"""
Extract depth maps from iPhone HEIF/HEIC portrait-mode photos.

Uses pillow-heif to read the auxiliary depth image embedded by iOS,
then combines it with EXIF focal length to compute a mm-per-pixel scale factor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PIL.ExifTags import Base as ExifBase

from app.models.job import CalibrationResult

logger = logging.getLogger("patchforge.depth")

# Known iPhone sensor widths in mm, keyed by EXIF model substring
_SENSOR_WIDTHS: dict[str, float] = {
    "iPhone 15 Pro Max": 9.8,
    "iPhone 15 Pro": 9.8,
    "iPhone 15": 7.0,
    "iPhone 14 Pro Max": 9.8,
    "iPhone 14 Pro": 9.8,
    "iPhone 14": 7.0,
    "iPhone 13 Pro Max": 7.0,
    "iPhone 13 Pro": 7.0,
    "iPhone 13": 7.0,
    "iPhone 12 Pro Max": 7.0,
    "iPhone 12 Pro": 7.0,
    "iPhone 12": 5.6,
}
_DEFAULT_SENSOR_WIDTH_MM = 6.17


def extract_depth_map(image_path: str | Path) -> Optional[np.ndarray]:
    """
    Try to extract a depth map from a HEIF file.
    Returns a float32 numpy array of per-pixel depth values, or None.
    """
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        logger.debug("pillow-heif not installed — skipping HEIF depth extraction")
        return None

    try:
        im = Image.open(str(image_path))
    except Exception as e:
        logger.warning("Failed to open HEIF image %s: %s", image_path, e)
        return None

    depth_images = im.info.get("depth_images")
    if not depth_images:
        logger.debug("No depth_images in HEIF metadata for %s", image_path)
        return None

    try:
        depth_pil = depth_images[0].to_pillow()
        return np.asarray(depth_pil, dtype=np.float32)
    except Exception as e:
        logger.warning("Failed to convert depth image to array: %s", e)
        return None


def _get_exif_data(image_path: str | Path) -> dict:
    try:
        im = Image.open(str(image_path))
        exif = im.getexif()
        return exif if exif else {}
    except Exception as e:
        logger.warning("Failed to read EXIF from %s: %s", image_path, e)
        return {}


def _estimate_sensor_width(exif: dict) -> float:
    model = exif.get(ExifBase.Model, "")
    for key, width in _SENSOR_WIDTHS.items():
        if key in model:
            return width
    return _DEFAULT_SENSOR_WIDTH_MM


def calibrate_from_depth(
    image_path: str | Path,
    depth_map: np.ndarray,
    image_width_px: int,
    image_height_px: int,
) -> Optional[CalibrationResult]:
    """
    Use the HEIF depth map + EXIF focal length to compute mm-per-pixel.

    The depth map gives distance-to-subject. Combined with focal length and
    sensor size, the pinhole model gives us the real-world scale:
        mm_per_pixel = (distance_m * sensor_width_mm) / (focal_length_mm * image_width_px)
    Which simplifies to:
        mm_per_pixel = distance_m * 1000 / focal_length_px
    Where focal_length_px = focal_length_mm * image_width_px / sensor_width_mm
    """
    exif = _get_exif_data(image_path)

    focal_length_tag = exif.get(ExifBase.FocalLength)
    if focal_length_tag is None:
        return None

    if hasattr(focal_length_tag, "numerator"):
        focal_length_mm = float(focal_length_tag.numerator) / float(focal_length_tag.denominator)
    else:
        focal_length_mm = float(focal_length_tag)

    if focal_length_mm <= 0:
        return None

    sensor_width_mm = _estimate_sensor_width(exif)

    focal_length_px = focal_length_mm * image_width_px / sensor_width_mm

    h, w = depth_map.shape[:2]
    cy, cx = h // 2, w // 2
    roi = depth_map[
        max(0, cy - h // 6):min(h, cy + h // 6),
        max(0, cx - w // 6):min(w, cx + w // 6),
    ]

    valid = roi[roi > 0]
    if len(valid) == 0:
        return None

    distance_units = float(np.median(valid))

    # Depth maps from iPhones are typically in disparity (inverse distance)
    # or normalized 0-255. Heuristic: if max > 10, treat as 8-bit disparity.
    if distance_units > 10:
        distance_m = 1.0 / (distance_units / 255.0 + 0.01)
    else:
        distance_m = distance_units

    distance_m = max(0.1, min(5.0, distance_m))

    mm_per_pixel = (distance_m * 1000.0) / focal_length_px

    confidence = 0.7
    if "LiDAR" in exif.get(ExifBase.Model, "") or "Pro" in exif.get(ExifBase.Model, ""):
        confidence = 0.85

    return CalibrationResult(
        scale_factor=round(mm_per_pixel, 6),
        method="heif_depth",
        confidence=confidence,
        depth_map_available=True,
    )
