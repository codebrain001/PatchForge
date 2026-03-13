"""
Hybrid thickness estimation pipeline for PatchForge.

Tries strategies in cascade order:
  1. LiDAR depth difference  (automatic, iPhone Pro only)
  2. Video multi-view depth  (automatic, if video key frames available)
  3. Side-photo via Gemini   (semi-automatic, if side photo uploaded)
  4. Manual fallback         (user sets slider)

Each strategy returns a ThicknessResult or None.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings
from app.core.exceptions import ThicknessEstimationError
from app.models.job import ThicknessMethod, ThicknessResult

logger = logging.getLogger("patchforge.thickness")


# ---------------------------------------------------------------------------
# Strategy 1: LiDAR depth difference
# ---------------------------------------------------------------------------

def estimate_from_lidar_depth(
    original_upload_path: str,
    mask: np.ndarray,
    scale_factor: float,
) -> Optional[ThicknessResult]:
    """
    Use the depth map embedded in an iPhone HEIF portrait photo.
    Compute the depth difference between the damaged area (inside the mask)
    and the surrounding intact surface.
    """
    ext = Path(original_upload_path).suffix.lower()
    if ext not in (".heic", ".heif"):
        return None

    try:
        from app.pipeline.depth_extraction import extract_depth_map
        depth_map = extract_depth_map(original_upload_path)
    except Exception:
        return None

    if depth_map is None:
        return None

    mask_resized = mask
    if depth_map.shape[:2] != mask.shape[:2]:
        mask_resized = cv2.resize(
            mask, (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    damage_mask = mask_resized > 127
    if np.sum(damage_mask) < 10:
        return None

    # Create a surrounding "ring" mask: dilate the damage mask and subtract
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(mask_resized, kernel, iterations=2)
    surround_mask = (dilated > 127) & (~damage_mask)

    if np.sum(surround_mask) < 10:
        return None

    depth_damage = depth_map[damage_mask]
    depth_surround = depth_map[surround_mask]

    valid_damage = depth_damage[depth_damage > 0]
    valid_surround = depth_surround[depth_surround > 0]

    if len(valid_damage) < 5 or len(valid_surround) < 5:
        return None

    median_damage = float(np.median(valid_damage))
    median_surround = float(np.median(valid_surround))

    depth_diff = abs(median_damage - median_surround)
    if depth_diff < 0.001:
        return None

    # Convert depth units to mm. iPhone depth maps use varying scales;
    # heuristic: if values are large (8-bit range), treat as disparity
    if max(median_damage, median_surround) > 10:
        disp_damage = median_damage / 255.0 + 0.01
        disp_surround = median_surround / 255.0 + 0.01
        dist_damage_m = 1.0 / disp_damage
        dist_surround_m = 1.0 / disp_surround
        thickness_mm = abs(dist_damage_m - dist_surround_m) * 1000.0
    else:
        thickness_mm = depth_diff * 1000.0

    thickness_mm = max(0.5, min(50.0, thickness_mm))

    confidence = 0.70
    if thickness_mm < 1.0 or thickness_mm > 30.0:
        confidence = 0.40

    logger.info(
        "LiDAR depth thickness: %.2f mm (damage=%.3f, surround=%.3f, conf=%.2f)",
        thickness_mm, median_damage, median_surround, confidence,
    )

    return ThicknessResult(
        thickness_mm=round(thickness_mm, 2),
        method=ThicknessMethod.LIDAR_DEPTH,
        confidence=confidence,
        depth_map_used=True,
        num_views_used=1,
    )


# ---------------------------------------------------------------------------
# Strategy 2: Video multi-view depth estimation
# ---------------------------------------------------------------------------

_depth_pipeline = None


def _load_depth_model():
    """Lazy-load the monocular depth estimation model."""
    global _depth_pipeline
    if _depth_pipeline is not None:
        return _depth_pipeline

    try:
        from transformers import pipeline
        _depth_pipeline = pipeline(
            "depth-estimation",
            model=settings.depth_model_name,
            device=0 if settings.device == "cuda" else -1,
        )
        return _depth_pipeline
    except Exception as e:
        logger.warning("Could not load depth model: %s", e)
        return None


def _estimate_depth_for_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
) -> Optional[float]:
    """
    Run monocular depth estimation on a single frame and compute the
    relative depth difference between the damaged region and its surroundings.

    Returns relative depth ratio (unitless) or None on failure.
    """
    pipe = _load_depth_model()
    if pipe is None:
        return None

    from PIL import Image as PILImage

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(frame_rgb)

    try:
        result = pipe(pil_img)
        depth_raw = result["depth"]
        if hasattr(depth_raw, "convert"):
            depth_map = np.array(depth_raw.convert("L"), dtype=np.float32)
        else:
            depth_map = np.array(depth_raw, dtype=np.float32)
    except Exception as e:
        logger.warning("Depth estimation failed for frame: %s", e)
        return None

    if depth_map.shape[:2] != mask.shape[:2]:
        mask_resized = cv2.resize(
            mask, (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        mask_resized = mask

    damage_mask = mask_resized > 127
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated = cv2.dilate(mask_resized, kernel, iterations=2)
    surround_mask = (dilated > 127) & (~damage_mask)

    if np.sum(damage_mask) < 5 or np.sum(surround_mask) < 5:
        return None

    depth_damage = float(np.median(depth_map[damage_mask]))
    depth_surround = float(np.median(depth_map[surround_mask]))

    if depth_surround < 0.001:
        return None

    return abs(depth_damage - depth_surround) / depth_surround


def estimate_from_video_frames(
    key_frame_paths: list[str],
    mask: np.ndarray,
    scale_factor: float,
    measurement_width_mm: float,
    measurement_height_mm: float = 0.0,
) -> Optional[ThicknessResult]:
    """
    Estimate thickness from multiple video key frames using monocular depth.

    The relative depth ratios from multiple frames are averaged, then
    converted to absolute mm using the calibrated scale and the smaller
    of the two known dimensions (width, height) for stability.
    """
    if len(key_frame_paths) < 2:
        return None

    ratios: list[float] = []

    for path in key_frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        ratio = _estimate_depth_for_frame(frame, mask)
        if ratio is not None and ratio > 0.001:
            ratios.append(ratio)

    if len(ratios) < 1:
        return None

    median_ratio = float(np.median(ratios))

    # Use the smaller dimension as the reference. The larger dimension often
    # includes alignment artifacts and is less reliable as a proxy for depth.
    ref_dim = measurement_width_mm
    if measurement_height_mm > 0:
        ref_dim = min(measurement_width_mm, measurement_height_mm)

    thickness_mm = median_ratio * ref_dim * settings.depth_width_scale
    thickness_mm = max(0.5, min(50.0, thickness_mm))

    confidence = min(0.75, 0.3 + 0.1 * len(ratios))
    std_dev = float(np.std(ratios)) if len(ratios) > 1 else 0.0
    if std_dev < 0.05:
        confidence += 0.1

    confidence = min(0.85, confidence)

    logger.info(
        "Video MVS thickness: %.2f mm (ratio=%.4f, %d views, std=%.4f, conf=%.2f)",
        thickness_mm, median_ratio, len(ratios), std_dev, confidence,
    )

    return ThicknessResult(
        thickness_mm=round(thickness_mm, 2),
        method=ThicknessMethod.VIDEO_MVS,
        confidence=round(confidence, 2),
        depth_map_used=True,
        num_views_used=len(ratios),
    )


# ---------------------------------------------------------------------------
# Strategy 3: Side photo via Gemini vision
# ---------------------------------------------------------------------------

def estimate_from_side_photo(
    side_image_path: str,
    scale_factor: float,
    measurement_width_mm: float,
    measurement_height_mm: float,
    manual_hint_mm: Optional[float] = None,
) -> Optional[ThicknessResult]:
    """
    Ask Gemini to analyze a side-angle photo and estimate break depth.

    The scale factor (mm/px) from calibration is provided so Gemini can
    reason about absolute distances.
    """
    from app.core.llm import is_llm_available, call_llm_vision, parse_json_response

    if not is_llm_available():
        logger.info("No LLM configured; skipping side-photo thickness.")
        return None

    if not Path(side_image_path).exists():
        return None

    img = cv2.imread(side_image_path)
    if img is None:
        return None

    try:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buf.tobytes()

        hint_text = ""
        if manual_hint_mm is not None:
            hint_text = f" The user estimates the depth is approximately {manual_hint_mm} mm."

        system = "You are a computer vision expert analyzing physical objects for 3D printing repair."
        prompt = (
            "You are analyzing a side-angle photograph of a broken/damaged object. "
            "The damage area has already been measured from a top-down view:\n"
            f"  - Width: {measurement_width_mm} mm\n"
            f"  - Height: {measurement_height_mm} mm\n"
            f"  - Calibration scale: {scale_factor:.4f} mm per pixel\n"
            f"{hint_text}\n\n"
            "From this side view, estimate the DEPTH (thickness) of the break/void "
            "in millimeters. Look for visible edges, shadows, and perspective cues.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"thickness_mm": <number>, "confidence": <0.0-1.0>, '
            '"reasoning": "<brief explanation>"}'
        )

        text, provider = call_llm_vision(system, prompt, image_bytes)
        logger.info("Side-photo analysis via %s", provider)

        parsed = parse_json_response(text)
        thickness = float(parsed.get("thickness_mm", 0))
        confidence = float(parsed.get("confidence", 0.5))

        if thickness <= 0:
            return None

        thickness = max(0.5, min(50.0, thickness))
        confidence = max(0.1, min(0.80, confidence))

        logger.info(
            "Side-photo thickness: %.2f mm (conf=%.2f, reason=%s)",
            thickness, confidence, parsed.get("reasoning", ""),
        )

        return ThicknessResult(
            thickness_mm=round(thickness, 2),
            method=ThicknessMethod.SIDE_PHOTO,
            confidence=round(confidence, 2),
            depth_map_used=False,
            num_views_used=1,
        )

    except Exception as e:
        logger.warning("Side-photo LLM analysis failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Hybrid router: try all strategies in cascade
# ---------------------------------------------------------------------------

def estimate_thickness(
    original_upload_path: Optional[str],
    mask: np.ndarray,
    scale_factor: float,
    measurement_width_mm: float,
    measurement_height_mm: float,
    key_frame_paths: Optional[list[str]] = None,
    side_image_path: Optional[str] = None,
    manual_hint_mm: Optional[float] = None,
) -> ThicknessResult:
    """
    Try all thickness estimation strategies in priority order.

    Returns the best result, or a manual-fallback result with zero confidence
    if all strategies fail.
    """
    strategies_tried: list[str] = []

    # Strategy 1: LiDAR depth
    if original_upload_path:
        strategies_tried.append("lidar_depth")
        result = estimate_from_lidar_depth(
            original_upload_path, mask, scale_factor,
        )
        if result is not None:
            logger.info("Thickness resolved via LiDAR depth: %.2f mm", result.thickness_mm)
            return result

    # Strategy 2: Video multi-view
    if key_frame_paths and len(key_frame_paths) >= 2:
        strategies_tried.append("video_mvs")
        result = estimate_from_video_frames(
            key_frame_paths, mask, scale_factor,
            measurement_width_mm, measurement_height_mm,
        )
        if result is not None:
            logger.info("Thickness resolved via video MVS: %.2f mm", result.thickness_mm)
            return result

    # Strategy 3: Side photo + Gemini
    if side_image_path and Path(side_image_path).exists():
        strategies_tried.append("side_photo")
        result = estimate_from_side_photo(
            side_image_path, scale_factor,
            measurement_width_mm, measurement_height_mm,
            manual_hint_mm=manual_hint_mm,
        )
        if result is not None:
            logger.info("Thickness resolved via side photo: %.2f mm", result.thickness_mm)
            return result

    # All strategies failed — return a manual fallback placeholder
    logger.warning(
        "All thickness strategies failed (tried: %s). Falling back to manual.",
        ", ".join(strategies_tried) if strategies_tried else "none",
    )

    default = settings.default_thickness_mm
    if manual_hint_mm is not None and manual_hint_mm > 0:
        default = manual_hint_mm

    return ThicknessResult(
        thickness_mm=default,
        method=ThicknessMethod.MANUAL,
        confidence=0.0,
        depth_map_used=False,
        num_views_used=0,
    )
