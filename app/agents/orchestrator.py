"""
Gemini-powered agent orchestrator.

Runs calibration -> segmentation -> measurement -> thickness agents for analysis,
and mesh -> validation agents for generation. Each agent produces
reasoning that gets logged on the job.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.agents.calibration_agent import CalibrationAgent
from app.agents.measurement_agent import MeasurementAgent
from app.agents.mesh_agent import MeshAgent
from app.agents.printer_agent import PrinterAgent
from app.agents.segmentation_agent import SegmentationAgent
from app.agents.thickness_agent import ThicknessAgent
from app.agents.validation_agent import ValidationAgent
from app.core.storage import job_mask_path, job_mesh_path, job_propagated_masks_dir, job_viz_path
from app.models.job import (
    Job, JobStatus, ReasoningEntry, ThicknessMethod, ThicknessResult,
    UploadType, DetectionMode, CalibrationResult,
)

from app.core.job_store import get_job, store_job

logger = logging.getLogger("patchforge.orchestrator")


# Rich event callback: receives (job_id, message_dict).
# Message dicts have a "type" key — "status", "reasoning", or "viz".
EventCallback = Callable[[str, dict], None]

# Authoritative coin diameter / across-flats lookup (mm).
# Used by both vision prompts and by the self-consistency validator.
_COIN_SIZES: dict[str, float] = {
    "us quarter": 24.26,
    "us penny": 19.05,
    "us nickel": 21.21,
    "us dime": 17.91,
    "euro 1€": 23.25,
    "euro 1 euro": 23.25,
    "euro 2€": 25.75,
    "euro 2 euro": 25.75,
    "euro 50c": 24.25,
    "euro 20c": 22.25,
    "uk £1": 23.43,
    "uk 1 pound": 23.43,
    "uk £2": 28.4,
    "uk 2 pound": 28.4,
    "uk 50p": 27.3,
    "uk 50 pence": 27.3,
    "50 pence": 27.3,
    "50p": 27.3,
    "uk 20p": 21.4,
    "uk 20 pence": 21.4,
    "uk 10p": 24.5,
    "uk 10 pence": 24.5,
    "uk 5p": 18.0,
    "uk 5 pence": 18.0,
    "uk 2p": 25.9,
    "uk 2 pence": 25.9,
    "uk 1p": 20.3,
    "uk 1 penny": 20.3,
    "zar r5": 26.0,
    "zar r2": 23.0,
    "zar r1": 20.0,
    "zar 50c": 22.0,
}

# Formatted string for use in LLM prompts
_COIN_LIST_PROMPT = (
    "DEFAULT: The coin in this image is most likely a UK £2 coin (28.4mm diameter, "
    "bimetallic — gold center with silver rim). Use 28.4mm unless the coin is clearly "
    "a different type.\n"
    "    Other coins for reference: "
    "UK £1=23.43mm (thin, gold, round), UK 50p=27.3mm (silver, heptagonal/7-sided), "
    "UK 20p=21.4mm, UK 10p=24.5mm, UK 5p=18mm, UK 2p=25.9mm, UK 1p=20.3mm, "
    "US quarter=24.26mm, US penny=19.05mm, US nickel=21.21mm, US dime=17.91mm, "
    "Euro 1€=23.25mm, Euro 2€=25.75mm, Euro 50c=24.25mm, "
    "ZAR R5=26mm, ZAR R2=23mm, ZAR R1=20mm, ZAR 50c=22mm"
)


_DEFAULT_COIN_SIZE_MM = 28.4  # UK £2 coin — default for this demo


def _lookup_coin_size(ref_name: str) -> float:
    """Fuzzy-match a reference_object string against known coin sizes.

    Falls back to UK £2 (28.4 mm) when no match is found.
    """
    if not ref_name:
        return _DEFAULT_COIN_SIZE_MM
    name = ref_name.lower().strip()
    if name in _COIN_SIZES:
        return _COIN_SIZES[name]
    for key, size in _COIN_SIZES.items():
        if key in name or name in key:
            return size
    logger.info("Coin '%s' not recognized — defaulting to UK £2 (28.4 mm)", ref_name)
    return _DEFAULT_COIN_SIZE_MM

# Singleton agents
_cal_agent = CalibrationAgent()
_seg_agent = SegmentationAgent()
_meas_agent = MeasurementAgent()
_thick_agent = ThicknessAgent()
_mesh_agent = MeshAgent()
_val_agent = ValidationAgent()
_print_agent = PrinterAgent()


async def _vision_calibration_fallback(
    image_bgr: np.ndarray,
    h_img: int,
    w_img: int,
) -> Optional[float]:
    """Ask the vision LLM to identify objects of known size for scale estimation."""
    from app.core.llm import call_llm_vision, parse_json_response

    try:
        max_dim = 800
        img = image_bgr
        resize_factor = 1.0
        if max(h_img, w_img) > max_dim:
            resize_factor = max_dim / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img * resize_factor), int(h_img * resize_factor)))

        disp_h, disp_w = img.shape[:2]
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = buf.tobytes()

        system = (
            "You are a calibration expert for a photo-to-3D-print repair pipeline. "
            "You identify objects of known real-world size in images and compute "
            "a mm-per-pixel scale factor from them."
        )
        prompt = (
            f"This image is {disp_w}x{disp_h} pixels. I need to determine the "
            "real-world scale (mm per pixel). There is no calibration marker.\n\n"
            "Identify ANY object in the image whose real-world size you know:\n"
            f"- Coins: {_COIN_LIST_PROMPT}\n"
            "- Fingers/hand (index finger width ~17mm, thumb ~20mm)\n"
            "- Credit card: 85.6mm x 54mm\n"
            "- USB-A port: 12mm wide, USB-C port: 8.25mm wide\n"
            "- Standard keyboard key: ~15mm\n\n"
            "DEFAULT: Assume the coin is a UK 2 pound coin (28.4mm) unless it is "
            "clearly something else. A UK £2 is bimetallic (gold center, silver rim). "
            "A UK £1 is smaller, thin, and all gold. Do not confuse them.\n\n"
            "Measure how many pixels wide that object appears in THIS image, "
            "then compute: mm_per_pixel = known_mm / pixel_width.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"reference_object": "<exact object name>", '
            '"reference_size_mm": <real-world width in mm>, '
            '"reference_size_px": <pixel width as measured in this image>, '
            '"mm_per_pixel": <computed scale = reference_size_mm / reference_size_px>, '
            '"confidence": <0.0-1.0>}'
        )

        text, provider = await asyncio.to_thread(
            call_llm_vision, system, prompt, image_bytes,
        )
        logger.info("Vision calibration via %s", provider)

        parsed = parse_json_response(text)
        mm_per_px = float(parsed.get("mm_per_pixel", 0))
        confidence = float(parsed.get("confidence", 0))
        ref_obj = parsed.get("reference_object", "unknown")

        if mm_per_px > 0.001 and confidence >= 0.3:
            original_mm_per_px = mm_per_px * resize_factor
            logger.info(
                "Vision calibration: %.5f mm/px (display) -> %.5f mm/px (original) from %s (conf=%.2f)",
                mm_per_px, original_mm_per_px, ref_obj, confidence,
            )
            return original_mm_per_px

    except Exception as e:
        logger.warning("Vision calibration fallback failed: %s", e)

    return None


def _generate_negative_points(
    cx: int, cy: int, w: int, h: int,
) -> list[dict]:
    """Generate negative SAM 2 point prompts on intact surfaces around the damage.

    Places points at ~20% of image dimension offset from the positive click
    in the four cardinal directions, clamped to image bounds with a margin.
    These tell SAM 2 "do NOT include these areas in the mask," preventing
    it from segmenting the entire object when clicked on a gap.
    """
    margin = 20
    offset_x = max(int(w * 0.15), 80)
    offset_y = max(int(h * 0.15), 80)

    candidates = [
        (cx - offset_x, cy),
        (cx + offset_x, cy),
        (cx, cy - offset_y),
        (cx, cy + offset_y),
    ]

    points = []
    for nx, ny in candidates:
        nx = max(margin, min(nx, w - margin))
        ny = max(margin, min(ny, h - margin))
        if abs(nx - cx) > 30 or abs(ny - cy) > 30:
            points.append({"x": nx, "y": ny, "label": 0})
    return points


async def _vision_locate_damage(
    image_bgr: np.ndarray,
) -> Optional[tuple[int, int]]:
    """Ask the vision LLM to locate the broken/missing piece in the image.

    Returns (x, y) pixel coordinates of the damage center, or None.
    """
    from app.core.llm import call_llm_vision, parse_json_response

    h_img, w_img = image_bgr.shape[:2]

    try:
        max_dim = 800
        img = image_bgr
        scale = 1.0
        if max(h_img, w_img) > max_dim:
            scale = max_dim / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = buf.tobytes()

        disp_h, disp_w = img.shape[:2]

        system = (
            "You are a computer vision expert specializing in damage detection "
            "for 3D-printable repair patches."
        )
        prompt = (
            f"This image is {disp_w}x{disp_h} pixels. It shows an object that has "
            "a piece broken off or missing. We need to find WHERE the missing piece "
            "was, so we can 3D-print a replacement patch to fill the gap.\n\n"
            "Find the GAP/VOID where material is missing (NOT the whole object) and "
            "return the PIXEL coordinates of the center of that gap. Look for:\n"
            "- A gap, void, or missing chunk where a piece broke off\n"
            "- A visible break line, fracture, or crack edge\n"
            "- An area where material is clearly absent compared to the rest\n\n"
            "IMPORTANT: Point to the CENTER of the missing area, not the center "
            "of the whole object.\n\n"
            "Coordinates must be within the image bounds "
            f"(0 <= x < {disp_w}, 0 <= y < {disp_h}).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"x": <pixel x of gap center>, '
            '"y": <pixel y of gap center>, '
            '"description": "<describe what is missing and where>", '
            '"confidence": <0.0-1.0>}'
        )

        text, provider = await asyncio.to_thread(
            call_llm_vision, system, prompt, image_bytes,
        )

        parsed = parse_json_response(text)
        x = int(float(parsed.get("x", 0)))
        y = int(float(parsed.get("y", 0)))
        conf = float(parsed.get("confidence", 0))
        desc = parsed.get("description", "")

        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)

        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))

        if conf >= 0.3:
            logger.info(
                "Vision damage location: (%d, %d) conf=%.2f — %s",
                x, y, conf, desc,
            )
            return (x, y)

    except Exception as e:
        logger.warning("Vision damage localization failed: %s", e)

    return None


@dataclass
class VisionMeasurement:
    width_mm: float
    height_mm: float
    thickness_mm: float
    confidence: float
    description: str


async def _vision_measure_damage(
    image_bgr: np.ndarray,
) -> Optional[VisionMeasurement]:
    """Ask the vision LLM to directly measure the break using visible reference objects.

    This bypasses all pixel-based measurement and asks the LLM to estimate
    real-world dimensions from visual context (coins, fingers, known objects).
    """
    from app.core.llm import call_llm_vision, parse_json_response

    try:
        h_img, w_img = image_bgr.shape[:2]
        max_dim = 1024
        img = image_bgr
        if max(h_img, w_img) > max_dim:
            s = max_dim / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img * s), int(h_img * s)))

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buf.tobytes()

        system = (
            "You are a precision measurement expert for 3D-printable repair parts. "
            "The target printer is a Bambu Lab A1 (build volume: 256 x 256 x 256 mm). "
            "You measure real-world dimensions from photos using proportional reasoning "
            "against objects of known size. You always report pixel measurements so your "
            "work can be verified."
        )
        prompt = (
            "This image shows an object with a piece broken off. We need to "
            "3D-print a PATCH that fits into the gap left by the missing piece.\n\n"
            "YOUR TASK: Measure the GAP/VOID — the space where the missing piece "
            "used to be. You are measuring the PATCH dimensions, NOT the whole "
            "object. The patch must fill the break from edge to edge.\n\n"
            "STEP 1 — Identify the reference object for scale:\n"
            f"  * Coins: {_COIN_LIST_PROMPT}\n"
            "  * Fingers: adult index finger width ~17mm, thumb ~20mm\n"
            "  * Credit card: 85.6mm x 54mm\n"
            "  DEFAULT: Assume the coin is a UK 2 pound coin (28.4mm) unless it is "
            "clearly something else. A UK £2 is bimetallic (gold center, silver rim). "
            "A UK £1 is smaller, thin, and all gold. A UK 50p is silver and heptagonal "
            "(7-sided). Do NOT confuse them.\n\n"
            "STEP 2 — Pixel measurement (MANDATORY — do this FIRST, before mm):\n"
            "  a) Measure the reference object's DIAMETER/WIDTH in pixels (ref_px)\n"
            "  b) Measure the break's WIDTH in pixels (break_width_px) — the shorter\n"
            "     horizontal extent of the ACTUAL gap opening only\n"
            "  c) Measure the break's HEIGHT in pixels (break_height_px) — the longer\n"
            "     vertical extent of the ACTUAL gap opening only\n\n"
            "  NOTE: Width and height are independent measurements. A rectangular break\n"
            "  will have DIFFERENT width and height values. Do NOT assume they are equal.\n\n"
            "STEP 3 — Convert to mm (MANDATORY formula — show your work):\n"
            "  ref_mm = known diameter of the reference object in mm\n"
            "  width_mm = ref_mm * (break_width_px / ref_px)\n"
            "  height_mm = ref_mm * (break_height_px / ref_px)\n"
            "  The values you return for width_mm and height_mm MUST equal these\n"
            "  computed values. Do NOT round, adjust, or apply margins.\n\n"
            "  WORKED EXAMPLE:\n"
            "    Coin: UK 2 pound (28.4mm), appears ~100px wide in image\n"
            "    Break: appears ~53px wide, ~106px tall\n"
            "    width_mm = 28.4 * (53/100) = 15.05\n"
            "    height_mm = 28.4 * (106/100) = 30.10\n\n"
            "STEP 4 — Estimate thickness (the DEPTH/WALL-THICKNESS of the break):\n"
            "  Thickness = how thick the WALL or SHELL was where it broke off. This\n"
            "  is the Z-depth of the patch — how far it protrudes from the surface.\n"
            "  Think of it as the distance from the front face to the back face of\n"
            "  the broken wall.\n\n"
            "  Option A (preferred): Look at the FRACTURE EDGE — the exposed broken\n"
            "  cross-section of the wall. Measure the NARROW dimension of that edge\n"
            "  (the wall depth), NOT the gap opening width. For example, if you see\n"
            "  a broken rim or ledge, measure from the outer surface to the inner\n"
            "  surface of that rim.\n"
            "    edge_thickness_px = <wall cross-section depth in pixels>\n"
            "    thickness_mm = ref_mm * (edge_thickness_px / ref_px)\n"
            "  NOTE: It is possible for thickness to be close to the width or height\n"
            "  if the object has thick walls. This is valid. Just make sure you are\n"
            "  measuring the wall cross-section, not re-measuring the gap opening.\n"
            "  Option B: If the edge cross-section is NOT visible, look for depth cues\n"
            "  (shadows, perspective, visible ledges). If you truly CANNOT measure\n"
            "  thickness, set thickness_mm to 0 and confidence below 0.3.\n"
            "  Do NOT guess thickness from material type or generic assumptions.\n\n"
            "STEP 5 — Visual sanity check:\n"
            "  Compare break dimensions to the reference object visually.\n"
            "  If the break is SMALLER than the coin, both dimensions MUST be < coin diameter.\n"
            "  If the break is LARGER than the coin, both dimensions MUST be > coin diameter.\n"
            "  If your mm values violate this, your pixel measurements are wrong — redo them.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"width_mm": <number from Step 3 formula>, '
            '"height_mm": <number from Step 3 formula>, '
            '"thickness_mm": <number from Step 4 formula or 0>, '
            '"reference_object": "<exact coin/object name and its known mm size>", '
            '"ref_px": <reference diameter in pixels>, '
            '"break_width_px": <break width in pixels>, '
            '"break_height_px": <break height in pixels>, '
            '"edge_thickness_px": <edge thickness in pixels if Option A, else 0>, '
            '"reasoning": "<show ref_mm * (break_px / ref_px) calculation for width and height>", '
            '"thickness_reasoning": "<Option A or B, show calculation if A>", '
            '"description": "<describe the break shape — rectangular, triangular, etc.>", '
            '"confidence": <0.0-1.0>}'
        )

        text, provider = await asyncio.to_thread(
            call_llm_vision, system, prompt, image_bytes,
        )
        logger.info("Vision measurement via %s", provider)

        parsed = parse_json_response(text)
        width = float(parsed.get("width_mm", 0))
        height = float(parsed.get("height_mm", 0))
        thickness = float(parsed.get("thickness_mm", 0))
        conf = float(parsed.get("confidence", 0))
        desc = parsed.get("description", "")
        ref = parsed.get("reference_object", "unknown")
        reasoning = parsed.get("reasoning", "")
        ref_px = parsed.get("ref_px", 0)
        break_w_px = parsed.get("break_width_px", 0)
        break_h_px = parsed.get("break_height_px", 0)
        thickness_reasoning = parsed.get("thickness_reasoning", "")

        logger.info(
            "Vision measurement reasoning: ref=%s ref_px=%s "
            "break_px=%sx%s — %s",
            ref, ref_px, break_w_px, break_h_px, reasoning,
        )
        if thickness_reasoning:
            logger.info("Vision thickness reasoning: %s", thickness_reasoning)

        edge_t_px = parsed.get("edge_thickness_px", 0)

        # --- Self-consistency validation ---
        # Recompute dimensions from the pixel fields the LLM itself reported.
        # If the returned mm values diverge >30% from the recomputed values,
        # the LLM contradicted its own reasoning — use the recomputed values.
        try:
            _ref_px = float(ref_px) if ref_px else 0
            _break_w = float(break_w_px) if break_w_px else 0
            _break_h = float(break_h_px) if break_h_px else 0
            _edge_t = float(edge_t_px) if edge_t_px else 0
        except (TypeError, ValueError):
            _ref_px = _break_w = _break_h = _edge_t = 0

        if _ref_px > 0 and _break_w > 0 and _break_h > 0:
            ref_size_mm = _lookup_coin_size(ref)
            recomputed_w = ref_size_mm * (_break_w / _ref_px)
            recomputed_h = ref_size_mm * (_break_h / _ref_px)

            if recomputed_w > 0 and abs(width - recomputed_w) / recomputed_w > 0.3:
                logger.warning(
                    "Vision self-contradiction: returned w=%.1f but pixel data gives %.1f — using recomputed",
                    width, recomputed_w,
                )
                width = recomputed_w

            if recomputed_h > 0 and abs(height - recomputed_h) / recomputed_h > 0.3:
                logger.warning(
                    "Vision self-contradiction: returned h=%.1f but pixel data gives %.1f — using recomputed",
                    height, recomputed_h,
                )
                height = recomputed_h

            # Thickness self-consistency (Option A: edge measured in pixels)
            if _edge_t > 0:
                recomputed_t = ref_size_mm * (_edge_t / _ref_px)

                if _break_w > 0 and _break_h > 0 and _edge_t == _break_w == _break_h:
                    logger.warning(
                        "edge_thickness_px (%.0f) equals BOTH break width(%.0f) "
                        "and height(%.0f) — LLM returned identical values for "
                        "all dimensions, discarding thickness",
                        _edge_t, _break_w, _break_h,
                    )
                    thickness = 0
                elif _edge_t == _break_w or _edge_t == _break_h:
                    logger.warning(
                        "edge_thickness_px (%.0f) matches one gap dimension "
                        "(w=%.0f, h=%.0f) — LLM may have re-measured width/height "
                        "as thickness, halving confidence",
                        _edge_t, _break_w, _break_h,
                    )
                    thickness = recomputed_t if recomputed_t > 0 else thickness
                    conf *= 0.5
                elif recomputed_t > 0 and thickness > 0:
                    if abs(thickness - recomputed_t) / max(recomputed_t, 0.1) > 0.3:
                        logger.warning(
                            "Vision thickness self-contradiction: returned t=%.1f but pixel data gives %.1f — using recomputed",
                            thickness, recomputed_t,
                        )
                        thickness = recomputed_t
                elif recomputed_t > 0 and thickness <= 0:
                    logger.info(
                        "Vision thickness recomputed from edge_thickness_px: %.1f mm",
                        recomputed_t,
                    )
                    thickness = recomputed_t

        if width <= 0 or height <= 0 or conf < 0.3:
            logger.warning(
                "Vision measurement rejected: %.1f x %.1f mm (conf=%.2f)",
                width, height, conf,
            )
            return None

        if width > height:
            width, height = height, width

        logger.info(
            "Vision measurement: %.1f x %.1f x %.1f mm (conf=%.2f, ref=%s) — %s",
            width, height, thickness, conf, ref, desc,
        )

        return VisionMeasurement(
            width_mm=width,
            height_mm=height,
            thickness_mm=thickness,
            confidence=conf,
            description=desc,
        )

    except Exception as e:
        logger.warning("Vision measurement failed: %s", e)

    return None


async def _vision_get_break_polygon(
    image_bgr: np.ndarray,
) -> Optional[np.ndarray]:
    """Ask the vision LLM to trace the outline of the break as pixel coordinates.

    Returns an OpenCV-style contour array, or None.
    """
    from app.core.llm import call_llm_vision, parse_json_response

    h_img, w_img = image_bgr.shape[:2]

    try:
        max_dim = 1024
        img = image_bgr
        scale = 1.0
        if max(h_img, w_img) > max_dim:
            scale = max_dim / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))

        disp_h, disp_w = img.shape[:2]
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buf.tobytes()

        system = (
            "You are a precise computer vision expert. You trace the outline of "
            "missing/broken areas in images so a replacement patch can be 3D-printed."
        )
        prompt = (
            f"This image is {disp_w}x{disp_h} pixels. It shows an object with a "
            "piece broken off — there is a visible gap/void where material is missing.\n\n"
            "TASK: Trace the OUTLINE of the GAP (the missing piece area) as a polygon.\n"
            "The polygon defines the shape of the PATCH we will 3D-print to fill the gap.\n"
            "Return the corner points in PIXEL coordinates.\n\n"
            "IMPORTANT — PREFER RECTANGLES:\n"
            "Most broken pieces leave a roughly rectangular gap. If the gap is even\n"
            "approximately rectangular, return EXACTLY 4 corner points forming a clean\n"
            "rectangle (top-left, top-right, bottom-right, bottom-left). Do NOT add\n"
            "extra points for minor edge irregularities — the 3D printer needs a\n"
            "clean shape, not a pixel-perfect trace of jagged fracture edges.\n"
            "Only use more than 4 points if the gap is genuinely triangular or has a\n"
            "clearly non-rectangular shape (e.g. an L-shape with a distinct step).\n\n"
            "RULES:\n"
            "- Trace ONLY the gap/void boundary — NOT the outline of the whole object\n"
            "- Use the FEWEST points that capture the gap shape (4 for rectangles)\n"
            "- Points should go clockwise around the gap boundary\n"
            "- Coordinates must be within the image bounds\n\n"
            "Respond with ONLY JSON:\n"
            '{"points": [[x1,y1], [x2,y2], ...], '
            '"shape_type": "<triangle|rectangle|irregular>", '
            '"confidence": <0.0-1.0>}'
        )

        text, provider = await asyncio.to_thread(
            call_llm_vision, system, prompt, image_bytes,
        )

        parsed = parse_json_response(text)
        points = parsed.get("points", [])
        conf = float(parsed.get("confidence", 0))
        shape_type = parsed.get("shape_type", "unknown")

        if len(points) < 3 or conf < 0.3:
            logger.warning("Vision polygon rejected: %d points, conf=%.2f", len(points), conf)
            return None

        pts = np.array(points, dtype=np.float32)

        if scale != 1.0:
            pts = pts / scale

        pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

        contour = pts.reshape(-1, 1, 2).astype(np.int32)
        area = cv2.contourArea(contour)

        if area < 50:
            logger.warning("Vision polygon too small: area=%d px", area)
            return None

        logger.info(
            "Vision polygon: %d points, shape=%s, area=%d px, conf=%.2f",
            len(points), shape_type, int(area), conf,
        )
        return contour

    except Exception as e:
        logger.warning("Vision polygon tracing failed: %s", e)

    return None


def _log_reasoning(job: Job, agent_name: str, stage: str, result) -> None:
    if hasattr(result, "success") and not result.success:
        logger.warning(
            "%s at stage '%s' recommends halting (confidence=%.2f): %s",
            agent_name, stage, result.confidence, result.reasoning,
        )
    job.reasoning_log.append(ReasoningEntry(
        agent=agent_name,
        stage=stage,
        reasoning=result.reasoning,
        suggestions=result.suggestions,
        confidence=result.confidence,
    ))


async def run_analysis(
    job: Job,
    image: np.ndarray,
    points: list[dict],
    marker_size_mm: float = 40.0,
    ref_line_start: Optional[list[int]] = None,
    ref_line_end: Optional[list[int]] = None,
    ref_line_mm: Optional[float] = None,
    webxr_scale: Optional[float] = None,
    on_progress: Optional[EventCallback] = None,
) -> Job:
    """Run Calibration -> Segmentation -> Measurement -> Thickness agents."""

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, {"type": "status", "status": status.value})

    def _emit(event: dict):
        if on_progress:
            on_progress(job.id, event)

    def _log_and_emit(agent_name: str, stage: str, result):
        _log_reasoning(job, agent_name, stage, result)
        entry = job.reasoning_log[-1]
        _emit({"type": "reasoning", "entry": entry.model_dump()})

    job.reasoning_log = []

    try:
        # --- Calibration Agent (consensus: run ALL strategies, LLM picks) ---
        _notify(JobStatus.CALIBRATING)
        depth_map = None
        try:
            cal_result, cal_analysis, depth_map = await _cal_agent.run(
                image,
                marker_size_mm=marker_size_mm,
                original_upload_path=job.original_upload_path,
                ref_line_start=tuple(ref_line_start) if ref_line_start else None,
                ref_line_end=tuple(ref_line_end) if ref_line_end else None,
                ref_line_mm=ref_line_mm,
                webxr_scale=webxr_scale,
            )

            if cal_result.scale_factor <= 0 or cal_result.method == "none":
                h_img, w_img = image.shape[:2]
                vision_scale = await _vision_calibration_fallback(image, h_img, w_img)
                if vision_scale is not None:
                    cal_result = CalibrationResult(
                        scale_factor=vision_scale,
                        method="vision_estimated",
                        confidence=0.5,
                    )
                else:
                    cal_result = CalibrationResult(
                        scale_factor=100.0 / max(h_img, w_img, 1),
                        method="estimated",
                        confidence=0.3,
                    )

            job.calibration = cal_result
            _log_and_emit("CalibrationAgent", "calibration_consensus", cal_analysis)
        except Exception as cal_err:
            logger.warning("Calibration consensus failed: %s — using vision fallback", cal_err)
            h_img, w_img = image.shape[:2]
            vision_scale = await _vision_calibration_fallback(image, h_img, w_img)
            estimated_scale = vision_scale if vision_scale else 100.0 / max(h_img, w_img, 1)
            cal_method = "vision_estimated" if vision_scale else "estimated"
            cal_confidence = 0.5 if vision_scale else 0.3

            cal_result = CalibrationResult(
                scale_factor=estimated_scale,
                method=cal_method,
                confidence=cal_confidence,
            )
            job.calibration = cal_result
            entry = ReasoningEntry(
                agent="CalibrationAgent",
                stage="calibration_consensus",
                reasoning=f"Calibration consensus failed ({cal_err}). Using {cal_method} scale "
                          f"({estimated_scale:.4f} mm/px).",
                suggestions=["Add an ArUco marker to the scene", "Draw a reference line"],
                confidence=cal_confidence,
            )
            job.reasoning_log.append(entry)
            _emit({"type": "reasoning", "entry": entry.model_dump()})
        _notify(JobStatus.CALIBRATED)

        # --- Segmentation Agent ---
        _notify(JobStatus.SEGMENTING)
        mask = None
        contours = None

        has_neg = any(p.get("label", 1) == 0 for p in points)
        if not has_neg and points:
            p0 = points[0]
            h_seg, w_seg = image.shape[:2]
            neg = _generate_negative_points(p0["x"], p0["y"], w_seg, h_seg)
            seg_points = list(points) + neg
        else:
            seg_points = points

        try:
            mask, contours, seg_analysis = await _seg_agent.run(image, seg_points)
            _log_and_emit("SegmentationAgent", "segmentation", seg_analysis)
        except Exception as seg_err:
            logger.warning("SAM 2 segmentation failed: %s — trying vision polygon fallback", seg_err)

        if contours is None or len(contours) == 0:
            vision_poly = await _vision_get_break_polygon(image)
            if vision_poly is not None:
                poly_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(poly_mask, [vision_poly], -1, 255, -1)
                mask = poly_mask
                contours = [vision_poly]
                logger.info("SAM 2 failed — using vision-traced polygon as fallback contour")
            else:
                raise RuntimeError(
                    "Segmentation failed: neither SAM 2 nor vision polygon "
                    "could identify the damaged area. Try clicking closer to the break."
                )

        job.contours = contours
        mask_path = job_mask_path(job.id)
        cv2.imwrite(str(mask_path), mask)

        # Generate and emit SAM 2 mask overlay visualization
        try:
            from app.pipeline.visualization import create_sam2_overlay
            overlay = create_sam2_overlay(image, mask, contours)
            viz_path = job_viz_path(job.id, "sam2_mask")
            cv2.imwrite(str(viz_path), overlay)
            _emit({"type": "viz", "name": "sam2_mask",
                    "url": f"/api/v1/jobs/{job.id}/viz/sam2_mask"})
        except Exception as viz_err:
            logger.warning("SAM 2 visualization failed (non-fatal): %s", viz_err)

        _notify(JobStatus.SEGMENTED)

        # --- Video Propagation (for video uploads with 2+ key frames) ---
        if (
            job.upload_type == UploadType.VIDEO
            and len(job.key_frame_paths) >= 2
            and job.click_points
        ):
            _notify(JobStatus.PROPAGATING_MASKS)
            try:
                from app.pipeline.video_segmentation import propagate_masks

                prop_dir = job_propagated_masks_dir(job.id)
                prop_result = await asyncio.to_thread(
                    propagate_masks,
                    key_frame_paths=job.key_frame_paths,
                    reference_frame_index=job.best_frame_index,
                    click_points=job.click_points,
                    output_dir=str(prop_dir),
                )
                job.propagated_mask_paths = [
                    m.mask_path for m in prop_result.masks
                ]
                job.propagation_stats = {
                    "total_frames_tracked": prop_result.total_frames_tracked,
                    "frames_with_damage": prop_result.frames_with_damage,
                    "mean_coverage": prop_result.mean_coverage,
                    "per_frame": [
                        {
                            "frame_index": m.frame_index,
                            "coverage_ratio": m.coverage_ratio,
                            "iou_with_reference": m.iou_with_reference,
                        }
                        for m in prop_result.masks
                    ],
                }
                _notify(JobStatus.MASKS_PROPAGATED)
            except Exception as e:
                logger.warning("Video propagation failed for %s (non-fatal): %s", job.id, e)
                _notify(JobStatus.SEGMENTED)

        # --- Measurement Agent (pixel-based) ---
        _notify(JobStatus.MEASURING)
        meas_result, meas_analysis = await _meas_agent.run(
            contours, cal_result.scale_factor, cal_result.method,
            calibration_confidence=cal_result.confidence,
            image_bgr=image, mask=mask,
        )
        _log_and_emit("MeasurementAgent", "measurement", meas_analysis)

        # --- Vision Measurement (primary) + integration ---
        meas_result, vision_meas = await _integrate_measurements(
            meas_result, image, cal_result.confidence, job,
        )
        # Emit the vision cross-check reasoning that _integrate_measurements added
        if job.reasoning_log and job.reasoning_log[-1].agent == "VisionMeasurement":
            _emit({"type": "reasoning", "entry": job.reasoning_log[-1].model_dump()})

        job.contours = _scale_vision_polygon_to_measurements(
            contours, meas_result, cal_result.scale_factor,
        )
        job.measurement = meas_result
        _notify(JobStatus.MEASURED)

        # --- Thickness Agent (consensus: run ALL strategies, LLM picks) ---
        _notify(JobStatus.ESTIMATING_DEPTH)
        vision_thickness = vision_meas.thickness_mm if vision_meas else None
        thick_result, thick_analysis = await _thick_agent.run(
            mask=mask,
            scale_factor=cal_result.scale_factor,
            measurement_width_mm=meas_result.width_mm,
            measurement_height_mm=meas_result.height_mm,
            original_upload_path=job.original_upload_path,
            key_frame_paths=job.key_frame_paths if job.key_frame_paths else None,
            side_image_path=job.side_image_path,
            depth_map=depth_map,
            vision_thickness_mm=vision_thickness,
            calibration_method=cal_result.method,
            image_bgr=image,
        )

        job.thickness_result = thick_result
        job.thickness_mm = thick_result.thickness_mm
        _log_and_emit("ThicknessAgent", "thickness_consensus", thick_analysis)

        # Generate and emit Depth Anything visualization
        try:
            from app.pipeline.visualization import create_depth_visualization
            depth_viz = await asyncio.to_thread(create_depth_visualization, image, mask)
            if depth_viz is not None:
                viz_path = job_viz_path(job.id, "depth_map")
                cv2.imwrite(str(viz_path), depth_viz)
                _emit({"type": "viz", "name": "depth_map",
                        "url": f"/api/v1/jobs/{job.id}/viz/depth_map"})
        except Exception as viz_err:
            logger.warning("Depth visualization failed (non-fatal): %s", viz_err)

        if thick_result.method != ThicknessMethod.MANUAL:
            _notify(JobStatus.DEPTH_ESTIMATED)
        else:
            _notify(JobStatus.MEASURED)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
        store_job(job)
        raise

    store_job(job)
    return job


def _validate_measurement_sanity(
    meas_result,
) -> float:
    """Check measurements against physical constraints. Returns a penalty multiplier (0-1)."""
    w = meas_result.width_mm
    h = meas_result.height_mm
    penalty = 1.0

    if w < 2.0 or h < 2.0:
        logger.warning("Measurement too small: %.1f x %.1f mm", w, h)
        penalty *= 0.5
    if w > 300.0 or h > 300.0:
        logger.warning("Measurement too large: %.1f x %.1f mm", w, h)
        penalty *= 0.5

    area = meas_result.area_mm2
    expected_area = w * h
    if expected_area > 0 and area > 0:
        area_ratio = area / expected_area
        if area_ratio > 3.0 or area_ratio < 0.1:
            logger.warning(
                "Area %.1f mm² inconsistent with w*h %.1f mm² (ratio=%.2f)",
                area, expected_area, area_ratio,
            )
            penalty *= 0.7

    return penalty


async def _integrate_measurements(
    meas_result,
    image_bgr: np.ndarray,
    cal_confidence: float,
    job: Job,
) -> tuple:
    """Run vision measurement and integrate with pixel-based measurement.

    Vision is the primary source; pixel measurements validate/refine it.
    Returns (updated meas_result, vision_meas_or_None).
    """
    vision_meas = await _vision_measure_damage(image_bgr)

    if vision_meas is not None:
        pw, ph = meas_result.width_mm, meas_result.height_mm
        vw, vh = vision_meas.width_mm, vision_meas.height_mm

        logger.info(
            "Pixel vs Vision: pixel=%.1fx%.1f  vision=%.1fx%.1f  "
            "vision_conf=%.2f  pixel_conf=%.2f  cal_conf=%.2f",
            pw, ph, vw, vh, vision_meas.confidence,
            meas_result.confidence, cal_confidence,
        )

        ratio_w = max(pw, vw) / max(min(pw, vw), 0.1)
        ratio_h = max(ph, vh) / max(min(ph, vh), 0.1)
        agree = ratio_w < 1.3 and ratio_h < 1.3

        vc = vision_meas.confidence
        pc = meas_result.confidence
        strategy = "No change"

        if agree:
            total = vc + pc
            if total > 0:
                blend_w = (pw * pc + vw * vc) / total
                blend_h = (ph * pc + vh * vc) / total
                meas_result.width_mm = round(blend_w, 2)
                meas_result.height_mm = round(blend_h, 2)
                meas_result.area_mm2 = round(blend_w * blend_h, 2)
                meas_result.perimeter_mm = round(2 * (blend_w + blend_h), 2)
                meas_result.confidence = round(max(pc, vc), 2)
                strategy = "Blended (agreement within 30%%)"
        elif cal_confidence < 0.6 and vc > pc:
            # Calibration is estimated/weak — pixel dimensions are unreliable.
            # Trust vision which uses its own reference object for scale.
            meas_result.width_mm = round(vw, 2)
            meas_result.height_mm = round(vh, 2)
            meas_result.area_mm2 = round(vw * vh, 2)
            meas_result.perimeter_mm = round(2 * (vw + vh), 2)
            meas_result.confidence = round(vc, 2)
            strategy = "Vision override (cal unreliable, vision more confident)"
        else:
            # Both have reasonable calibration — weighted blend.
            logger.warning(
                "Vision/pixel disagree: ratio_w=%.2f ratio_h=%.2f — blending",
                ratio_w, ratio_h,
            )
            total = vc + pc
            if total > 0:
                blend_w = (pw * pc + vw * vc) / total
                blend_h = (ph * pc + vh * vc) / total
                meas_result.width_mm = round(blend_w, 2)
                meas_result.height_mm = round(blend_h, 2)
                meas_result.area_mm2 = round(blend_w * blend_h, 2)
                meas_result.perimeter_mm = round(2 * (blend_w + blend_h), 2)
                meas_result.confidence = round(min(max(pc, vc), vc * 0.9), 2)
            strategy = "Blended (disagreement, weighted)"

        logger.info(
            "Measurement strategy: %s -> %.1fx%.1f mm",
            strategy, meas_result.width_mm, meas_result.height_mm,
        )

        job.reasoning_log.append(ReasoningEntry(
            agent="VisionMeasurement",
            stage="vision_cross_check",
            reasoning=(
                f"Vision LLM estimated break: {vw:.1f}x{vh:.1f}x"
                f"{vision_meas.thickness_mm:.1f}mm ({vision_meas.description}). "
                f"Pixel-based: {pw:.1f}x{ph:.1f}mm. "
                f"Strategy: {strategy}."
            ),
            suggestions=[],
            confidence=vision_meas.confidence,
        ))

    sanity_penalty = _validate_measurement_sanity(meas_result)
    if sanity_penalty < 1.0:
        meas_result.confidence = round(meas_result.confidence * sanity_penalty, 2)
        logger.info(
            "Sanity check penalty: %.2f -> confidence now %.2f",
            sanity_penalty, meas_result.confidence,
        )

    return meas_result, vision_meas


def _scale_vision_polygon_to_measurements(
    contours: list[np.ndarray],
    meas_result,
    scale_factor: float,
) -> list[np.ndarray]:
    """Scale a vision polygon so its pixel bbox matches the measured mm dimensions."""
    if not contours or len(contours[0]) < 3:
        return contours

    contour = contours[0]
    x, y, w_px, h_px = cv2.boundingRect(contour)

    if w_px < 1 or h_px < 1 or scale_factor <= 0:
        return contours

    w_mm_actual = w_px * scale_factor
    h_mm_actual = h_px * scale_factor
    w_mm_target = meas_result.width_mm
    h_mm_target = meas_result.height_mm

    if w_mm_actual <= 0 or h_mm_actual <= 0:
        return contours

    ratio_w = w_mm_target / w_mm_actual
    ratio_h = h_mm_target / h_mm_actual

    if 0.7 < ratio_w < 1.3 and 0.7 < ratio_h < 1.3:
        return contours

    avg_ratio = (ratio_w + ratio_h) / 2
    cx = x + w_px / 2.0
    cy = y + h_px / 2.0

    pts = contour.reshape(-1, 2).astype(np.float64)
    pts[:, 0] = cx + (pts[:, 0] - cx) * avg_ratio
    pts[:, 1] = cy + (pts[:, 1] - cy) * avg_ratio
    scaled = pts.astype(np.int32).reshape(-1, 1, 2)

    logger.info(
        "Scaled vision polygon: actual %.1fx%.1f mm -> target %.1fx%.1f mm (factor %.2f)",
        w_mm_actual, h_mm_actual, w_mm_target, h_mm_target, avg_ratio,
    )
    return [scaled]


async def run_before_after_analysis(
    job: Job,
    before_image: np.ndarray,
    after_image: np.ndarray,
    marker_size_mm: float = 40.0,
    on_progress: Optional[EventCallback] = None,
) -> Job:
    """Run Before/After comparison -> Calibration -> Measurement -> Thickness."""

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, {"type": "status", "status": status.value})

    def _emit(event: dict):
        if on_progress:
            on_progress(job.id, event)

    def _log_and_emit(agent_name: str, stage: str, result):
        _log_reasoning(job, agent_name, stage, result)
        entry = job.reasoning_log[-1]
        _emit({"type": "reasoning", "entry": entry.model_dump()})

    job.reasoning_log = []

    try:
        # --- Before/After Comparison ---
        _notify(JobStatus.COMPARING_IMAGES)
        from app.pipeline.before_after import detect_damage

        ba_result = await asyncio.to_thread(
            detect_damage, before_image, after_image,
        )

        ba_mask = ba_result.mask
        job.detection_mode = DetectionMode.BEFORE_AFTER
        job.before_after_stats = {
            "num_matches": ba_result.num_matches,
            "alignment_confidence": ba_result.alignment_confidence,
            "damage_coverage": ba_result.damage_coverage,
            "centroid": list(ba_result.centroid),
        }

        job.click_points = [
            {"x": ba_result.centroid[0], "y": ba_result.centroid[1], "label": 1}
        ]

        vision_loc = await _vision_locate_damage(after_image)
        if vision_loc is not None:
            cx, cy = vision_loc
            job.click_points = [{"x": cx, "y": cy, "label": 1}]
            logger.info("Using vision-located damage center: (%d, %d)", cx, cy)
        else:
            cx, cy = ba_result.centroid
            logger.info("Vision locate failed, using before/after centroid: (%d, %d)", cx, cy)

        # --- SAM 2 refinement ---
        used_vision_loc = vision_loc is not None
        h_img, w_img = after_image.shape[:2]
        img_area = h_img * w_img

        neg_points = _generate_negative_points(cx, cy, w_img, h_img)
        sam_points = [{"x": cx, "y": cy, "label": 1}] + neg_points
        logger.info(
            "SAM 2 prompt: positive=(%d,%d) + %d negative points",
            cx, cy, len(neg_points),
        )

        try:
            _notify(JobStatus.SEGMENTING)
            sam_mask, sam_contours, seg_analysis = await _seg_agent.run(
                after_image, sam_points,
            )
            _log_and_emit("SegmentationAgent", "sam2_refinement", seg_analysis)

            sam_area = int(np.sum(sam_mask > 0))
            ba_area = int(np.sum(ba_mask > 0))
            sam_ratio = sam_area / img_area

            max_acceptable = 0.15
            if sam_ratio > max_acceptable:
                logger.info(
                    "SAM 2 covers %.1f%% of image (>%.0f%%) — likely segmented "
                    "entire object, rejecting",
                    sam_ratio * 100, max_acceptable * 100,
                )
                mask = ba_mask
                contours = ba_result.contours
            elif sam_area > img_area * 0.0005:
                mask = sam_mask
                contours = sam_contours
                logger.info(
                    "SAM 2 refinement accepted: %d px (%.2f%% of image)",
                    sam_area, sam_ratio * 100,
                )
            else:
                mask = ba_mask
                contours = ba_result.contours
                logger.info(
                    "SAM 2 mask rejected (area %d, %.2f%%), keeping before/after mask",
                    sam_area, sam_ratio * 100,
                )
        except Exception as sam_err:
            logger.warning("SAM 2 refinement failed (non-fatal): %s", sam_err)
            mask = ba_mask
            contours = ba_result.contours

        # Emit SAM 2 overlay visualization BEFORE vision polygon overrides
        try:
            from app.pipeline.visualization import create_sam2_overlay
            overlay = create_sam2_overlay(after_image, mask, contours)
            viz_path = job_viz_path(job.id, "sam2_mask")
            cv2.imwrite(str(viz_path), overlay)
            _emit({"type": "viz", "name": "sam2_mask",
                    "url": f"/api/v1/jobs/{job.id}/viz/sam2_mask"})
        except Exception as viz_err:
            logger.warning("SAM 2 visualization failed (non-fatal): %s", viz_err)

        # --- Vision polygon (always called — primary contour source) ---
        vision_poly = await _vision_get_break_polygon(after_image)
        if vision_poly is not None:
            poly_mask = np.zeros(after_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(poly_mask, [vision_poly], -1, 255, -1)
            mask = poly_mask
            contours = [vision_poly]
            logger.info("Using vision-traced polygon as primary damage contour")
        else:
            mask_coverage = int(np.sum(mask > 0)) / img_area if img_area > 0 else 0
            logger.info(
                "Vision polygon unavailable — using pixel mask (coverage %.2f%%)",
                mask_coverage * 100,
            )

        job.contours = contours
        mask_path = job_mask_path(job.id)
        cv2.imwrite(str(mask_path), mask)

        _notify(JobStatus.SEGMENTED)

        logger.info(
            "Before/after for %s: %d matches, %.1f%% confidence, %.2f%% coverage",
            job.id, ba_result.num_matches,
            ba_result.alignment_confidence * 100,
            ba_result.damage_coverage * 100,
        )

        # --- Calibration Agent (consensus on the "after" image) ---
        _notify(JobStatus.CALIBRATING)
        ba_depth_map = None
        try:
            cal_result, cal_analysis, ba_depth_map = await _cal_agent.run(
                after_image,
                marker_size_mm=marker_size_mm,
                original_upload_path=job.original_upload_path,
            )
            if cal_result.scale_factor <= 0 or cal_result.method == "none":
                h_img, w_img = after_image.shape[:2]
                vision_scale = await _vision_calibration_fallback(after_image, h_img, w_img)
                if vision_scale:
                    cal_result = CalibrationResult(scale_factor=vision_scale, method="vision_estimated", confidence=0.5)
                else:
                    cal_result = CalibrationResult(scale_factor=100.0 / max(h_img, w_img, 1), method="estimated", confidence=0.3)
            job.calibration = cal_result
            _log_and_emit("CalibrationAgent", "calibration_consensus", cal_analysis)
        except Exception as cal_err:
            logger.warning("Calibration consensus failed: %s", cal_err)
            h_img, w_img = after_image.shape[:2]
            vision_scale = await _vision_calibration_fallback(after_image, h_img, w_img)
            estimated_scale = vision_scale if vision_scale else 100.0 / max(h_img, w_img, 1)
            cal_method = "vision_estimated" if vision_scale else "estimated"
            cal_confidence = 0.5 if vision_scale else 0.3
            job.calibration = CalibrationResult(scale_factor=estimated_scale, method=cal_method, confidence=cal_confidence)
            entry = ReasoningEntry(
                agent="CalibrationAgent", stage="calibration_consensus",
                reasoning=f"Calibration consensus failed ({cal_err}). Using {cal_method} scale ({estimated_scale:.4f} mm/px).",
                suggestions=["Add an ArUco marker", "Draw a reference line"], confidence=cal_confidence,
            )
            job.reasoning_log.append(entry)
            _emit({"type": "reasoning", "entry": entry.model_dump()})
        _notify(JobStatus.CALIBRATED)

        # --- Measurement Agent (pixel-based) ---
        _notify(JobStatus.MEASURING)
        ba_cal = job.calibration
        meas_result, meas_analysis = await _meas_agent.run(
            contours, ba_cal.scale_factor, ba_cal.method,
            calibration_confidence=ba_cal.confidence,
            image_bgr=after_image, mask=mask,
        )
        _log_and_emit("MeasurementAgent", "measurement", meas_analysis)

        meas_result, vision_meas = await _integrate_measurements(
            meas_result, after_image, ba_cal.confidence, job,
        )
        if job.reasoning_log and job.reasoning_log[-1].agent == "VisionMeasurement":
            _emit({"type": "reasoning", "entry": job.reasoning_log[-1].model_dump()})

        job.contours = _scale_vision_polygon_to_measurements(
            contours, meas_result, ba_cal.scale_factor,
        )
        job.measurement = meas_result
        _notify(JobStatus.MEASURED)

        # --- Thickness Agent (consensus with shared depth map) ---
        _notify(JobStatus.ESTIMATING_DEPTH)
        vision_thickness = vision_meas.thickness_mm if vision_meas else None
        thick_result, thick_analysis = await _thick_agent.run(
            mask=mask,
            scale_factor=ba_cal.scale_factor,
            measurement_width_mm=meas_result.width_mm,
            measurement_height_mm=meas_result.height_mm,
            original_upload_path=job.original_upload_path,
            key_frame_paths=job.key_frame_paths if job.key_frame_paths else None,
            side_image_path=job.side_image_path,
            depth_map=ba_depth_map,
            vision_thickness_mm=vision_thickness,
            calibration_method=ba_cal.method,
            image_bgr=after_image,
        )

        job.thickness_result = thick_result
        job.thickness_mm = thick_result.thickness_mm
        _log_and_emit("ThicknessAgent", "thickness_consensus", thick_analysis)

        # Generate and emit Depth Anything visualization
        try:
            from app.pipeline.visualization import create_depth_visualization
            depth_viz = await asyncio.to_thread(create_depth_visualization, after_image, mask)
            if depth_viz is not None:
                viz_path = job_viz_path(job.id, "depth_map")
                cv2.imwrite(str(viz_path), depth_viz)
                _emit({"type": "viz", "name": "depth_map",
                        "url": f"/api/v1/jobs/{job.id}/viz/depth_map"})
        except Exception as viz_err:
            logger.warning("Depth visualization failed (non-fatal): %s", viz_err)

        if thick_result.method != ThicknessMethod.MANUAL:
            _notify(JobStatus.DEPTH_ESTIMATED)
        else:
            _notify(JobStatus.MEASURED)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
        store_job(job)
        raise

    store_job(job)
    return job


async def run_mesh_generation(
    job: Job,
    thickness_mm: float = 3.0,
    chamfer_mm: float = 0.0,
    on_progress: Optional[EventCallback] = None,
) -> Job:
    """Run Mesh Agent -> Validation Agent."""

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, {"type": "status", "status": status.value})

    if not job.contours or job.calibration is None or job.measurement is None:
        job.status = JobStatus.FAILED
        job.error = "Cannot generate mesh: analysis not completed (missing contours, calibration, or measurement)."
        store_job(job)
        raise RuntimeError(job.error)

    try:
        # --- Mesh Agent ---
        _notify(JobStatus.GENERATING_MESH)
        job.thickness_mm = thickness_mm
        output_path = job_mesh_path(job.id)

        mesh_result, mesh_analysis = await _mesh_agent.run(
            job.contours[0],
            job.calibration.scale_factor,
            thickness_mm,
            output_path,
            chamfer_mm=chamfer_mm,
        )
        job.mesh = mesh_result
        _log_reasoning(job, "MeshAgent", "mesh_generation", mesh_analysis)

        # --- Validation Agent ---
        _notify(JobStatus.VALIDATING)
        val_analysis = await _val_agent.run(
            job.calibration,
            job.measurement,
            mesh_result,
            thickness_mm,
        )
        _log_reasoning(job, "ValidationAgent", "validation", val_analysis)

        _notify(JobStatus.READY)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
        store_job(job)
        raise

    store_job(job)
    return job


async def run_prompt_mesh_generation(
    job: Job,
    thickness_mm: float | None = None,
    chamfer_mm: float = 0.0,
    on_progress: EventCallback | None = None,
) -> Job:
    """Generate a mesh from a prompt-parsed shape description."""
    from app.pipeline.prompt_to_mesh import generate_mesh_from_shape

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, {"type": "status", "status": status.value})

    if not job.parsed_shape:
        job.status = JobStatus.FAILED
        job.error = "Cannot generate mesh: no parsed shape from prompt."
        store_job(job)
        raise RuntimeError(job.error)

    try:
        _notify(JobStatus.GENERATING_MESH)

        shape = dict(job.parsed_shape)
        if thickness_mm is not None:
            shape["thickness_mm"] = thickness_mm

        output_path = job_mesh_path(job.id)
        mesh_result = generate_mesh_from_shape(shape, output_path, chamfer_mm)
        job.mesh = mesh_result
        job.thickness_mm = shape["thickness_mm"]

        job.reasoning_log.append(ReasoningEntry(
            agent="PromptMeshGenerator",
            stage="mesh_generation",
            reasoning=(
                f"Generated {shape['shape_type']} mesh from prompt: "
                f"{shape.get('description', '')}. "
                f"Dimensions: {shape['width_mm']}x{shape['height_mm']}x"
                f"{shape['thickness_mm']} mm."
            ),
            suggestions=[],
            confidence=0.95,
        ))

        _notify(JobStatus.VALIDATING)
        val_analysis = await _val_agent.run(
            None,
            job.measurement,
            mesh_result,
            shape["thickness_mm"],
        )
        _log_reasoning(job, "ValidationAgent", "validation", val_analysis)

        _notify(JobStatus.READY)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
        store_job(job)
        raise

    store_job(job)
    return job


async def run_print_check(
    job: Job,
    on_progress: Optional[EventCallback] = None,
) -> Job:
    """Run a single PrinterAgent check on the current print status."""
    from app.pipeline import printer as printer_svc

    if not printer_svc.is_connected():
        return job

    status = await asyncio.to_thread(printer_svc.get_status)
    job.print_status = status

    filament = "PLA"
    if job.printer_config:
        filament = job.printer_config.filament

    try:
        analysis = await _print_agent.run(
            status=status,
            mesh=job.mesh,
            filament=filament,
        )
        _log_reasoning(job, "PrinterAgent", "print_monitoring", analysis)
    except Exception as e:
        logger.warning("PrinterAgent analysis failed: %s", e)

    if (
        job.status == JobStatus.PRINTING
        and status.progress_pct is not None
        and status.progress_pct >= 100
    ):
        job.status = JobStatus.PRINT_COMPLETE

    store_job(job)
    return job


