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
from app.core.storage import job_mask_path, job_mesh_path, job_propagated_masks_dir
from app.models.job import (
    Job, JobStatus, ReasoningEntry, ThicknessMethod, ThicknessResult,
    UploadType, DetectionMode, CalibrationResult,
)

from app.core.job_store import get_job, store_job

logger = logging.getLogger("patchforge.orchestrator")


ProgressCallback = Callable[[str, JobStatus], None]

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
        if max(h_img, w_img) > max_dim:
            s = max_dim / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img * s), int(h_img * s)))

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = buf.tobytes()

        system = (
            "You are a calibration expert. Analyze images to identify objects of known "
            "real-world size so we can compute a mm-per-pixel scale factor."
        )
        prompt = (
            f"This image is {w_img}x{h_img} pixels. I need to determine the real-world "
            "scale (mm per pixel). There is no calibration marker.\n\n"
            "Look for ANY objects in the image whose real-world size you can estimate:\n"
            "- Fingers/hand (index finger ~17mm wide)\n"
            "- Coins, USB ports, screws, pens, credit cards\n"
            "- Standard brick, tile, keyboard keys\n"
            "- The broken object itself if you can identify what it is\n\n"
            "Estimate how many pixels wide that known object appears in the image, "
            "then compute mm_per_pixel = known_mm / pixel_width.\n\n"
            "Respond with ONLY JSON:\n"
            '{"reference_object": "<what you identified>", '
            '"reference_size_mm": <real-world width in mm>, '
            '"reference_size_px": <pixel width in image>, '
            '"mm_per_pixel": <computed scale>, '
            '"confidence": <0.0-1.0>}'
        )

        text, provider = await asyncio.to_thread(
            call_llm_vision, system, prompt, image_bytes,
        )
        logger.info("Vision calibration via %s", provider)

        parsed = parse_json_response(text)
        scale = float(parsed.get("mm_per_pixel", 0))
        confidence = float(parsed.get("confidence", 0))
        ref_obj = parsed.get("reference_object", "unknown")

        if scale > 0.001 and confidence >= 0.3:
            logger.info(
                "Vision calibration: %.5f mm/px from %s (conf=%.2f)",
                scale, ref_obj, confidence,
            )
            return scale

    except Exception as e:
        logger.warning("Vision calibration fallback failed: %s", e)

    return None


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

        system = (
            "You are a computer vision expert specializing in damage detection "
            "for 3D-printable repair parts."
        )
        prompt = (
            f"This image is {w_img}x{h_img} pixels. It shows a broken or damaged "
            "object — a piece has been broken off or is missing.\n\n"
            "Find the broken/missing area and return the PIXEL coordinates of "
            "its approximate center. Look for:\n"
            "- A gap, void, or missing chunk in the object\n"
            "- A visible break line or crack\n"
            "- An area where material is clearly absent\n\n"
            "Respond with ONLY JSON:\n"
            '{"x": <pixel x coordinate of damage center>, '
            '"y": <pixel y coordinate of damage center>, '
            '"description": "<brief description of the damage>", '
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
            "You can estimate real-world dimensions from photos by comparing to "
            "objects of known size."
        )
        prompt = (
            "This image shows a broken/damaged object. A piece has been broken off "
            "and we need to 3D-print a replacement patch that fits into the gap.\n\n"
            "YOUR TASK: Measure the FULL EXTENT of the missing piece — imagine the "
            "piece that broke off and was removed. You are measuring that missing "
            "piece so we can print a replacement. Include the ENTIRE break area "
            "from edge to edge of the damage, not just the deepest notch.\n\n"
            "STEP 1 — Find a reference object for scale:\n"
            "  * Coins: US quarter=24.26mm, US penny=19.05mm, Euro 1€=23.25mm, "
            "Euro 2€=25.75mm, UK £1=23.43mm, UK £2=28.4mm, "
            "ZAR R5=26mm, ZAR R2=23mm, ZAR R1=20mm, ZAR 50c=22mm\n"
            "  * Fingers: adult index finger width ~17mm, thumb ~20mm\n"
            "  * Credit card: 85.6mm x 54mm\n\n"
            "STEP 2 — Proportional reasoning (MANDATORY — do this exactly):\n"
            "  a) Estimate the coin/reference DIAMETER in pixels in the image\n"
            "  b) Estimate the break WIDTH in pixels in the image\n"
            "  c) Estimate the break HEIGHT in pixels in the image\n"
            "  d) Compute: width_mm = coin_mm * (break_width_px / coin_px)\n"
            "  e) Compute: height_mm = coin_mm * (break_height_px / coin_px)\n\n"
            "  WORKED EXAMPLE:\n"
            "    Coin: Euro 1€ (23mm), appears ~100px wide in image\n"
            "    Break: appears ~65px wide, ~130px tall\n"
            "    width = 23 * (65/100) = 14.95mm ≈ 15mm\n"
            "    height = 23 * (130/100) = 29.9mm ≈ 30mm\n\n"
            "  COMMON MISTAKE: Do NOT underestimate the pixel sizes. Measure from "
            "the outermost edges of the break, including the full gap.\n\n"
            "STEP 3 — Measure the missing piece:\n"
            "  * width_mm = FULL width from outermost edge to outermost edge\n"
            "  * height_mm = FULL height from outermost edge to outermost edge\n"
            "  * Add 10%% margin — too big can be sanded, too small won't fit\n\n"
            "STEP 4 — Estimate thickness (CRITICAL — follow this carefully):\n"
            "  Option A (preferred): If the break EDGE is visible (you can see the wall\n"
            "  cross-section at the fracture), measure it with proportional reasoning:\n"
            "    * Estimate the edge/wall thickness in pixels\n"
            "    * thickness_mm = coin_mm * (edge_thickness_px / coin_px)\n"
            "  Option B: If the break edge is NOT visible (top-down only), infer from\n"
            "  the object type and proportions:\n"
            "    * 3D-printed plastic walls: typically 2-4mm\n"
            "    * Injection-molded plastic: typically 1.5-3mm\n"
            "    * Ceramic/porcelain: typically 3-8mm\n"
            "    * General rule: thickness ≈ width / 5 to width / 3\n"
            "  * Report which option you used in thickness_reasoning\n\n"
            "STEP 5 — Visual cross-check (MANDATORY):\n"
            "  * Visually compare the break to the coin side-by-side\n"
            "  * If the break appears WIDER than the coin, width MUST be > coin diameter\n"
            "  * If the break appears TALLER than the coin, height MUST be > coin diameter\n"
            "  * If the break is roughly 2/3 the coin width, measurement ≈ 2/3 of coin diameter\n"
            "  * If your measurements violate these visual checks, redo the calculation\n\n"
            "Respond with ONLY JSON:\n"
            '{"width_mm": <number>, '
            '"height_mm": <number>, '
            '"thickness_mm": <number>, '
            '"reference_object": "<what you identified>", '
            '"ref_px": <reference object width in pixels>, '
            '"break_width_px": <break width in pixels>, '
            '"break_height_px": <break height in pixels>, '
            '"reasoning": "<show coin_mm * (break_px / coin_px) calculation>", '
            '"thickness_reasoning": "<explain how thickness was estimated: Option A or B, show calculation>", '
            '"description": "<describe the break shape>", '
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
            "You are a precise computer vision expert. You can identify break "
            "boundaries in images and trace their outlines as polygon coordinates."
        )
        prompt = (
            f"This image is {disp_w}x{disp_h} pixels. It shows a broken object "
            "where a piece has been broken off.\n\n"
            "TASK: Trace the OUTLINE of the missing/broken area as a polygon.\n"
            "Return the corner points of the break boundary in PIXEL coordinates.\n\n"
            "RULES:\n"
            "- Trace ONLY the break/gap boundary, not the whole object\n"
            "- Follow the break edges — where material was removed\n"
            "- Include the straight edges of the original object that border the gap\n"
            "- Use 4-8 points that define the break shape\n"
            "- For a rectangular break: 4 corners\n"
            "- For a triangular break (corner chip): 3-4 points\n"
            "- For an irregular break: 5-8 points following the edge\n"
            "- Points should go clockwise around the break boundary\n"
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
    on_progress: Optional[ProgressCallback] = None,
) -> Job:
    """Run Calibration -> Segmentation -> Measurement -> Thickness agents."""

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, status)

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

            # If the consensus returned no usable result, try vision fallback
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
            _log_reasoning(job, "CalibrationAgent", "calibration_consensus", cal_analysis)
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
            job.reasoning_log.append(ReasoningEntry(
                agent="CalibrationAgent",
                stage="calibration_consensus",
                reasoning=f"Calibration consensus failed ({cal_err}). Using {cal_method} scale "
                          f"({estimated_scale:.4f} mm/px).",
                suggestions=["Add an ArUco marker to the scene", "Draw a reference line"],
                confidence=cal_confidence,
            ))
        _notify(JobStatus.CALIBRATED)

        # --- Segmentation Agent ---
        _notify(JobStatus.SEGMENTING)
        mask = None
        contours = None
        try:
            mask, contours, seg_analysis = await _seg_agent.run(image, points)
            _log_reasoning(job, "SegmentationAgent", "segmentation", seg_analysis)
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
        _log_reasoning(job, "MeasurementAgent", "measurement", meas_analysis)

        # --- Vision Measurement (primary) + integration ---
        meas_result, vision_meas = await _integrate_measurements(
            meas_result, image, cal_result.confidence, job,
        )

        job.contours = _scale_vision_polygon_to_measurements(
            contours, meas_result, cal_result.scale_factor,
        )
        job.measurement = meas_result
        _notify(JobStatus.MEASURED)

        # --- Thickness Agent (consensus: run ALL strategies, LLM picks) ---
        # Pass the depth_map extracted during calibration to avoid redundant I/O
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
        _log_reasoning(job, "ThicknessAgent", "thickness_consensus", thick_analysis)

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

        if agree:
            vc = vision_meas.confidence
            pc = meas_result.confidence
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
        elif vision_meas.confidence >= 0.5:
            meas_result.width_mm = round(vw, 2)
            meas_result.height_mm = round(vh, 2)
            meas_result.area_mm2 = round(vw * vh, 2)
            meas_result.perimeter_mm = round(2 * (vw + vh), 2)
            meas_result.confidence = round(vision_meas.confidence, 2)
            strategy = "Vision override (primary, conf >= 0.5)"
        elif cal_confidence < 0.6 or meas_result.confidence < 0.6:
            meas_result.width_mm = round(vw, 2)
            meas_result.height_mm = round(vh, 2)
            meas_result.area_mm2 = round(vw * vh, 2)
            meas_result.perimeter_mm = round(2 * (vw + vh), 2)
            meas_result.confidence = round(vision_meas.confidence, 2)
            strategy = "Vision override (pixel unreliable)"
        else:
            vc = vision_meas.confidence
            pc = meas_result.confidence
            total = vc + pc
            if total > 0:
                blend_w = (pw * pc + vw * vc) / total
                blend_h = (ph * pc + vh * vc) / total
                meas_result.width_mm = round(blend_w, 2)
                meas_result.height_mm = round(blend_h, 2)
                meas_result.area_mm2 = round(blend_w * blend_h, 2)
                meas_result.perimeter_mm = round(2 * (blend_w + blend_h), 2)
                meas_result.confidence = round(max(pc, vc), 2)
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
    on_progress: Optional[ProgressCallback] = None,
) -> Job:
    """Run Before/After comparison -> Calibration -> Measurement -> Thickness."""

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, status)

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

        # --- Vision-based damage localization ---
        # The before/after centroid is unreliable when the object moved between
        # shots. Ask the LLM to visually identify where the break is.
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
        img_area = ba_mask.shape[0] * ba_mask.shape[1]
        try:
            _notify(JobStatus.SEGMENTING)
            sam_mask, sam_contours, seg_analysis = await _seg_agent.run(
                after_image, [{"x": cx, "y": cy, "label": 1}],
            )
            _log_reasoning(job, "SegmentationAgent", "sam2_refinement", seg_analysis)

            sam_area = int(np.sum(sam_mask > 0))
            ba_area = int(np.sum(ba_mask > 0))
            sam_ratio = sam_area / img_area

            # SAM 2 segments OBJECTS, not damage. If it covers >40% of the image
            # it grabbed the whole object, not the break. We never want that.
            if sam_ratio > 0.40:
                logger.info(
                    "SAM 2 covers %.1f%% of image — segmented entire object, rejecting",
                    sam_ratio * 100,
                )
                mask = ba_mask
                contours = ba_result.contours
            elif sam_area > img_area * 0.0005 and (
                not used_vision_loc and sam_area < ba_area * 2.5
                or used_vision_loc
            ):
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
            _log_reasoning(job, "CalibrationAgent", "calibration_consensus", cal_analysis)
        except Exception as cal_err:
            logger.warning("Calibration consensus failed: %s", cal_err)
            h_img, w_img = after_image.shape[:2]
            vision_scale = await _vision_calibration_fallback(after_image, h_img, w_img)
            estimated_scale = vision_scale if vision_scale else 100.0 / max(h_img, w_img, 1)
            cal_method = "vision_estimated" if vision_scale else "estimated"
            cal_confidence = 0.5 if vision_scale else 0.3
            job.calibration = CalibrationResult(scale_factor=estimated_scale, method=cal_method, confidence=cal_confidence)
            job.reasoning_log.append(ReasoningEntry(
                agent="CalibrationAgent", stage="calibration_consensus",
                reasoning=f"Calibration consensus failed ({cal_err}). Using {cal_method} scale ({estimated_scale:.4f} mm/px).",
                suggestions=["Add an ArUco marker", "Draw a reference line"], confidence=cal_confidence,
            ))
        _notify(JobStatus.CALIBRATED)

        # --- Measurement Agent (pixel-based) ---
        _notify(JobStatus.MEASURING)
        ba_cal = job.calibration
        meas_result, meas_analysis = await _meas_agent.run(
            contours, ba_cal.scale_factor, ba_cal.method,
            calibration_confidence=ba_cal.confidence,
            image_bgr=after_image, mask=mask,
        )
        _log_reasoning(job, "MeasurementAgent", "measurement", meas_analysis)

        meas_result, vision_meas = await _integrate_measurements(
            meas_result, after_image, ba_cal.confidence, job,
        )

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
        _log_reasoning(job, "ThicknessAgent", "thickness_consensus", thick_analysis)

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
    on_progress: Optional[ProgressCallback] = None,
) -> Job:
    """Run Mesh Agent -> Validation Agent."""

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, status)

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
    on_progress: ProgressCallback | None = None,
) -> Job:
    """Generate a mesh from a prompt-parsed shape description."""
    from app.pipeline.prompt_to_mesh import generate_mesh_from_shape

    def _notify(status: JobStatus):
        job.status = status
        store_job(job)
        if on_progress:
            on_progress(job.id, status)

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
    on_progress: Optional[ProgressCallback] = None,
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


