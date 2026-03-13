from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect

from fastapi.responses import FileResponse

from app.core.storage import job_side_image_path, job_reference_image_path, job_mask_path, job_viz_path
from app.models.job import Job, JobStatus, DetectionMode
from app.models.schemas import JobResponse, SegmentRequest, GenerateRequest
from app.agents.orchestrator import get_job, store_job, run_analysis, run_before_after_analysis, run_mesh_generation, run_prompt_mesh_generation
from app.api.helpers import job_to_response as _job_to_response


def _decode_image_bytes(
    contents: bytes, filename: str = "", content_type: str = "",
) -> Optional[np.ndarray]:
    """Decode image bytes to a BGR numpy array, with HEIC/HEIF support."""
    ext = Path(filename).suffix.lower() if filename else ""
    ct = content_type.lower()
    is_heif = (
        ext in (".heic", ".heif")
        or "heic" in ct
        or "heif" in ct
    )

    if is_heif:
        return _decode_heif(contents)

    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    return _decode_heif(contents)


def _decode_heif(contents: bytes) -> Optional[np.ndarray]:
    """Try decoding bytes as HEIC/HEIF via pillow-heif."""
    try:
        from pillow_heif import register_heif_opener
        from PIL import Image
        import io

        register_heif_opener()
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

router = APIRouter()

_ws_connections: dict[str, list[WebSocket]] = {}
_ws_lock = asyncio.Lock()


async def _ws_send_raw(job_id: str, payload: dict):
    """Send an arbitrary JSON message to all WebSocket connections for a job."""
    async with _ws_lock:
        conns = list(_ws_connections.get(job_id, []))
    for ws in conns:
        try:
            await ws.send_json(payload)
        except Exception:
            pass


async def _ws_notify(job_id: str, status: JobStatus):
    """Legacy status-only notification (kept for backward compatibility)."""
    await _ws_send_raw(job_id, {"job_id": job_id, "status": status.value})


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return _job_to_response(job)


@router.post("/jobs/{job_id}/segment", response_model=JobResponse)
async def segment_job(job_id: str, req: SegmentRequest):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status not in (
        JobStatus.UPLOADED, JobStatus.FRAMES_READY, JobStatus.MASKS_PROPAGATED,
        JobStatus.MEASURED, JobStatus.DEPTH_ESTIMATED, JobStatus.FAILED,
    ):
        raise HTTPException(409, f"Job is currently {job.status.value}; cannot restart analysis.")

    positive = [p for p in req.points if p.label == 1]
    if not positive:
        raise HTTPException(400, "At least one positive click point is required.")

    image_path = job.image_path
    if (
        req.frame_index is not None
        and job.key_frame_paths
        and 0 <= req.frame_index < len(job.key_frame_paths)
    ):
        image_path = job.key_frame_paths[req.frame_index]

    if not image_path:
        raise HTTPException(409, "No image attached to this job.")
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(500, "Stored image could not be read.")

    points = [p.model_dump() for p in req.points]
    job.click_points = points

    def on_progress(jid: str, msg: dict):
        try:
            loop = asyncio.get_running_loop()
            msg.setdefault("job_id", jid)
            loop.call_soon_threadsafe(asyncio.ensure_future, _ws_send_raw(jid, msg))
        except RuntimeError:
            pass

    try:
        job = await run_analysis(
            job, image, points,
            marker_size_mm=req.marker_size_mm,
            ref_line_start=req.ref_line_start,
            ref_line_end=req.ref_line_end,
            ref_line_mm=req.ref_line_mm,
            webxr_scale=req.webxr_scale_mm_per_px,
            on_progress=on_progress,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    return _job_to_response(job)


@router.post("/jobs/{job_id}/generate", response_model=JobResponse)
async def generate_mesh_endpoint(job_id: str, req: GenerateRequest):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status not in (
        JobStatus.MEASURED, JobStatus.DEPTH_ESTIMATED,
        JobStatus.READY, JobStatus.FAILED,
    ):
        raise HTTPException(409, f"Job must be in 'measured' or 'depth_estimated' state, currently: {job.status.value}.")

    # Resolve thickness: explicit request > auto-inferred > default
    if req.thickness_mm is not None:
        thickness = req.thickness_mm
    elif job.thickness_result is not None:
        thickness = job.thickness_result.thickness_mm
    else:
        thickness = job.thickness_mm

    def on_progress(jid: str, msg: dict):
        try:
            loop = asyncio.get_running_loop()
            msg.setdefault("job_id", jid)
            loop.call_soon_threadsafe(asyncio.ensure_future, _ws_send_raw(jid, msg))
        except RuntimeError:
            pass

    try:
        if job.detection_mode == DetectionMode.PROMPT:
            job = await run_prompt_mesh_generation(
                job,
                thickness_mm=thickness,
                chamfer_mm=req.chamfer_mm,
                on_progress=on_progress,
            )
        else:
            job = await run_mesh_generation(
                job,
                thickness_mm=thickness,
                chamfer_mm=req.chamfer_mm,
                on_progress=on_progress,
            )
    except Exception as e:
        raise HTTPException(500, str(e))

    return _job_to_response(job)


@router.post("/jobs/{job_id}/reference", response_model=JobResponse)
async def upload_reference_image(
    job_id: str,
    file: UploadFile = File(...),
):
    """Upload a 'before' reference image for automatic damage detection."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status not in (
        JobStatus.UPLOADED, JobStatus.FRAMES_READY, JobStatus.MASKS_PROPAGATED,
        JobStatus.MEASURED, JobStatus.DEPTH_ESTIMATED, JobStatus.FAILED,
    ):
        raise HTTPException(409, f"Cannot upload reference now ({job.status.value}).")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "Reference image exceeds 20 MB limit.")

    image = _decode_image_bytes(contents, file.filename or "", file.content_type or "")
    if image is None:
        raise HTTPException(400, "Could not decode reference image.")

    ref_path = job_reference_image_path(job_id)
    cv2.imwrite(str(ref_path), image)
    job.reference_image_path = str(ref_path)
    job.detection_mode = DetectionMode.BEFORE_AFTER
    store_job(job)

    return _job_to_response(job)


@router.post("/jobs/{job_id}/auto-detect", response_model=JobResponse)
async def auto_detect_damage(
    job_id: str,
    marker_size_mm: float = 40.0,
):
    """Run before/after comparison to auto-detect damage without click points."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if not job.reference_image_path:
        raise HTTPException(409, "Upload a reference image first (POST /jobs/{id}/reference).")
    if job.status not in (
        JobStatus.UPLOADED, JobStatus.FRAMES_READY, JobStatus.MASKS_PROPAGATED,
        JobStatus.MEASURED, JobStatus.DEPTH_ESTIMATED, JobStatus.FAILED,
    ):
        raise HTTPException(409, f"Cannot run auto-detection now ({job.status.value}).")

    if not job.image_path:
        raise HTTPException(409, "No image attached to this job.")
    after_image = cv2.imread(job.image_path)
    if after_image is None:
        raise HTTPException(500, "Stored image could not be read.")

    before_image = cv2.imread(job.reference_image_path)
    if before_image is None:
        raise HTTPException(500, "Reference image could not be read.")

    def on_progress(jid: str, msg: dict):
        try:
            loop = asyncio.get_running_loop()
            msg.setdefault("job_id", jid)
            loop.call_soon_threadsafe(asyncio.ensure_future, _ws_send_raw(jid, msg))
        except RuntimeError:
            pass

    try:
        job = await run_before_after_analysis(
            job, before_image, after_image,
            marker_size_mm=marker_size_mm,
            on_progress=on_progress,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    return _job_to_response(job)


@router.post("/jobs/{job_id}/video", response_model=JobResponse)
async def attach_video(
    job_id: str,
    file: UploadFile = File(...),
):
    """Attach a video to an existing job for thickness estimation context.

    The video is processed into key frames that the thickness estimation
    pipeline uses for multi-view depth analysis.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status not in (
        JobStatus.UPLOADED, JobStatus.FRAMES_READY, JobStatus.MASKS_PROPAGATED,
        JobStatus.MEASURED, JobStatus.DEPTH_ESTIMATED, JobStatus.FAILED,
    ):
        raise HTTPException(409, f"Cannot attach video now ({job.status.value}).")

    contents = await file.read()
    if len(contents) > 200 * 1024 * 1024:
        raise HTTPException(400, "Video exceeds 200 MB limit.")

    from pathlib import Path
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        raise HTTPException(400, "Unsupported video format.")

    from app.core.storage import job_video_path
    video_path = job_video_path(job_id, ext)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(video_path), "wb") as f:
        f.write(contents)
    job.video_path = str(video_path)

    try:
        from app.pipeline.video_processing import extract_keyframes
        result = await asyncio.to_thread(extract_keyframes, str(video_path), job_id)
        if not result.frame_infos:
            raise ValueError("No key frames extracted from video.")
        job.key_frame_paths = [fi.path for fi in result.frame_infos]
        job.best_frame_index = result.best_frame_index
        store_job(job)
    except Exception as e:
        try:
            Path(str(video_path)).unlink(missing_ok=True)
        except OSError:
            pass
        job.video_path = None
        job.key_frame_paths = []
        store_job(job)
        raise HTTPException(500, f"Video processing failed: {e}")

    return _job_to_response(job)


@router.post("/jobs/{job_id}/side-photo", response_model=JobResponse)
async def upload_side_photo(
    job_id: str,
    file: UploadFile = File(...),
    manual_hint_mm: float | None = None,
):
    """Upload a side-angle photo for thickness estimation fallback."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status not in (
        JobStatus.MEASURED, JobStatus.DEPTH_ESTIMATED, JobStatus.FAILED,
    ):
        raise HTTPException(
            409,
            "Run segmentation first. Side photos are used after the damage area is identified.",
        )

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "Side photo exceeds 20 MB limit.")

    image = _decode_image_bytes(contents, file.filename or "", file.content_type or "")
    if image is None:
        raise HTTPException(400, "Could not decode side photo.")

    side_path = job_side_image_path(job_id)
    cv2.imwrite(str(side_path), image)
    job.side_image_path = str(side_path)
    store_job(job)

    # Re-run thickness estimation with the side photo now available
    mask_path = job_mask_path(job_id)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or job.calibration is None or job.measurement is None:
        raise HTTPException(409, "Analysis results missing — re-run segmentation first.")

    try:
        from app.pipeline.thickness_estimation import estimate_thickness

        result = await asyncio.to_thread(
            estimate_thickness,
            original_upload_path=job.original_upload_path,
            mask=mask,
            scale_factor=job.calibration.scale_factor,
            measurement_width_mm=job.measurement.width_mm,
            measurement_height_mm=job.measurement.height_mm,
            key_frame_paths=job.key_frame_paths if job.key_frame_paths else None,
            side_image_path=str(side_path),
            manual_hint_mm=manual_hint_mm,
        )

        job.thickness_result = result
        job.thickness_mm = result.thickness_mm
        job.status = JobStatus.DEPTH_ESTIMATED
        store_job(job)

    except Exception as e:
        raise HTTPException(500, f"Thickness estimation failed: {e}")

    return _job_to_response(job)


@router.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    """Delete a job and all associated files."""
    from app.core.job_store import delete_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    file_fields = [
        job.image_path, job.original_upload_path, job.video_path,
        job.reference_image_path, job.side_image_path,
    ]
    file_fields.extend(job.key_frame_paths)
    file_fields.extend(job.propagated_mask_paths)

    mask = job_mask_path(job_id)
    if mask.exists():
        file_fields.append(str(mask))

    if job.mesh and job.mesh.file_path:
        file_fields.append(job.mesh.file_path)

    for path_str in file_fields:
        if path_str:
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError:
                pass

    delete_job(job_id)
    return {"detail": f"Job {job_id} deleted."}


@router.get("/jobs/{job_id}/viz/{name}")
async def get_visualization(job_id: str, name: str):
    """Serve a visualization image (SAM 2 overlay or depth map)."""
    if name not in ("sam2_mask", "depth_map"):
        raise HTTPException(400, "Unknown visualization type.")
    path = job_viz_path(job_id, name)
    if not path.exists():
        raise HTTPException(404, "Visualization not yet available.")
    return FileResponse(str(path), media_type="image/jpeg")


@router.websocket("/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    async with _ws_lock:
        _ws_connections.setdefault(job_id, []).append(websocket)

    job = get_job(job_id)
    if job:
        await websocket.send_json({"job_id": job_id, "status": job.status.value})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            conns = _ws_connections.get(job_id, [])
            if websocket in conns:
                conns.remove(websocket)
