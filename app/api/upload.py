from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np

from app.core.storage import (
    generate_job_id, job_upload_path, job_video_path, ensure_dirs,
)
from app.models.job import Job, JobStatus, UploadType
from app.models.schemas import JobResponse
from app.agents.orchestrator import store_job
from app.api.helpers import job_to_response as _job_to_response

logger = logging.getLogger("patchforge.upload")

router = APIRouter()

ALLOWED_IMAGE_TYPES = (
    "image/jpeg", "image/png", "image/webp",
    "image/heic", "image/heif",
)

ALLOWED_VIDEO_TYPES = (
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-msvideo", "video/x-matroska",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

MAX_IMAGE_BYTES = 20 * 1024 * 1024     # 20 MB
MAX_VIDEO_BYTES = 200 * 1024 * 1024    # 200 MB


def _classify_upload(filename: str, content_type: str) -> str:
    """Return 'image', 'video', or raise on unknown type."""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if content_type in ALLOWED_IMAGE_TYPES:
        return "image"
    if content_type in ALLOWED_VIDEO_TYPES:
        return "video"
    return "unknown"


# ---------------------------------------------------------------------------
# Unified upload endpoint
# ---------------------------------------------------------------------------

@router.post("/jobs", response_model=JobResponse, status_code=201)
async def upload_file(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    filename = file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()

    kind = _classify_upload(filename, content_type)
    if kind == "unknown":
        raise HTTPException(
            400,
            "Unsupported file type. Upload a photo (JPEG, PNG, WebP, HEIC) "
            "or a short video (MP4, MOV, WebM).",
        )

    ensure_dirs()
    contents = await file.read()

    if kind == "image":
        return await _handle_image_upload(contents, ext, content_type)
    else:
        return await _handle_video_upload(contents, ext)


# ---------------------------------------------------------------------------
# Image upload (unchanged logic, cleaner structure)
# ---------------------------------------------------------------------------

async def _handle_image_upload(
    contents: bytes, ext: str, content_type: str,
) -> JobResponse:
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "Image exceeds 20 MB limit.")

    job_id = generate_job_id()
    is_heif = ext in (".heic", ".heif") or "heif" in content_type or "heic" in content_type

    original_suffix = ext if ext else ".png"
    original_path = job_upload_path(job_id, original_suffix)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(original_path), "wb") as f:
        f.write(contents)

    if is_heif:
        try:
            from pillow_heif import register_heif_opener
            from PIL import Image
            import io

            register_heif_opener()
            pil_img = Image.open(io.BytesIO(contents))
            pil_img = pil_img.convert("RGB")
            png_arr = np.array(pil_img)
            image = cv2.cvtColor(png_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(400, f"Could not decode HEIF image: {e}")
    else:
        arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image.")

    png_path = job_upload_path(job_id, ".png")
    cv2.imwrite(str(png_path), image)

    job = Job(
        id=job_id,
        status=JobStatus.UPLOADED,
        upload_type=UploadType.IMAGE,
        image_path=str(png_path),
        original_upload_path=str(original_path),
    )
    store_job(job)
    logger.info("Image job %s created (%s)", job_id, ext)

    return _job_to_response(job)


# ---------------------------------------------------------------------------
# Video upload → save + extract key frames
# ---------------------------------------------------------------------------

async def _handle_video_upload(contents: bytes, ext: str) -> JobResponse:
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(400, "Video exceeds 200 MB limit.")

    job_id = generate_job_id()

    video_suffix = ext if ext else ".mp4"
    video_path = job_video_path(job_id, video_suffix)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(video_path), "wb") as f:
        f.write(contents)

    job = Job(
        id=job_id,
        status=JobStatus.EXTRACTING_FRAMES,
        upload_type=UploadType.VIDEO,
        video_path=str(video_path),
    )
    store_job(job)

    try:
        from app.pipeline.video_processing import extract_keyframes, get_best_frame_bgr

        result = await asyncio.to_thread(
            extract_keyframes, str(video_path), job_id,
        )

        job.key_frame_paths = [fi.path for fi in result.frame_infos]
        job.best_frame_index = result.best_frame_index

        best_bgr = get_best_frame_bgr(job_id, result)
        png_path = job_upload_path(job_id, ".png")
        cv2.imwrite(str(png_path), best_bgr)
        job.image_path = str(png_path)

        job.status = JobStatus.FRAMES_READY
        store_job(job)

        logger.info(
            "Video job %s: %d key frames extracted, best=%d",
            job_id, len(result.frame_infos), result.best_frame_index,
        )

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"Video processing failed: {e}"
        store_job(job)
        raise HTTPException(500, f"Video processing failed: {e}")

    return _job_to_response(job)


