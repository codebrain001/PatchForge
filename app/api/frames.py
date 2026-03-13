from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.agents.orchestrator import get_job
from app.models.job import UploadType
from app.models.schemas import FrameListResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Propagated mask endpoints
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/propagated-masks")
async def list_propagated_masks(job_id: str):
    """List all propagated mask URLs and per-frame stats for a video job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.upload_type != UploadType.VIDEO:
        raise HTTPException(400, "This job is not a video upload.")
    if not job.propagated_mask_paths:
        raise HTTPException(409, "Masks have not been propagated yet.")

    mask_urls = [
        f"/api/v1/jobs/{job_id}/propagated-masks/{i}"
        for i in range(len(job.propagated_mask_paths))
    ]

    return {
        "job_id": job_id,
        "mask_count": len(job.propagated_mask_paths),
        "mask_urls": mask_urls,
        "reference_frame_index": job.best_frame_index,
        "stats": job.propagation_stats,
    }


@router.get("/jobs/{job_id}/propagated-masks/{frame_index}")
async def get_propagated_mask(job_id: str, frame_index: int):
    """Download a specific propagated mask image."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if not job.propagated_mask_paths:
        raise HTTPException(409, "No propagated masks available.")
    if frame_index < 0 or frame_index >= len(job.propagated_mask_paths):
        raise HTTPException(
            404,
            f"Frame index {frame_index} out of range "
            f"(0-{len(job.propagated_mask_paths) - 1}).",
        )

    path = Path(job.propagated_mask_paths[frame_index])
    if not path.exists():
        raise HTTPException(500, "Propagated mask file missing from disk.")

    return FileResponse(
        path=str(path),
        media_type="image/png",
        filename=f"patchforge_{job_id}_propmask_{frame_index:04d}.png",
    )


@router.get("/jobs/{job_id}/frames", response_model=FrameListResponse)
async def list_frames(job_id: str):
    """List all extracted key frames for a video job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.upload_type != UploadType.VIDEO:
        raise HTTPException(400, "This job is not a video upload.")
    if not job.key_frame_paths:
        raise HTTPException(409, "Frames have not been extracted yet.")

    frame_urls = [
        f"/api/v1/jobs/{job_id}/frames/{i}"
        for i in range(len(job.key_frame_paths))
    ]

    return FrameListResponse(
        job_id=job_id,
        frame_count=len(job.key_frame_paths),
        frame_urls=frame_urls,
        best_frame_index=job.best_frame_index,
    )


@router.get("/jobs/{job_id}/frames/{frame_index}")
async def get_frame(job_id: str, frame_index: int):
    """Download a specific key frame image."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.upload_type != UploadType.VIDEO:
        raise HTTPException(400, "This job is not a video upload.")
    if frame_index < 0 or frame_index >= len(job.key_frame_paths):
        raise HTTPException(
            404,
            f"Frame index {frame_index} out of range "
            f"(0-{len(job.key_frame_paths) - 1}).",
        )

    path = Path(job.key_frame_paths[frame_index])
    if not path.exists():
        raise HTTPException(500, "Frame file missing from disk.")

    return FileResponse(
        path=str(path),
        media_type="image/png",
        filename=f"patchforge_{job_id}_frame_{frame_index:04d}.png",
    )


@router.put("/jobs/{job_id}/frames/best/{frame_index}")
async def set_best_frame(job_id: str, frame_index: int):
    """Override the auto-selected best frame for segmentation."""
    from app.agents.orchestrator import store_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.upload_type != UploadType.VIDEO:
        raise HTTPException(400, "This job is not a video upload.")
    if frame_index < 0 or frame_index >= len(job.key_frame_paths):
        raise HTTPException(
            404,
            f"Frame index {frame_index} out of range "
            f"(0-{len(job.key_frame_paths) - 1}).",
        )

    job.best_frame_index = frame_index

    import cv2
    best_path = Path(job.key_frame_paths[frame_index])
    if best_path.exists():
        from app.core.storage import job_upload_path

        img = cv2.imread(str(best_path))
        if img is not None:
            png_path = job_upload_path(job_id, ".png")
            cv2.imwrite(str(png_path), img)
            job.image_path = str(png_path)

    store_job(job)

    return {
        "job_id": job_id,
        "best_frame_index": frame_index,
        "message": f"Best frame updated to {frame_index}.",
    }
