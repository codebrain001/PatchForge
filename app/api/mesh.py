from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.agents.orchestrator import get_job
from app.models.job import JobStatus

router = APIRouter()


@router.get("/jobs/{job_id}/mesh")
async def download_mesh(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status not in (
        JobStatus.READY, JobStatus.SENDING_TO_PRINTER,
        JobStatus.PRINTING, JobStatus.PRINT_COMPLETE,
    ) or job.mesh is None:
        raise HTTPException(409, "Mesh is not ready yet.")

    path = Path(job.mesh.file_path)
    if not path.exists():
        raise HTTPException(500, "Mesh file missing from disk.")

    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=f"patchforge_{job_id}.stl",
    )


@router.get("/jobs/{job_id}/mask")
async def download_mask(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    from app.core.storage import job_mask_path

    path = job_mask_path(job_id)
    if not path.exists():
        raise HTTPException(404, "Mask not generated yet.")

    return FileResponse(
        path=str(path),
        media_type="image/png",
        filename=f"patchforge_{job_id}_mask.png",
    )
