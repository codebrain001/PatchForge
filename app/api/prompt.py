from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.storage import generate_job_id, ensure_dirs
from app.models.job import (
    Job, JobStatus, UploadType, DetectionMode,
    MeasurementResult, ThicknessResult, ThicknessMethod,
)
from app.models.schemas import PromptRequest, JobResponse
from app.agents.orchestrator import store_job
from app.api.helpers import job_to_response as _job_to_response

logger = logging.getLogger("patchforge.prompt")

router = APIRouter()


@router.post("/jobs/prompt", response_model=JobResponse, status_code=201)
async def create_job_from_prompt(req: PromptRequest):
    """Parse a natural language prompt and create a job with pre-populated dimensions."""
    ensure_dirs()

    from app.pipeline.prompt_to_mesh import parse_prompt

    try:
        parsed_shape = await parse_prompt(req.prompt)
    except Exception as e:
        logger.error("Prompt parsing failed: %s", e)
        raise HTTPException(500, f"Failed to parse prompt: {e}")

    job_id = generate_job_id()
    w = parsed_shape["width_mm"]
    h = parsed_shape["height_mm"]
    t = parsed_shape["thickness_mm"]

    job = Job(
        id=job_id,
        status=JobStatus.MEASURED,
        upload_type=UploadType.IMAGE,
        detection_mode=DetectionMode.PROMPT,
        prompt_text=req.prompt,
        parsed_shape=parsed_shape,
        thickness_mm=t,
        measurement=MeasurementResult(
            width_mm=w,
            height_mm=h,
            area_mm2=round(w * h, 2),
            perimeter_mm=round(2 * (w + h), 2),
            bounding_rect_mm=[0, 0, w, h],
            confidence=0.9,
        ),
        thickness_result=ThicknessResult(
            thickness_mm=t,
            method=ThicknessMethod.MANUAL,
            confidence=1.0,
        ),
    )
    store_job(job)
    logger.info("Prompt job %s created: %s", job_id, parsed_shape.get("description", ""))

    return _job_to_response(job)
