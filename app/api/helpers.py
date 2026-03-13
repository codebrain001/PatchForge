from __future__ import annotations

from app.models.job import Job
from app.models.schemas import JobResponse


def job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status.value,
        upload_type=job.upload_type.value,
        detection_mode=job.detection_mode.value if hasattr(job.detection_mode, 'value') else job.detection_mode,
        has_reference_image=job.reference_image_path is not None,
        error=job.error,
        calibration=job.calibration.model_dump() if job.calibration else None,
        measurement=job.measurement.model_dump() if job.measurement else None,
        thickness=job.thickness_result.model_dump() if job.thickness_result else None,
        mesh=job.mesh.model_dump() if job.mesh else None,
        print_status=job.print_status.model_dump() if job.print_status else None,
        key_frame_count=len(job.key_frame_paths),
        propagated_mask_count=len(job.propagated_mask_paths),
        propagation_stats=job.propagation_stats,
        before_after_stats=job.before_after_stats,
        parsed_shape=job.parsed_shape,
        reasoning_log=[e.model_dump() for e in job.reasoning_log],
    )
