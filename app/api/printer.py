from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.agents.orchestrator import get_job, store_job
from app.config import settings
from app.models.job import JobStatus, PrinterConfig, PrintStatus
from app.models.schemas import (
    PrinterConnectRequest, PrintRequest, PrinterStatusResponse,
)
from app.pipeline import printer as printer_svc
from app.pipeline import slicer as slicer_svc

logger = logging.getLogger("patchforge.api.printer")

router = APIRouter(prefix="/printer")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@router.post("/connect", response_model=PrinterStatusResponse)
async def connect_printer(req: PrinterConnectRequest):
    """Connect to a Bambu Lab printer on the local network."""
    config = PrinterConfig(
        ip_address=req.ip_address,
        access_code=req.access_code,
        serial=req.serial,
        filament=req.filament,
    )
    try:
        status = await asyncio.to_thread(printer_svc.connect, config)
    except Exception as e:
        raise HTTPException(502, f"Printer connection failed: {e}")

    return _status_to_response(status)


@router.post("/disconnect")
async def disconnect_printer():
    """Disconnect from the printer."""
    await asyncio.to_thread(printer_svc.disconnect)
    return {"message": "Disconnected from printer."}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=PrinterStatusResponse)
async def printer_status():
    """Get the current printer status."""
    status = await asyncio.to_thread(printer_svc.get_status)
    return _status_to_response(status)


# ---------------------------------------------------------------------------
# Print a job
# ---------------------------------------------------------------------------

@router.get("/slicer-status")
async def slicer_status():
    """Check whether Bambu Studio CLI is available for slicing."""
    available = await asyncio.to_thread(slicer_svc.is_slicer_available)
    return {
        "available": available,
        "auto_slice": settings.auto_slice,
        "filament": settings.bambu_filament,
        "nozzle": settings.bambu_nozzle,
    }


@router.post("/jobs/{job_id}/print")
async def print_job(job_id: str, req: PrintRequest | None = None):
    """Slice the STL (if needed), upload to printer, and start printing."""
    if req is None:
        req = PrintRequest()

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.mesh is None or job.status not in (
        JobStatus.READY, JobStatus.PRINT_COMPLETE,
    ):
        raise HTTPException(409, "Mesh must be generated before printing.")

    if not printer_svc.is_connected():
        raise HTTPException(502, "Printer is not connected. Call /printer/connect first.")

    stl_path = job.mesh.file_path

    # --- Auto-slice STL -> .3mf if Bambu Studio is available ---
    file_to_upload = stl_path
    remote_name = f"patchforge_{job_id}.stl"
    sliced = False

    if settings.auto_slice and slicer_svc.is_slicer_available():
        job.status = JobStatus.SLICING
        store_job(job)

        try:
            sliced_path = await asyncio.to_thread(
                slicer_svc.slice_stl,
                stl_path,
                filament=settings.bambu_filament,
                nozzle=settings.bambu_nozzle,
            )
            file_to_upload = sliced_path
            remote_name = f"patchforge_{job_id}.3mf"
            sliced = True
            logger.info("Auto-sliced %s -> %s", stl_path, sliced_path)
        except Exception as e:
            logger.warning("Auto-slice failed, uploading raw STL: %s", e)
            file_to_upload = stl_path
            remote_name = f"patchforge_{job_id}.stl"

    # --- Upload to printer ---
    job.status = JobStatus.SENDING_TO_PRINTER
    store_job(job)

    try:
        await asyncio.to_thread(printer_svc.upload_file, file_to_upload, remote_name)
    except Exception as e:
        job.status = JobStatus.READY
        store_job(job)
        raise HTTPException(502, f"File upload to printer failed: {e}")

    # --- Start print ---
    try:
        success = await asyncio.to_thread(
            printer_svc.start_print,
            remote_name,
            use_ams=req.use_ams,
            flow_calibration=req.flow_calibration,
        )
    except Exception as e:
        job.status = JobStatus.READY
        store_job(job)
        raise HTTPException(502, f"Failed to start print: {e}")

    if not success:
        job.status = JobStatus.READY
        store_job(job)
        raise HTTPException(502, "Printer refused the print command.")

    job.status = JobStatus.PRINTING
    job.print_status = await asyncio.to_thread(printer_svc.get_status)
    store_job(job)

    return {
        "job_id": job_id,
        "status": "printing",
        "sliced": sliced,
        "message": f"Print started: {remote_name}",
    }


@router.get("/jobs/{job_id}/print-status")
async def print_status(job_id: str):
    """Poll the current print progress for a job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    if not printer_svc.is_connected():
        return _status_to_response(PrintStatus(connected=False, state="disconnected"))

    status = await asyncio.to_thread(printer_svc.get_status)
    job.print_status = status

    # Auto-transition to PRINT_COMPLETE when printer goes idle after printing
    if (
        job.status == JobStatus.PRINTING
        and status.progress_pct is not None
        and status.progress_pct >= 100
    ):
        job.status = JobStatus.PRINT_COMPLETE

    store_job(job)
    return _status_to_response(status)


# ---------------------------------------------------------------------------
# Print controls
# ---------------------------------------------------------------------------

@router.post("/stop")
async def stop():
    """Stop the current print."""
    try:
        await asyncio.to_thread(printer_svc.stop_print)
        return {"message": "Print stopped."}
    except Exception as e:
        raise HTTPException(502, f"Failed to stop: {e}")


@router.post("/pause")
async def pause():
    """Pause the current print."""
    try:
        await asyncio.to_thread(printer_svc.pause_print)
        return {"message": "Print paused."}
    except Exception as e:
        raise HTTPException(502, f"Failed to pause: {e}")


@router.post("/resume")
async def resume():
    """Resume the paused print."""
    try:
        await asyncio.to_thread(printer_svc.resume_print)
        return {"message": "Print resumed."}
    except Exception as e:
        raise HTTPException(502, f"Failed to resume: {e}")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@router.get("/camera")
async def camera_frame():
    """Get the latest camera frame as base64 JPEG."""
    frame = await asyncio.to_thread(printer_svc.get_camera_frame_b64)
    if frame is None:
        raise HTTPException(503, "Camera feed unavailable.")
    return {"frame_b64": frame}


# ---------------------------------------------------------------------------
# Light control
# ---------------------------------------------------------------------------

@router.post("/light/{state}")
async def light_control(state: str):
    """Turn the chamber light on or off."""
    if state not in ("on", "off"):
        raise HTTPException(400, "State must be 'on' or 'off'.")
    success = await asyncio.to_thread(printer_svc.set_light, state == "on")
    return {"light": state, "success": success}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_to_response(status: PrintStatus) -> PrinterStatusResponse:
    return PrinterStatusResponse(
        connected=status.connected,
        state=status.state,
        progress_pct=status.progress_pct,
        remaining_seconds=status.remaining_seconds,
        bed_temp=status.bed_temp,
        nozzle_temp=status.nozzle_temp,
        error_code=status.error_code,
        file_name=status.file_name,
    )
