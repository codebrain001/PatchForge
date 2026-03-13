import re
import uuid
from pathlib import Path

from app.config import settings

_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def ensure_dirs() -> None:
    for d in (
        settings.upload_dir,
        settings.masks_dir,
        settings.meshes_dir,
        settings.videos_dir,
        settings.frames_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)


def generate_job_id() -> str:
    return uuid.uuid4().hex[:12]


def validate_job_id(job_id: str) -> str:
    """Validate that a job_id is a safe hex string. Raises ValueError if not."""
    if not _JOB_ID_RE.match(job_id):
        raise ValueError(f"Invalid job ID format: {job_id!r}")
    return job_id


# --- Image paths ---

def job_upload_path(job_id: str, suffix: str = ".png") -> Path:
    return settings.upload_dir / f"{job_id}{suffix}"


def job_mask_path(job_id: str) -> Path:
    return settings.masks_dir / f"{job_id}_mask.png"


def job_mesh_path(job_id: str) -> Path:
    return settings.meshes_dir / f"{job_id}.stl"


# --- Video paths ---

def job_video_path(job_id: str, suffix: str = ".mp4") -> Path:
    return settings.videos_dir / f"{job_id}{suffix}"


def job_frames_dir(job_id: str) -> Path:
    d = settings.frames_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_frame_path(job_id: str, index: int) -> Path:
    return job_frames_dir(job_id) / f"frame_{index:04d}.png"


# --- Propagated masks directory ---

def job_propagated_masks_dir(job_id: str) -> Path:
    d = settings.frames_dir / job_id / "propagated"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Reference ("before") image path ---

def job_reference_image_path(job_id: str) -> Path:
    return settings.upload_dir / f"{job_id}_reference.png"


# --- Side-image path ---

def job_side_image_path(job_id: str) -> Path:
    return settings.upload_dir / f"{job_id}_side.png"
