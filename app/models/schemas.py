from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Point prompts for segmentation
# ---------------------------------------------------------------------------

class PointPrompt(BaseModel):
    x: int
    y: int
    label: int = Field(default=1, description="1=foreground, 0=background")


# ---------------------------------------------------------------------------
# Analysis / segmentation request
# ---------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    points: list[PointPrompt] = Field(
        min_length=1,
        description="Click points on the image (positive and negative prompts)"
    )
    marker_size_mm: float = Field(default=40.0, description="Known ArUco marker side length in mm")
    ref_line_start: Optional[list[int]] = Field(default=None, description="Reference line start [x,y] in pixels")
    ref_line_end: Optional[list[int]] = Field(default=None, description="Reference line end [x,y] in pixels")
    ref_line_mm: Optional[float] = Field(default=None, description="Reference line real-world length in mm")
    webxr_scale_mm_per_px: Optional[float] = Field(default=None, description="Pre-computed scale from WebXR AR measurement")
    frame_index: Optional[int] = Field(default=None, description="Key-frame index to segment (video jobs)")


# ---------------------------------------------------------------------------
# Mesh generation request
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    thickness_mm: Optional[float] = Field(
        default=None, gt=0, le=50,
        description="Extrusion thickness in mm. If omitted, uses auto-inferred thickness.",
    )
    chamfer_mm: float = Field(default=0.0, ge=0, le=5.0, description="Edge chamfer radius in mm")


# ---------------------------------------------------------------------------
# Side-photo upload for thickness fallback
# ---------------------------------------------------------------------------

class SidePhotoThicknessRequest(BaseModel):
    manual_hint_mm: Optional[float] = Field(
        default=None, gt=0, le=100,
        description="Optional user hint for expected thickness (helps AI estimate)",
    )


# ---------------------------------------------------------------------------
# Prompt-based patch generation
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str = Field(
        min_length=3, max_length=1000,
        description="Natural language description of the desired patch",
    )


# ---------------------------------------------------------------------------
# Printer connection
# ---------------------------------------------------------------------------

class PrinterConnectRequest(BaseModel):
    ip_address: str = Field(description="Bambu printer LAN IP address")
    access_code: str = Field(description="8-digit access code from printer screen")
    serial: str = Field(description="Printer serial number")
    filament: str = Field(default="PLA", description="Loaded filament type")


class PrintRequest(BaseModel):
    use_ams: bool = Field(default=False, description="Whether to use AMS system")
    flow_calibration: bool = Field(default=True, description="Run flow calibration before printing")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    id: str
    status: str
    upload_type: str = "image"
    detection_mode: str = "click"
    has_reference_image: bool = False
    error: str | None = None
    calibration: dict | None = None
    measurement: dict | None = None
    thickness: dict | None = None
    mesh: dict | None = None
    print_status: dict | None = None
    key_frame_count: int = 0
    propagated_mask_count: int = 0
    propagation_stats: dict | None = None
    before_after_stats: dict | None = None
    parsed_shape: dict | None = None
    reasoning_log: list[dict] = []


class PrinterStatusResponse(BaseModel):
    connected: bool = False
    state: str = "unknown"
    progress_pct: int | None = None
    remaining_seconds: int | None = None
    bed_temp: float | None = None
    nozzle_temp: float | None = None
    error_code: int = 0
    file_name: str = ""


class FrameListResponse(BaseModel):
    job_id: str
    frame_count: int
    frame_urls: list[str]
    best_frame_index: int
