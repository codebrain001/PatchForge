from __future__ import annotations

import json
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING_FRAMES = "extracting_frames"
    FRAMES_READY = "frames_ready"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    COMPARING_IMAGES = "comparing_images"
    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    PROPAGATING_MASKS = "propagating_masks"
    MASKS_PROPAGATED = "masks_propagated"
    MEASURING = "measuring"
    MEASURED = "measured"
    ESTIMATING_DEPTH = "estimating_depth"
    DEPTH_ESTIMATED = "depth_estimated"
    GENERATING_MESH = "generating_mesh"
    VALIDATING = "validating"
    READY = "ready"
    SLICING = "slicing"
    SENDING_TO_PRINTER = "sending_to_printer"
    PRINTING = "printing"
    PRINT_COMPLETE = "print_complete"
    FAILED = "failed"


class ThicknessMethod(str, Enum):
    VIDEO_MVS = "video_mvs"
    SIDE_PHOTO = "side_photo"
    LIDAR_DEPTH = "lidar_depth"
    VISION_ESTIMATE = "vision_estimate"
    MANUAL = "manual"


class UploadType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectionMode(str, Enum):
    CLICK = "click"
    BEFORE_AFTER = "before_after"
    PROMPT = "prompt"


class CalibrationResult(BaseModel):
    scale_factor: float = Field(description="mm per pixel")
    method: str = "aruco"
    marker_id: Optional[int] = None
    confidence: float = 1.0
    depth_map_available: bool = False


class MeasurementResult(BaseModel):
    width_mm: float
    height_mm: float
    area_mm2: float
    perimeter_mm: float
    bounding_rect_mm: list[float] = Field(description="[x, y, w, h] in mm")
    min_enclosing_radius_mm: float = 0.0
    confidence: float = 1.0


class ThicknessResult(BaseModel):
    thickness_mm: float
    method: ThicknessMethod
    confidence: float = 0.0
    depth_map_used: bool = False
    num_views_used: int = 0


class MeshResult(BaseModel):
    file_path: str
    vertex_count: int
    face_count: int
    volume_mm3: float
    surface_area_mm2: float
    is_watertight: bool
    bounding_box_mm: list[list[float]]


class PrinterConfig(BaseModel):
    ip_address: str
    access_code: str
    serial: str
    filament: str = "PLA"


class PrintStatus(BaseModel):
    connected: bool = False
    state: str = "unknown"
    progress_pct: Optional[int] = None
    remaining_seconds: Optional[int] = None
    bed_temp: Optional[float] = None
    nozzle_temp: Optional[float] = None
    error_code: int = 0
    file_name: str = ""


class ReasoningEntry(BaseModel):
    agent: str
    stage: str
    reasoning: str
    suggestions: list[str] = []
    confidence: float = 1.0


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.UPLOADED
    upload_type: UploadType = UploadType.IMAGE

    # Primary image (top-down / best frame)
    image_path: Optional[str] = None
    original_upload_path: Optional[str] = None

    # Video pipeline
    video_path: Optional[str] = None
    key_frame_paths: list[str] = Field(default_factory=list)
    best_frame_index: int = 0

    # Before/after comparison
    reference_image_path: Optional[str] = None
    detection_mode: DetectionMode = DetectionMode.CLICK
    before_after_stats: Optional[dict] = None

    # Side photo (fallback thickness estimation)
    side_image_path: Optional[str] = None

    click_points: list[dict] = Field(default_factory=list)

    # Video propagation — per-frame masks from SAM 2 video predictor
    propagated_mask_paths: list[str] = Field(default_factory=list)
    propagation_stats: Optional[dict] = None

    # Thickness — may be auto-inferred or manual
    thickness_mm: float = 3.0
    thickness_result: Optional[ThicknessResult] = None

    calibration: Optional[CalibrationResult] = None
    contours: Optional[list] = Field(default=None, exclude=True)
    contours_serialized: Optional[str] = Field(default=None, description="JSON-serialized contours for persistence")
    measurement: Optional[MeasurementResult] = None
    mesh: Optional[MeshResult] = None

    # Printer integration
    printer_config: Optional[PrinterConfig] = None
    print_status: Optional[PrintStatus] = None

    # Prompt-based patch generation
    prompt_text: Optional[str] = None
    parsed_shape: Optional[dict] = None

    reasoning_log: list[ReasoningEntry] = Field(default_factory=list)
    error: Optional[str] = None

    @model_validator(mode="after")
    def _restore_contours_from_serialized(self):
        if self.contours is None and self.contours_serialized is not None:
            try:
                raw = json.loads(self.contours_serialized)
                self.contours = [np.array(c, dtype=np.int32) for c in raw]
            except Exception:
                pass
        return self

    def serialize_contours(self) -> None:
        """Call before persisting to sync contours -> contours_serialized."""
        if self.contours is not None:
            try:
                self.contours_serialized = json.dumps(
                    [c.tolist() for c in self.contours]
                )
            except Exception:
                self.contours_serialized = None
        else:
            self.contours_serialized = None
