class PatchForgeError(Exception):
    """Base exception for PatchForge."""


class CalibrationError(PatchForgeError):
    """Raised when scale calibration fails (e.g. no ArUco marker detected)."""


class SegmentationError(PatchForgeError):
    """Raised when SAM 2 segmentation fails or produces an empty mask."""


class MeasurementError(PatchForgeError):
    """Raised when contour measurement cannot be completed."""


class MeshGenerationError(PatchForgeError):
    """Raised when 3D mesh generation or validation fails."""


class VideoProcessingError(PatchForgeError):
    """Raised when video decoding, frame extraction, or analysis fails."""


class ThicknessEstimationError(PatchForgeError):
    """Raised when all thickness inference strategies fail."""


class PrinterError(PatchForgeError):
    """Raised when communication with the Bambu Lab printer fails."""
