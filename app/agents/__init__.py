"""
PatchForge Multi-Agent System
=============================

Seven specialized agents form the decision-making layer of the pipeline.
The LLM is the core thinking engine — every agent uses it to reason about
results, arbitrate between competing strategies, and decide whether the
pipeline should proceed.

Agent Registry
--------------

+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| Agent             | Pipeline Tools                | LLM Decision Authority                       | Input -> Output              |
+===================+===============================+==============================================+==============================+
| CalibrationAgent  | calibrate_all()               | Sees ALL calibration candidates (HEIF depth, | image, params ->             |
|                   |   -> HEIF depth extraction    | ArUco, WebXR, reference line). Picks the     | CalibrationResult,           |
|                   |   -> ArUco marker detection   | most trustworthy scale factor or blends      | AgentResult,                 |
|                   |   -> WebXR AR measurement     | multiple. Returns consensus decision.        | depth_map                    |
|                   |   -> Reference line calc      |                                              |                              |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| SegmentationAgent | segment()                     | Evaluates mask quality: coverage ratio,      | image, click_points ->       |
|                   |   -> SAM 2 point-prompt       | compactness, edge smoothness. Suggests       | mask, contours,              |
|                   |   -> OpenCV contour extract   | better click points or multi-point strategy. | AgentResult                  |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| MeasurementAgent  | measure()                     | Checks dimensional plausibility. If image    | contours, scale ->           |
|                   |   -> OpenCV contour analysis  | is provided, uses vision to cross-validate   | MeasurementResult,           |
|                   |   -> minAreaRect, boundingBox | pixel measurements against visible objects.  | AgentResult                  |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| ThicknessAgent    | estimate_thickness_all()      | Sees ALL thickness candidates (LiDAR depth,  | mask, scale, dims,           |
|                   |   -> LiDAR depth difference   | video MVS, side photo, monocular depth,      | depth_map, vision_t ->       |
|                   |   -> Video multi-view stereo  | vision estimate). Considers physical         | ThicknessResult,             |
|                   |   -> Side-photo LLM analysis  | plausibility against damage dimensions.      | AgentResult                  |
|                   |   -> Monocular depth model    | Returns consensus decision.                  |                              |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| MeshAgent         | generate_mesh()               | Evaluates mesh printability: watertightness, | contour, scale, thickness -> |
|                   |   -> Shapely polygon ops      | volume, face count, wall thickness. Suggests | MeshResult,                  |
|                   |   -> Trimesh extrusion        | chamfer or thickness adjustments.            | AgentResult                  |
|                   |   -> Chamfer edge processing  |                                              |                              |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| ValidationAgent   | (none — reads prior results)  | Holistic QA: cross-checks calibration,       | cal, meas, mesh, thickness ->|
|                   |                               | segmentation, measurement, and mesh results. | AgentResult                  |
|                   |                               | Issues final go/no-go for printing.          |                              |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+
| PrinterAgent      | (none — reads printer status) | Monitors Bambu Lab printer: temperatures,    | status, mesh, filament ->    |
|                   |                               | progress, error codes. Flags issues like     | AgentResult                  |
|                   |                               | overheating or stalled print jobs.            |                              |
+-------------------+-------------------------------+----------------------------------------------+------------------------------+

Pipeline Flow
-------------

Standard (single photo):
    CalibrationAgent -> SegmentationAgent -> MeasurementAgent -> ThicknessAgent
                                                                        |
    PrinterAgent <- ValidationAgent <- MeshAgent <-----------------------

Before/After (two photos):
    BeforeAfterCV -> SegmentationAgent -> CalibrationAgent -> MeasurementAgent -> ThicknessAgent
                                                                                       |
    PrinterAgent <- ValidationAgent <- MeshAgent <-------------------------------------

Consensus Decision Pattern (Calibration & Thickness):
    1. Run ALL available strategies simultaneously
    2. Collect every result with its confidence score
    3. Present ALL candidates to the LLM with domain context
    4. LLM picks the best result, blends multiple, or explains disagreement
    5. Return a single authoritative result with explained reasoning
"""

from app.agents.calibration_agent import CalibrationAgent
from app.agents.measurement_agent import MeasurementAgent
from app.agents.mesh_agent import MeshAgent
from app.agents.printer_agent import PrinterAgent
from app.agents.segmentation_agent import SegmentationAgent
from app.agents.thickness_agent import ThicknessAgent
from app.agents.validation_agent import ValidationAgent

__all__ = [
    "CalibrationAgent",
    "MeasurementAgent",
    "MeshAgent",
    "PrinterAgent",
    "SegmentationAgent",
    "ThicknessAgent",
    "ValidationAgent",
]
