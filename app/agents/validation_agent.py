from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.models.job import CalibrationResult, MeasurementResult, MeshResult

ROLE = (
    "You are a holistic quality assurance agent for a photo-to-3D-print patch pipeline. "
    "The target printer is a Bambu Lab A1 (build volume: 256 x 256 x 256 mm). "
    "The pipeline produces a REPLACEMENT PATCH to fill a gap in a broken object — "
    "NOT a copy of the whole object. You review the entire pipeline output: "
    "calibration accuracy, patch dimensional plausibility, and mesh printability. "
    "Key constraints: minimum wall thickness 0.8mm, maximum 256mm in any axis, "
    "overhang angles above 45 degrees need supports. "
    "If calibration confidence is below 0.5, flag the measurements as unreliable. "
    "Provide a concise human-readable summary with print recommendations."
)


class ValidationAgent(Agent):
    def __init__(self):
        super().__init__("ValidationAgent", ROLE)

    async def run(
        self,
        calibration: CalibrationResult | None,
        measurement: MeasurementResult,
        mesh: MeshResult,
        thickness_mm: float,
    ) -> AgentResult:
        context = {
            "calibration_method": calibration.method if calibration else "prompt",
            "calibration_confidence": calibration.confidence if calibration else 1.0,
            "width_mm": measurement.width_mm,
            "height_mm": measurement.height_mm,
            "area_mm2": measurement.area_mm2,
            "thickness_mm": thickness_mm,
            "mesh_volume_mm3": mesh.volume_mm3,
            "mesh_is_watertight": mesh.is_watertight,
            "mesh_vertex_count": mesh.vertex_count,
            "mesh_face_count": mesh.face_count,
            "mesh_bounding_box": mesh.bounding_box_mm,
        }

        return await self.analyze(context)
