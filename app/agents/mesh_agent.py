from __future__ import annotations

from pathlib import Path

import numpy as np

from app.agents.base import Agent, AgentResult
from app.models.job import MeshResult

ROLE = (
    "You are a 3D printing and mesh generation expert. "
    "You evaluate generated STL meshes for printability: watertightness, volume, "
    "face count, wall thickness adequacy, and bounding box dimensions. "
    "Suggest thickness or chamfer adjustments based on the part dimensions. "
    "Flag meshes that are too thin to print or too large for typical printers."
)


class MeshAgent(Agent):
    def __init__(self):
        super().__init__("MeshAgent", ROLE)

    async def run(
        self,
        contour: np.ndarray,
        scale_factor: float,
        thickness_mm: float,
        output_path: Path,
        chamfer_mm: float = 0.0,
    ) -> tuple[MeshResult, AgentResult]:
        from app.pipeline.mesh_generation import generate_mesh

        result = generate_mesh(
            contour, scale_factor, thickness_mm, output_path,
            chamfer_mm=chamfer_mm,
        )

        context = {
            "vertex_count": result.vertex_count,
            "face_count": result.face_count,
            "volume_mm3": result.volume_mm3,
            "surface_area_mm2": result.surface_area_mm2,
            "is_watertight": result.is_watertight,
            "bounding_box_mm": result.bounding_box_mm,
            "thickness_mm": thickness_mm,
            "chamfer_mm": chamfer_mm,
        }

        analysis = await self.analyze(context)
        return result, analysis
