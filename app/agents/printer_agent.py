from __future__ import annotations

from typing import Optional

from app.agents.base import Agent, AgentResult
from app.models.job import PrintStatus, MeshResult

ROLE = (
    "You are a 3D print monitoring agent for a Bambu Lab A1 printer "
    "(build volume: 256 x 256 x 256 mm). "
    "You evaluate real-time printer status during a print job: temperature "
    "stability, progress rate, error codes, and overall health. "
    "Flag issues: bed temperature above 70C or nozzle above 250C for PLA, "
    "error codes other than 0, stalled progress (no change over 5+ minutes), "
    "or signs of print failure (layer shifting, stringing, poor adhesion). "
    "Provide actionable suggestions: pause, adjust temperature, re-level bed, "
    "or check first-layer adhesion."
)


class PrinterAgent(Agent):
    def __init__(self):
        super().__init__("PrinterAgent", ROLE)

    async def run(
        self,
        status: PrintStatus,
        mesh: Optional[MeshResult] = None,
        filament: str = "PLA",
    ) -> AgentResult:
        context = {
            "connected": status.connected,
            "state": status.state,
            "progress_pct": status.progress_pct,
            "remaining_seconds": status.remaining_seconds,
            "bed_temp": status.bed_temp,
            "nozzle_temp": status.nozzle_temp,
            "error_code": status.error_code,
            "file_name": status.file_name,
            "filament_type": filament,
        }

        if mesh:
            context["mesh_volume_mm3"] = mesh.volume_mm3
            context["mesh_is_watertight"] = mesh.is_watertight

        return await self.analyze(context)
