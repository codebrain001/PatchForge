"""
Bambu Studio CLI slicer integration for PatchForge.

Slices an STL file into a print-ready .3mf using Bambu Studio's
command-line interface with the appropriate machine, filament,
and process profiles for the Bambu Lab A1.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.config import settings
from app.core.exceptions import PrinterError

logger = logging.getLogger("patchforge.slicer")

_BAMBU_STUDIO_PATHS = [
    r"C:\Program Files\Bambu Studio\bambu-studio.exe",
    r"C:\Program Files (x86)\Bambu Studio\bambu-studio.exe",
    r"/Applications/BambuStudio.app/Contents/MacOS/BambuStudio",
    "/usr/bin/bambu-studio",
]

_PROFILES_BASE = r"C:\Program Files\Bambu Studio\resources\profiles\BBL"


def _find_bambu_studio() -> Optional[str]:
    """Locate the Bambu Studio executable."""
    custom = settings.bambu_studio_path
    if custom and Path(custom).is_file():
        return custom

    for p in _BAMBU_STUDIO_PATHS:
        if Path(p).is_file():
            return p

    found = shutil.which("bambu-studio")
    return found


def _resolve_profile(directory: str, pattern: str) -> Optional[str]:
    """Find a profile JSON by partial name match."""
    profile_dir = Path(_PROFILES_BASE) / directory
    if not profile_dir.is_dir():
        return None
    for f in profile_dir.glob("*.json"):
        if pattern.lower() in f.name.lower() and "template" not in f.name.lower():
            return str(f)
    return None


def _build_settings_path(filament: str, nozzle: str = "0.4") -> list[str]:
    """Build the --load-settings argument from machine + process + filament profiles."""
    machine = _resolve_profile("machine", f"A1 {nozzle} nozzle")
    process = _resolve_profile("process", f"0.20mm Standard @BBL A1")
    filament_profile = _resolve_profile("filament", f"Bambu {filament} @BBL A1")

    if not filament_profile:
        filament_profile = _resolve_profile("filament", f"Bambu PLA Basic @BBL A1")

    parts = [p for p in [machine, process, filament_profile] if p]
    return parts


def is_slicer_available() -> bool:
    """Check whether Bambu Studio CLI is available on this system."""
    return _find_bambu_studio() is not None


def slice_stl(
    stl_path: str,
    output_path: Optional[str] = None,
    filament: str = "PLA",
    nozzle: str = "0.4",
) -> str:
    """
    Slice an STL file into a print-ready .3mf using Bambu Studio CLI.

    Args:
        stl_path: Path to the input STL file.
        output_path: Where to save the sliced .3mf. If None, uses
                     the same directory and name as the STL.
        filament: Filament type (PLA, ABS, PETG, etc.).
        nozzle: Nozzle diameter string (0.2, 0.4, 0.6, 0.8).

    Returns:
        Path to the sliced .3mf file.

    Raises:
        PrinterError: If slicing fails or Bambu Studio is not found.
    """
    exe = _find_bambu_studio()
    if exe is None:
        raise PrinterError(
            "Bambu Studio not found. Install it from https://bambulab.com/en/download/studio "
            "or set BAMBU_STUDIO_PATH in your .env file."
        )

    stl = Path(stl_path)
    if not stl.exists():
        raise PrinterError(f"STL file not found: {stl_path}")

    if output_path is None:
        output_3mf = stl.with_suffix(".3mf")
    else:
        output_3mf = Path(output_path)

    output_dir = output_3mf.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    settings_files = _build_settings_path(filament, nozzle)

    cmd = [exe, "--slice", "0"]

    if settings_files:
        cmd.extend(["--load-settings", ";".join(settings_files)])

    cmd.extend([
        "--export-3mf", str(output_3mf),
        "--outputdir", str(output_dir),
        str(stl),
    ])

    logger.info("Slicing STL: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(output_dir),
        )
    except subprocess.TimeoutExpired:
        raise PrinterError("Slicing timed out after 120 seconds.")
    except Exception as e:
        raise PrinterError(f"Failed to run Bambu Studio slicer: {e}")

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        stdout = result.stdout.strip() if result.stdout else ""
        detail = stderr or stdout or "Unknown error"
        logger.error("Slicer failed (code %d): %s", result.returncode, detail)
        raise PrinterError(f"Slicing failed: {detail}")

    if not output_3mf.exists():
        possible = list(output_dir.glob("*.3mf"))
        if possible:
            output_3mf = possible[0]
        else:
            raise PrinterError(
                "Slicer ran but no .3mf output was found. "
                "Check Bambu Studio installation."
            )

    logger.info("Sliced successfully: %s (%.1f KB)", output_3mf, output_3mf.stat().st_size / 1024)
    return str(output_3mf)
