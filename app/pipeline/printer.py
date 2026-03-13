"""
Bambu Lab printer integration for PatchForge.

Manages a singleton printer connection and provides functions
to connect, upload STL/3MF files, start prints, and poll status.
Uses the bambulabs_api package for MQTT/FTP communication.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from app.config import settings
from app.core.exceptions import PrinterError
from app.models.job import PrinterConfig, PrintStatus

logger = logging.getLogger("patchforge.printer")

# Singleton printer connection
_printer = None
_printer_config: Optional[PrinterConfig] = None


def _get_printer():
    global _printer
    if _printer is None:
        raise PrinterError("Printer is not connected. Call connect() first.")
    return _printer


def connect(config: PrinterConfig) -> PrintStatus:
    """Establish connection to the Bambu Lab printer via MQTT."""
    global _printer, _printer_config

    try:
        import bambulabs_api as bl
    except ImportError:
        raise PrinterError(
            "bambulabs_api is not installed. Run: pip install bambulabs_api"
        )

    if _printer is not None:
        try:
            _printer.disconnect()
        except Exception:
            pass
        _printer = None

    try:
        _printer = bl.Printer(
            config.ip_address,
            config.access_code,
            config.serial,
        )
        _printer.connect()

        # Allow MQTT handshake time
        time.sleep(3)

        if not _printer.mqtt_client_connected():
            _printer = None
            raise PrinterError(
                f"Could not connect to printer at {config.ip_address}. "
                "Verify IP, access code, and serial number."
            )

        _printer_config = config

        logger.info(
            "Connected to Bambu printer at %s (serial: %s)",
            config.ip_address, config.serial,
        )

        return get_status()

    except PrinterError:
        raise
    except Exception as e:
        _printer = None
        raise PrinterError(f"Connection failed: {e}") from e


def disconnect() -> None:
    """Disconnect from the printer."""
    global _printer, _printer_config
    if _printer is not None:
        try:
            _printer.disconnect()
        except Exception:
            pass
        _printer = None
        _printer_config = None
        logger.info("Disconnected from Bambu printer.")


def is_connected() -> bool:
    """Check if a printer connection is active."""
    if _printer is None:
        return False
    try:
        return _printer.mqtt_client_connected()
    except Exception:
        return False


def get_status() -> PrintStatus:
    """Poll the current printer status."""
    if not is_connected():
        return PrintStatus(connected=False, state="disconnected")

    printer = _get_printer()

    try:
        state = printer.get_state()
        state_str = state.value if hasattr(state, "value") else str(state)
    except Exception:
        state_str = "unknown"

    try:
        pct = printer.get_percentage()
        progress = int(pct) if isinstance(pct, (int, float)) else None
    except Exception:
        progress = None

    try:
        remaining = printer.get_time()
        remaining_sec = int(remaining) if isinstance(remaining, (int, float)) else None
    except Exception:
        remaining_sec = None

    try:
        bed = printer.get_bed_temperature()
        bed_temp = float(bed) if bed is not None else None
    except Exception:
        bed_temp = None

    try:
        nozzle = printer.get_nozzle_temperature()
        nozzle_temp = float(nozzle) if nozzle is not None else None
    except Exception:
        nozzle_temp = None

    try:
        err = printer.print_error_code()
        error_code = int(err) if err is not None else 0
    except Exception:
        error_code = 0

    try:
        fname = printer.get_file_name()
        file_name = str(fname) if fname else ""
    except Exception:
        file_name = ""

    return PrintStatus(
        connected=True,
        state=state_str,
        progress_pct=progress,
        remaining_seconds=remaining_sec,
        bed_temp=bed_temp,
        nozzle_temp=nozzle_temp,
        error_code=error_code,
        file_name=file_name,
    )


def upload_file(file_path: str, remote_filename: Optional[str] = None) -> str:
    """Upload a file to the printer's SD card via FTP."""
    printer = _get_printer()

    path = Path(file_path)
    if not path.exists():
        raise PrinterError(f"File not found: {file_path}")

    if remote_filename is None:
        remote_filename = path.name
    else:
        remote_filename = Path(remote_filename).name

    try:
        with open(str(path), "rb") as f:
            result = printer.upload_file(f, remote_filename)
        logger.info("Uploaded %s to printer as %s", file_path, remote_filename)
        return result
    except Exception as e:
        raise PrinterError(f"File upload failed: {e}") from e


def start_print(
    filename: str,
    use_ams: bool = False,
    flow_calibration: bool = True,
) -> bool:
    """Start printing a file already on the printer."""
    printer = _get_printer()

    try:
        success = printer.start_print(
            filename,
            plate_number=1,
            use_ams=use_ams,
            ams_mapping=[0],
            flow_calibration=flow_calibration,
        )
        if success:
            logger.info("Started printing: %s", filename)
        else:
            logger.warning("start_print returned False for %s", filename)
        return success
    except Exception as e:
        raise PrinterError(f"Failed to start print: {e}") from e


def stop_print() -> bool:
    """Stop the current print."""
    printer = _get_printer()
    try:
        return printer.stop_print()
    except Exception as e:
        raise PrinterError(f"Failed to stop print: {e}") from e


def pause_print() -> bool:
    """Pause the current print."""
    printer = _get_printer()
    try:
        return printer.pause_print()
    except Exception as e:
        raise PrinterError(f"Failed to pause print: {e}") from e


def resume_print() -> bool:
    """Resume a paused print."""
    printer = _get_printer()
    try:
        return printer.resume_print()
    except Exception as e:
        raise PrinterError(f"Failed to resume print: {e}") from e


def get_camera_frame_b64() -> Optional[str]:
    """Get the latest camera frame as a base64-encoded JPEG string."""
    if not is_connected():
        return None

    printer = _get_printer()
    try:
        frame = printer.get_camera_frame()
        return frame if frame else None
    except Exception:
        return None


def get_light_state() -> str:
    """Get the printer chamber light state."""
    printer = _get_printer()
    try:
        return printer.get_light_state()
    except Exception:
        return "unknown"


def set_light(on: bool) -> bool:
    """Turn the chamber light on or off."""
    printer = _get_printer()
    try:
        return printer.turn_light_on() if on else printer.turn_light_off()
    except Exception:
        return False
