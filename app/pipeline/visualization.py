"""
Visualization utilities for real-time pipeline feedback.

Generates colored overlays and depth maps that are streamed to the
frontend via WebSocket so the user can watch SAM 2 and Depth Anything
in action during analysis.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("patchforge.visualization")


def create_sam2_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    contours: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Create a colored overlay showing the SAM 2 segmentation result.

    The damage region is highlighted in the brand primary colour with a
    white contour outline drawn on top.
    """
    overlay = image_bgr.copy()

    colored = np.zeros_like(image_bgr)
    colored[mask > 127] = [96, 75, 254]  # ~#FE4B60 in BGR
    overlay = cv2.addWeighted(overlay, 0.6, colored, 0.4, 0)

    if contours:
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    else:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)

    return overlay


def create_depth_visualization(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Run Depth Anything V2 on *image_bgr* and return a colourised depth map.

    Returns ``None`` when the depth model cannot be loaded (e.g. missing
    weights or unsupported device).
    """
    try:
        from app.pipeline.thickness_estimation import _load_depth_model
        from PIL import Image as PILImage

        pipe = _load_depth_model()
        if pipe is None:
            return None

        frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(frame_rgb)

        result = pipe(pil_img)
        depth_raw = result["depth"]
        if hasattr(depth_raw, "convert"):
            depth_map = np.array(depth_raw.convert("L"), dtype=np.float32)
        else:
            depth_map = np.array(depth_raw, dtype=np.float32)

        normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

        if colored.shape[:2] != image_bgr.shape[:2]:
            colored = cv2.resize(colored, (image_bgr.shape[1], image_bgr.shape[0]))

        if mask is not None:
            m = mask
            if m.shape[:2] != colored.shape[:2]:
                m = cv2.resize(m, (colored.shape[1], colored.shape[0]), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(colored, cnts, -1, (255, 255, 255), 2)

        return colored

    except Exception as e:
        logger.warning("Depth visualization failed: %s", e)
        return None
