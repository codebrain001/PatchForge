from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.core.exceptions import SegmentationError

logger = logging.getLogger("patchforge.segmentation")

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model

    from transformers import Sam2Processor, Sam2Model

    model_name = settings.sam2_model_name
    _processor = Sam2Processor.from_pretrained(model_name)
    _model = Sam2Model.from_pretrained(model_name).to(settings.device)
    _model.eval()
    return _processor, _model


def segment(
    image_bgr: np.ndarray,
    click_x: int,
    click_y: int,
    additional_points: list[tuple[int, int]] | None = None,
    negative_points: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Run SAM 2 segmentation with point prompts (positive and negative).

    Returns:
        mask: binary uint8 mask (H, W) with 255 for foreground
        contours: list of OpenCV contour arrays, sorted largest-first
    """
    processor, model = _load_model()

    h, w = image_bgr.shape[:2]
    if not (0 <= click_x < w and 0 <= click_y < h):
        raise SegmentationError(
            f"Click point ({click_x}, {click_y}) is outside image bounds ({w}x{h})."
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    all_points = [[click_x, click_y]]
    all_labels = [1]

    for px, py in (additional_points or []):
        all_points.append([px, py])
        all_labels.append(1)

    for px, py in (negative_points or []):
        all_points.append([px, py])
        all_labels.append(0)

    input_points = [[[pt for pt in all_points]]]
    input_labels = [[all_labels]]

    inputs = processor(
        images=pil_image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(settings.device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_masks = outputs.pred_masks.cpu()
    original_sizes = inputs["original_sizes"].cpu()
    has_reshaped = "reshaped_input_sizes" in inputs
    if has_reshaped:
        reshaped_sizes = inputs["reshaped_input_sizes"].cpu()
        try:
            masks = processor.post_process_masks(
                pred_masks, original_sizes, reshaped_sizes,
            )
        except TypeError:
            masks = processor.post_process_masks(
                pred_masks, original_sizes,
            )
    else:
        masks = processor.post_process_masks(
            pred_masks, original_sizes,
        )

    if len(masks) == 0 or masks[0].numel() == 0:
        raise SegmentationError("SAM 2 produced no masks for the given points.")

    all_masks = masks[0].squeeze(0).numpy()
    if all_masks.ndim == 3:
        scores = outputs.iou_scores.cpu().squeeze().numpy()
        mask = _select_best_mask(all_masks, scores, h * w)
    else:
        mask = all_masks

    binary_mask = (mask > 0).astype(np.uint8) * 255
    binary_mask = _postprocess_mask(binary_mask)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise SegmentationError("No contours found in the segmentation mask.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return binary_mask, contours


def _select_best_mask(
    all_masks: np.ndarray,
    scores: np.ndarray,
    img_area: int,
) -> np.ndarray:
    """Pick the best SAM 2 mask for damage/gap segmentation.

    SAM 2 returns multiple masks (typically 3) ranked by predicted IoU.
    For gap/void detection we want the SMALLEST mask that still has a
    reasonable IoU, because the largest mask usually covers the entire
    object rather than just the missing piece.

    Strategy:
      1. Filter masks with IoU >= 0.4 (minimum quality)
      2. Among those, prefer masks covering 0.05%-15% of the image
      3. Pick the smallest mask in that range
      4. Fallback: smallest mask with IoU >= 0.4, then highest IoU
    """
    n_masks = all_masks.shape[0]
    if n_masks == 1:
        return all_masks[0]

    candidates = []
    for i in range(n_masks):
        area = int(np.sum(all_masks[i] > 0))
        ratio = area / img_area if img_area > 0 else 0
        candidates.append((i, scores[i], area, ratio))
        logger.debug(
            "SAM 2 mask %d: IoU=%.3f area=%d (%.2f%% of image)",
            i, scores[i], area, ratio * 100,
        )

    qualified = [(i, s, a, r) for i, s, a, r in candidates if s >= 0.4]
    if not qualified:
        best_idx = int(np.argmax(scores))
        logger.debug("No masks with IoU >= 0.4, falling back to highest IoU (mask %d)", best_idx)
        return all_masks[best_idx]

    in_range = [(i, s, a, r) for i, s, a, r in qualified if 0.0005 <= r <= 0.15]
    if in_range:
        in_range.sort(key=lambda x: x[2])
        chosen = in_range[0]
        logger.debug(
            "Selected smallest in-range mask %d: IoU=%.3f area=%d (%.2f%%)",
            chosen[0], chosen[1], chosen[2], chosen[3] * 100,
        )
        return all_masks[chosen[0]]

    qualified.sort(key=lambda x: x[2])
    chosen = qualified[0]
    logger.debug(
        "Selected smallest qualified mask %d: IoU=%.3f area=%d (%.2f%%)",
        chosen[0], chosen[1], chosen[2], chosen[3] * 100,
    )
    return all_masks[chosen[0]]


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, np.uint8(255), np.uint8(0))

    return mask
