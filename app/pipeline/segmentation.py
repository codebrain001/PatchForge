from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.core.exceptions import SegmentationError

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
        best_idx = int(np.argmax(scores))
        mask = all_masks[best_idx]
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
