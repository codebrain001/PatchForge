"""
SAM 2 Video Propagation for PatchForge.

Takes the initial segmentation from the best frame and propagates
the damage mask across all extracted key frames using SAM 2's
streaming video predictor. This provides multi-view masks that
improve thickness estimation and give the user visual confirmation
across angles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.core.exceptions import SegmentationError

logger = logging.getLogger("patchforge.video_seg")

_video_processor = None
_video_model = None


def _load_video_model():
    global _video_processor, _video_model
    if _video_model is not None:
        return _video_processor, _video_model

    from transformers import Sam2VideoProcessor, Sam2VideoModel

    model_name = settings.sam2_model_name
    _video_processor = Sam2VideoProcessor.from_pretrained(model_name)
    _video_model = Sam2VideoModel.from_pretrained(model_name).to(
        settings.device, dtype=torch.bfloat16
    )
    _video_model.eval()
    return _video_processor, _video_model


@dataclass
class PropagatedMask:
    frame_index: int
    mask_path: str
    coverage_ratio: float
    iou_with_reference: float


@dataclass
class VideoPropagationResult:
    masks: list[PropagatedMask] = field(default_factory=list)
    reference_frame_index: int = 0
    total_frames_tracked: int = 0
    mean_coverage: float = 0.0
    frames_with_damage: int = 0


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up a propagated mask with morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, np.uint8(255), np.uint8(0))


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks (resizing if needed)."""
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def propagate_masks(
    key_frame_paths: list[str],
    reference_frame_index: int,
    click_points: list[dict],
    output_dir: str | Path,
) -> VideoPropagationResult:
    """
    Propagate the damage segmentation from the reference frame across
    all key frames using SAM 2's video predictor.

    Args:
        key_frame_paths: Paths to all extracted key frame PNGs.
        reference_frame_index: Which frame index has the user's click points.
        click_points: List of {x, y, label} dicts from user interaction.
        output_dir: Directory to save per-frame mask PNGs.

    Returns:
        VideoPropagationResult with per-frame mask metadata.
    """
    if len(key_frame_paths) < 2:
        raise SegmentationError(
            "Video propagation requires at least 2 key frames."
        )

    if reference_frame_index < 0 or reference_frame_index >= len(key_frame_paths):
        raise SegmentationError(
            f"Reference frame index {reference_frame_index} out of range "
            f"(0-{len(key_frame_paths) - 1})."
        )

    processor, model = _load_video_model()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_pil: list[Image.Image] = []
    for p in key_frame_paths:
        img = cv2.imread(p)
        if img is None:
            raise SegmentationError(f"Cannot read key frame: {p}")
        frames_pil.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    ref_size = frames_pil[reference_frame_index].size
    ref_h, ref_w = ref_size[1], ref_size[0]

    pos_pts = [[p["x"], p["y"]] for p in click_points if p.get("label", 1) == 1]
    neg_pts = [[p["x"], p["y"]] for p in click_points if p.get("label", 1) == 0]
    all_pts = pos_pts + neg_pts
    all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)

    if not all_pts:
        raise SegmentationError("No click points provided for propagation.")

    device = settings.device

    inference_session = processor.init_video_session(
        video=frames_pil,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    input_points = [[[pt for pt in all_pts]]]
    input_labels = [[all_labels]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=reference_frame_index,
        obj_ids=1,
        input_points=input_points,
        input_labels=input_labels,
    )

    _ = model(
        inference_session=inference_session,
        frame_idx=reference_frame_index,
    )

    video_segments: dict[int, np.ndarray] = {}

    for sam2_output in model.propagate_in_video_iterator(inference_session):
        frame_idx = sam2_output.frame_idx
        res_masks = processor.post_process_masks(
            [sam2_output.pred_masks],
            original_sizes=[[inference_session.video_height,
                             inference_session.video_width]],
            binarize=False,
        )[0]
        logits = res_masks.squeeze().cpu().numpy()
        binary = (logits > 0).astype(np.uint8) * 255
        binary = _postprocess_mask(binary)
        video_segments[frame_idx] = binary

    ref_mask = video_segments.get(reference_frame_index)
    if ref_mask is None:
        raise SegmentationError(
            "SAM 2 video propagation did not produce a mask for the reference frame."
        )

    result = VideoPropagationResult(
        reference_frame_index=reference_frame_index,
        total_frames_tracked=len(video_segments),
    )

    coverage_values: list[float] = []
    damage_count = 0

    for i in range(len(key_frame_paths)):
        mask = video_segments.get(i)
        mask_path = out_dir / f"prop_mask_{i:04d}.png"

        if mask is None:
            empty = np.zeros((ref_h, ref_w), dtype=np.uint8)
            cv2.imwrite(str(mask_path), empty)
            result.masks.append(PropagatedMask(
                frame_index=i, mask_path=str(mask_path),
                coverage_ratio=0.0, iou_with_reference=0.0,
            ))
            continue

        if mask.shape != (ref_h, ref_w):
            mask = cv2.resize(mask, (ref_w, ref_h),
                              interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(mask_path), mask)

        img_area = mask.shape[0] * mask.shape[1]
        mask_area = int(np.sum(mask > 0))
        coverage = mask_area / img_area if img_area > 0 else 0.0
        iou = _compute_iou(ref_mask, mask)

        min_coverage = 0.001
        has_damage = coverage >= min_coverage

        if has_damage:
            damage_count += 1
            coverage_values.append(coverage)

        result.masks.append(PropagatedMask(
            frame_index=i, mask_path=str(mask_path),
            coverage_ratio=round(coverage, 5),
            iou_with_reference=round(iou, 4),
        ))

    result.frames_with_damage = damage_count
    result.mean_coverage = (
        round(float(np.mean(coverage_values)), 5)
        if coverage_values else 0.0
    )

    logger.info(
        "Video propagation complete: %d/%d frames tracked, %d with damage, "
        "mean coverage %.4f",
        result.total_frames_tracked, len(key_frame_paths),
        result.frames_with_damage, result.mean_coverage,
    )

    return result
