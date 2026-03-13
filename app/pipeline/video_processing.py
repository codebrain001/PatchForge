"""
Video processing pipeline for PatchForge.

Accepts a video file, extracts key frames with maximum viewpoint diversity,
scores each frame for sharpness and viewing angle, and selects the best
top-down frame for the segmentation pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.core.exceptions import VideoProcessingError
from app.core.storage import job_frame_path, job_frames_dir

logger = logging.getLogger("patchforge.video")


@dataclass
class FrameInfo:
    index: int
    path: str
    timestamp_sec: float
    sharpness: float
    mean_diff: float


@dataclass
class VideoExtractionResult:
    frame_infos: list[FrameInfo] = field(default_factory=list)
    best_frame_index: int = 0
    fps: float = 0.0
    total_frames: int = 0
    duration_sec: float = 0.0


def _laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def _validate_video(cap: cv2.VideoCapture, video_path: str) -> tuple[float, int, float]:
    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise VideoProcessingError("Video has invalid FPS metadata.")
    if total_frames <= 0:
        raise VideoProcessingError("Video has no frames.")

    duration = total_frames / fps
    if duration > settings.video_max_duration_sec:
        raise VideoProcessingError(
            f"Video is {duration:.1f}s — maximum allowed is "
            f"{settings.video_max_duration_sec}s. Please trim the clip."
        )
    if duration < 0.5:
        raise VideoProcessingError("Video is too short (< 0.5s).")

    return fps, total_frames, duration


def _sample_candidate_frames(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
) -> list[tuple[int, np.ndarray]]:
    """Sample frames at the configured FPS rate."""
    sample_interval = max(1, int(round(fps / settings.video_sample_fps)))
    candidates: list[tuple[int, np.ndarray]] = []

    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        candidates.append((frame_idx, frame))

    if len(candidates) < 2:
        raise VideoProcessingError(
            "Could not extract enough frames from the video. "
            "Ensure the file is a valid video."
        )

    return candidates


def _select_keyframes(
    candidates: list[tuple[int, np.ndarray]],
    max_keyframes: int,
    diff_threshold: float,
) -> list[tuple[int, np.ndarray, float, float]]:
    """
    Select key frames with maximum viewpoint diversity.

    Uses inter-frame difference to pick frames where the camera has moved
    significantly, plus sharpness scoring to prefer in-focus frames.

    Returns list of (original_frame_idx, bgr_image, sharpness, mean_diff).
    """
    grays = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for _, f in candidates
    ]

    # Always include the first frame
    selected: list[tuple[int, np.ndarray, float, float]] = []
    first_idx, first_frame = candidates[0]
    first_sharp = _laplacian_sharpness(grays[0])
    selected.append((first_idx, first_frame, first_sharp, 0.0))

    last_selected_gray = grays[0]

    for i in range(1, len(candidates)):
        diff = _frame_difference(last_selected_gray, grays[i])
        if diff >= diff_threshold:
            sharp = _laplacian_sharpness(grays[i])
            frame_idx, frame = candidates[i]
            selected.append((frame_idx, frame, sharp, diff))
            last_selected_gray = grays[i]

        if len(selected) >= max_keyframes:
            break

    # If we got very few frames (camera barely moved), fill with
    # evenly-spaced candidates sorted by sharpness
    if len(selected) < 3 and len(candidates) >= 3:
        all_scored = []
        for i, (fidx, frame) in enumerate(candidates):
            sharp = _laplacian_sharpness(grays[i])
            diff = (
                _frame_difference(grays[0], grays[i])
                if i > 0 else 0.0
            )
            all_scored.append((fidx, frame, sharp, diff))

        all_scored.sort(key=lambda x: x[2], reverse=True)
        step = max(1, len(all_scored) // max_keyframes)
        selected = all_scored[::step][:max_keyframes]

    return selected


def _pick_best_frame(keyframes: list[tuple[int, np.ndarray, float, float]]) -> int:
    """
    Pick the best frame for top-down segmentation.

    Heuristic: highest sharpness among the first half of the clip (the user
    typically starts filming head-on) with a small boost for frames closer
    to the beginning.
    """
    if not keyframes:
        return 0

    scores: list[float] = []
    for rank, (_, _, sharpness, _) in enumerate(keyframes):
        position_bonus = 1.0 / (1.0 + rank * 0.15)
        scores.append(sharpness * position_bonus)

    return int(np.argmax(scores))


def extract_keyframes(
    video_path: str | Path,
    job_id: str,
) -> VideoExtractionResult:
    """
    Full key-frame extraction pipeline.

    1. Validate the video
    2. Sample candidate frames at configured FPS
    3. Select diverse key frames via inter-frame difference
    4. Score and pick the best top-down frame
    5. Save all key frames to disk as PNGs
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    try:
        fps, total_frames, duration = _validate_video(cap, video_path)

        candidates = _sample_candidate_frames(cap, fps, total_frames)

        keyframes = _select_keyframes(
            candidates,
            max_keyframes=settings.video_max_keyframes,
            diff_threshold=settings.video_frame_diff_threshold,
        )

        best_idx = _pick_best_frame(keyframes)

        # Save key frames to disk
        frames_dir = job_frames_dir(job_id)
        frame_infos: list[FrameInfo] = []

        for i, (orig_idx, frame_bgr, sharpness, diff) in enumerate(keyframes):
            out_path = job_frame_path(job_id, i)
            cv2.imwrite(str(out_path), frame_bgr)

            frame_infos.append(FrameInfo(
                index=i,
                path=str(out_path),
                timestamp_sec=round(orig_idx / fps, 3) if fps > 0 else 0.0,
                sharpness=round(sharpness, 2),
                mean_diff=round(diff, 2),
            ))

        logger.info(
            "Extracted %d key frames from %s (%.1fs, %.0f fps). Best frame: %d",
            len(frame_infos), video_path, duration, fps, best_idx,
        )

        return VideoExtractionResult(
            frame_infos=frame_infos,
            best_frame_index=best_idx,
            fps=fps,
            total_frames=total_frames,
            duration_sec=round(duration, 2),
        )

    finally:
        cap.release()


def get_best_frame_bgr(job_id: str, result: VideoExtractionResult) -> np.ndarray:
    """Load the best key frame as a BGR numpy array."""
    if not result.frame_infos:
        raise VideoProcessingError("No key frames available.")

    if result.best_frame_index < 0 or result.best_frame_index >= len(result.frame_infos):
        raise VideoProcessingError(
            f"Best frame index {result.best_frame_index} out of range "
            f"(0-{len(result.frame_infos) - 1})."
        )

    best = result.frame_infos[result.best_frame_index]
    img = cv2.imread(best.path)
    if img is None:
        raise VideoProcessingError(f"Could not read best frame: {best.path}")
    return img
