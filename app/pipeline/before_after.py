"""
Before/After image comparison for PatchForge.

Aligns a reference ("before") image to the current ("after") image using
feature matching, then computes the pixel-level difference to automatically
detect the missing/broken region without requiring user click points.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.exceptions import SegmentationError

logger = logging.getLogger("patchforge.before_after")

MIN_MATCH_COUNT = 6
DIFF_BLUR_KSIZE = 7
MORPH_KERNEL_SIZE = 7
MIN_DAMAGE_AREA_RATIO = 0.0005
MAX_DAMAGE_AREA_RATIO = 0.5
MATCHING_RESOLUTION = 1200


@dataclass
class BeforeAfterResult:
    mask: np.ndarray
    contours: list[np.ndarray]
    aligned_before: np.ndarray
    diff_map: np.ndarray
    num_matches: int
    alignment_confidence: float
    damage_coverage: float
    centroid: tuple[int, int]


def _resize_to_match(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Resize source to match target's dimensions via fit-inside + pad.

    Uses min scale to keep all content visible (no cropping), then pads
    the shorter dimension with border pixels to reach the target size.
    This preserves all features for matching.
    """
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    if (sh, sw) == (th, tw):
        return source

    scale = min(tw / sw, th / sh)
    new_w = int(sw * scale)
    new_h = int(sh * scale)
    resized = cv2.resize(source, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left

    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REPLICATE,
    )


def _enhance_for_matching(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to boost feature contrast for better matching."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _knn_ratio_filter(
    raw_matches: list,
    ratio: float = 0.75,
) -> list:
    """Apply Lowe's ratio test to KNN matches."""
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def _try_detector(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    detector_name: str,
) -> tuple[list, list, list] | None:
    """Try a specific feature detector. Returns (kp1, kp2, good_matches) or None."""
    try:
        if detector_name == "ORB":
            det = cv2.ORB_create(
                nfeatures=10000,
                scaleFactor=1.2,
                nLevels=16,
                edgeThreshold=15,
                patchSize=31,
                fastThreshold=10,
            )
            norm = cv2.NORM_HAMMING
        elif detector_name == "AKAZE":
            det = cv2.AKAZE_create(threshold=0.001)
            norm = cv2.NORM_HAMMING
        elif detector_name == "SIFT":
            det = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.03)
            norm = cv2.NORM_L2
        else:
            return None
    except Exception:
        return None

    kp1, des1 = det.detectAndCompute(before_gray, None)
    kp2, des2 = det.detectAndCompute(after_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.info("%s: insufficient keypoints (kp1=%d, kp2=%d)",
                    detector_name, len(kp1) if kp1 else 0, len(kp2) if kp2 else 0)
        return None

    if norm == cv2.NORM_L2:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        try:
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        except Exception:
            matcher = cv2.BFMatcher(norm, crossCheck=False)

    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = _knn_ratio_filter(raw_matches, ratio=0.75)

    if len(good) < MIN_MATCH_COUNT:
        good = _knn_ratio_filter(raw_matches, ratio=0.80)
    if len(good) < MIN_MATCH_COUNT:
        good = _knn_ratio_filter(raw_matches, ratio=0.85)
        if len(good) >= MIN_MATCH_COUNT:
            logger.info("%s: needed relaxed ratio (0.85) to get %d matches", detector_name, len(good))
    if len(good) < MIN_MATCH_COUNT:
        good = _knn_ratio_filter(raw_matches, ratio=0.90)
        if len(good) >= MIN_MATCH_COUNT:
            logger.info("%s: needed very relaxed ratio (0.90) to get %d matches", detector_name, len(good))

    if len(good) < MIN_MATCH_COUNT:
        logger.info("%s: only %d matches (need %d)", detector_name, len(good), MIN_MATCH_COUNT)
        return None

    logger.info("%s: %d good matches from %d raw (kp1=%d, kp2=%d)",
                detector_name, len(good), len(raw_matches), len(kp1), len(kp2))
    return kp1, kp2, good


def _detect_and_match(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
) -> tuple[list, list, list]:
    """Detect features and match between before/after images.

    Tries multiple detectors (ORB -> AKAZE -> SIFT) on both raw and
    CLAHE-enhanced images for maximum robustness.
    """
    detectors = ["ORB", "AKAZE", "SIFT"]

    for enhanced in [False, True]:
        bg = _enhance_for_matching(before_gray) if enhanced else before_gray
        ag = _enhance_for_matching(after_gray) if enhanced else after_gray
        label = "enhanced" if enhanced else "raw"

        for det_name in detectors:
            result = _try_detector(bg, ag, det_name)
            if result is not None:
                logger.info("Matched with %s (%s): %d matches", det_name, label, len(result[2]))
                return result

    raise SegmentationError(
        "Could not find enough feature matches between before and after images. "
        "Try taking the photos from a more similar angle, or ensure there are "
        "overlapping background elements visible in both shots."
    )


def _upscale_for_matching(
    img: np.ndarray,
    target_long_edge: int,
) -> tuple[np.ndarray, float]:
    """Upscale an image so its long edge is at least target_long_edge pixels.

    Returns (resized_image, scale_factor_used).
    """
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge >= target_long_edge:
        return img, 1.0
    scale = target_long_edge / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC), scale


def _align_images(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """
    Align the before image to the after image's perspective using homography.

    Returns:
        aligned_before: Warped before image matching after's geometry.
        num_matches: Number of feature matches used.
        confidence: Alignment confidence (inlier ratio).
    """
    # Upscale both images to a common resolution for feature matching.
    # This prevents losing features when the after image comes from a video
    # (often 848x480) while the before image is high-res (2048x1536).
    before_up, s_before = _upscale_for_matching(before_bgr, MATCHING_RESOLUTION)
    after_up, s_after = _upscale_for_matching(after_bgr, MATCHING_RESOLUTION)

    before_match = _resize_to_match(before_up, after_up)
    before_gray = cv2.cvtColor(before_match, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_up, cv2.COLOR_BGR2GRAY)

    logger.info(
        "Aligning images: before %s (up %.2fx) -> match %s, after %s (up %.2fx) -> %s",
        before_bgr.shape[:2], s_before, before_match.shape[:2],
        after_bgr.shape[:2], s_after, after_up.shape[:2],
    )

    kp1, kp2, good = _detect_and_match(before_gray, after_gray)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    h_up, w_up = after_up.shape[:2]
    h_orig, w_orig = after_bgr.shape[:2]
    aligned_up = None
    inlier_count = 0
    method_used = "homography"

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        cond = np.linalg.cond(H)
        if cond < 1e6:
            inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
            confidence = inlier_count / len(good) if len(good) > 0 else 0.0
            if confidence >= 0.2:
                aligned_up = cv2.warpPerspective(before_match, H, (w_up, h_up))
                method_used = "homography"
            else:
                logger.info("Homography confidence too low (%.1f%%), trying affine...", confidence * 100)
        else:
            logger.info("Homography degenerate (cond=%.0f), trying affine...", cond)

    if aligned_up is None and len(good) >= 3:
        M, inlier_mask_aff = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is not None:
            inlier_count = int(inlier_mask_aff.sum()) if inlier_mask_aff is not None else 0
            aligned_up = cv2.warpAffine(before_match, M, (w_up, h_up))
            method_used = "affine"
            logger.info("Affine alignment succeeded with %d inliers", inlier_count)

    if aligned_up is None:
        raise SegmentationError(
            "Could not compute alignment between before and after images. "
            "Try taking both photos from a more similar angle."
        )

    confidence = inlier_count / len(good) if len(good) > 0 else 0.0
    logger.info("Alignment method: %s, confidence: %.1f%%", method_used, confidence * 100)

    # Scale the aligned result back to the after image's original resolution
    aligned = cv2.resize(aligned_up, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

    return aligned, len(good), confidence


def _compute_diff_mask(
    aligned_before: np.ndarray,
    after_bgr: np.ndarray,
    threshold: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the difference between aligned before and after images.

    Uses multi-channel absolute difference with adaptive thresholding
    to handle lighting variations.

    Returns:
        binary_mask: uint8 mask (255 = damage region).
        diff_map: Raw grayscale difference image (for visualization).
    """
    before_gray = cv2.cvtColor(aligned_before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)

    before_blur = cv2.GaussianBlur(before_gray, (DIFF_BLUR_KSIZE, DIFF_BLUR_KSIZE), 0)
    after_blur = cv2.GaussianBlur(after_gray, (DIFF_BLUR_KSIZE, DIFF_BLUR_KSIZE), 0)

    diff = cv2.absdiff(before_blur, after_blur)

    before_lab = cv2.cvtColor(aligned_before, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2LAB)
    color_diff = np.mean(np.abs(
        before_lab.astype(np.float32) - after_lab.astype(np.float32)
    ), axis=2).astype(np.uint8)

    combined = cv2.addWeighted(diff, 0.5, color_diff, 0.5, 0)

    _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

    # Also try Otsu for robustness — use the one that produces a more reasonable mask
    _, otsu_binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_area = binary.shape[0] * binary.shape[1]
    fixed_ratio = np.sum(binary > 0) / img_area
    otsu_ratio = np.sum(otsu_binary > 0) / img_area

    if MIN_DAMAGE_AREA_RATIO <= otsu_ratio <= MAX_DAMAGE_AREA_RATIO:
        binary = otsu_binary
    elif not (MIN_DAMAGE_AREA_RATIO <= fixed_ratio <= MAX_DAMAGE_AREA_RATIO):
        found = False
        for t in [20, 25, 35, 40, 50]:
            _, trial = cv2.threshold(combined, t, 255, cv2.THRESH_BINARY)
            r = np.sum(trial > 0) / img_area
            if MIN_DAMAGE_AREA_RATIO <= r <= MAX_DAMAGE_AREA_RATIO:
                binary = trial
                found = True
                break
        if not found:
            raise SegmentationError(
                "Could not find a valid damage mask at any threshold. "
                "The images may be too similar or lighting differs too much."
            )

    return binary, combined


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up the difference mask with morphological operations.

    Instead of naively picking the largest connected component, scores each
    component by centrality (closer to image center), compactness (area vs
    perimeter — real damage tends to be blobby, not edge-hugging), and
    rejects components that sit flush against the image border.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    img_area = h * w
    cx_img, cy_img = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)
    edge_margin = int(min(h, w) * 0.02) + 1

    best_label = 1
    best_score = -1.0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        area_ratio = area / img_area

        if area_ratio < MIN_DAMAGE_AREA_RATIO:
            continue

        bx = stats[label_id, cv2.CC_STAT_LEFT]
        by = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]

        touches_left = bx <= edge_margin
        touches_top = by <= edge_margin
        touches_right = (bx + bw) >= (w - edge_margin)
        touches_bottom = (by + bh) >= (h - edge_margin)
        edge_count = sum([touches_left, touches_top, touches_right, touches_bottom])

        if edge_count >= 3:
            continue

        ccx, ccy = centroids[label_id]
        dist = np.sqrt((ccx - cx_img) ** 2 + (ccy - cy_img) ** 2)
        centrality = 1.0 - (dist / max_dist)

        comp_contours, _ = cv2.findContours(
            np.where(labels == label_id, np.uint8(255), np.uint8(0)),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if comp_contours:
            perimeter = cv2.arcLength(comp_contours[0], True)
        else:
            perimeter = 0.0
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        edge_penalty = 1.0 - 0.3 * edge_count

        score = (
            0.3 * centrality
            + 0.3 * compactness
            + 0.2 * min(area_ratio / 0.1, 1.0)
            + 0.2 * edge_penalty
        )

        logger.debug(
            "Component %d: area=%.4f centrality=%.2f compact=%.2f "
            "edges=%d score=%.3f",
            label_id, area_ratio, centrality, compactness, edge_count, score,
        )

        if score > best_score:
            best_score = score
            best_label = label_id

    logger.info("Selected component %d (score=%.3f)", best_label, best_score)
    return np.where(labels == best_label, np.uint8(255), np.uint8(0))


def _compute_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Find the centroid of the mask for potential SAM 2 refinement."""
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        h, w = mask.shape
        return w // 2, h // 2
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def detect_damage(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    diff_threshold: int = 30,
) -> BeforeAfterResult:
    """
    Full before/after damage detection pipeline.

    1. Align the before image to the after image using ORB + homography.
    2. Compute pixel-level difference (grayscale + LAB color space).
    3. Threshold and clean the difference into a binary damage mask.
    4. Extract contours and compute the centroid.

    Args:
        before_bgr: The reference "intact" image (BGR).
        after_bgr: The current "broken" image (BGR).
        diff_threshold: Pixel difference threshold (0-255).

    Returns:
        BeforeAfterResult with mask, contours, aligned before, and stats.
    """
    aligned_before, num_matches, alignment_confidence = _align_images(
        before_bgr, after_bgr,
    )

    logger.info(
        "Image alignment: %d matches, %.1f%% inlier confidence",
        num_matches, alignment_confidence * 100,
    )

    raw_mask, diff_map = _compute_diff_mask(
        aligned_before, after_bgr, threshold=diff_threshold,
    )

    clean_mask = _postprocess_mask(raw_mask)

    img_area = clean_mask.shape[0] * clean_mask.shape[1]
    damage_area = int(np.sum(clean_mask > 0))
    damage_coverage = damage_area / img_area if img_area > 0 else 0.0

    if damage_coverage < MIN_DAMAGE_AREA_RATIO:
        raise SegmentationError(
            "No significant difference detected between before and after images. "
            "The images may be too similar or alignment failed."
        )

    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise SegmentationError(
            "No contours found in the difference mask."
        )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    centroid = _compute_centroid(clean_mask)

    logger.info(
        "Before/after detection: %.2f%% coverage, %d contours, centroid=(%d, %d)",
        damage_coverage * 100, len(contours), centroid[0], centroid[1],
    )

    return BeforeAfterResult(
        mask=clean_mask,
        contours=contours,
        aligned_before=aligned_before,
        diff_map=diff_map,
        num_matches=num_matches,
        alignment_confidence=round(alignment_confidence, 3),
        damage_coverage=round(damage_coverage, 5),
        centroid=centroid,
    )
