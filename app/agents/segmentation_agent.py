from __future__ import annotations

import cv2
import numpy as np

from app.agents.base import Agent, AgentResult

ROLE = (
    "You are an expert segmentation agent in a photo-to-3D-print patch pipeline. "
    "The goal is to segment the GAP/VOID where a piece broke off — NOT the whole object. "
    "We need the outline of the missing piece so we can 3D-print a replacement patch. "
    "You evaluate SAM 2 segmentation masks for quality: coverage ratio, edge smoothness, "
    "fragmentation, and whether the mask captures ONLY the break area. "
    "If the mask covers more than 30% of the image, it almost certainly segmented the "
    "whole object instead of just the gap — reject it. "
    "Suggest better click points or negative points to exclude the intact object."
)


class SegmentationAgent(Agent):
    def __init__(self):
        super().__init__("SegmentationAgent", ROLE)

    async def run(
        self,
        image_bgr: np.ndarray,
        points: list[dict],
    ) -> tuple[np.ndarray, list[np.ndarray], AgentResult]:
        from app.pipeline.segmentation import segment

        pos_points = [(p["x"], p["y"]) for p in points if p.get("label", 1) == 1]
        neg_points = [(p["x"], p["y"]) for p in points if p.get("label", 1) == 0]

        if not pos_points:
            raise ValueError("At least one positive click point is required for segmentation.")

        mask, contours = segment(
            image_bgr,
            click_x=pos_points[0][0],
            click_y=pos_points[0][1],
            additional_points=pos_points[1:],
            negative_points=neg_points,
        )

        img_area = image_bgr.shape[0] * image_bgr.shape[1]
        mask_area = int(np.sum(mask > 0))
        coverage_ratio = mask_area / img_area if img_area > 0 else 0

        contour = contours[0]
        perimeter = cv2.arcLength(contour, closed=True)
        area = cv2.contourArea(contour)
        compactness = (4 * 3.14159 * area) / (perimeter ** 2) if perimeter > 0 else 0

        context = {
            "mask_area_px": mask_area,
            "image_area_px": img_area,
            "coverage_ratio": round(coverage_ratio, 4),
            "num_contours": len(contours),
            "primary_contour_area": round(area, 1),
            "compactness": round(compactness, 3),
            "num_positive_points": len(pos_points),
            "num_negative_points": len(neg_points),
        }

        analysis = await self.analyze(context)
        return mask, contours, analysis
