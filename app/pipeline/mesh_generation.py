from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from app.core.exceptions import MeshGenerationError
from app.models.job import MeshResult

logger = logging.getLogger("patchforge.mesh")


def _smooth_contour(contour: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth a contour to remove pixel staircase while preserving angular shape.

    For fundamentally simple shapes (rectangles, triangles), uses aggressive
    polygonal approximation to extract the core shape. Only applies Gaussian
    smoothing to complex, organic contours that benefit from it.
    """
    pts = np.squeeze(contour)
    if pts.ndim != 2 or pts.shape[0] < 5:
        return contour

    if pts.shape[0] < 10:
        return contour

    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if contour_area < 1:
        return contour

    _, _, rect_w, rect_h = cv2.boundingRect(contour)
    rect_area = rect_w * rect_h
    fill_ratio = contour_area / rect_area if rect_area > 0 else 0

    if fill_ratio > 0.70:
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        approx_pts = np.squeeze(approx)
        if approx_pts.ndim == 2 and 3 <= approx_pts.shape[0] <= 8:
            logger.info(
                "Shape is simple (fill=%.0f%%), reduced %d -> %d vertices",
                fill_ratio * 100, pts.shape[0], approx_pts.shape[0],
            )
            return approx
    else:
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        approx_pts = np.squeeze(approx)
        if approx_pts.ndim == 2 and 3 <= approx_pts.shape[0] <= 10:
            logger.info(
                "Shape simplified to %d vertices (fill=%.0f%%)",
                approx_pts.shape[0], fill_ratio * 100,
            )
            return approx

    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    pts = np.squeeze(approx).astype(np.float64)

    if pts.ndim != 2 or pts.shape[0] < 4:
        return approx

    if sigma > 0 and pts.shape[0] > 15:
        xs = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="wrap")
        ys = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="wrap")
        pts = np.stack([xs, ys], axis=1)

    return pts.astype(np.float32).reshape(-1, 1, 2)


def _contour_to_polygon(
    contour: np.ndarray,
    simplify_tolerance: float = 0.5,
    smooth_radius: float = 0.0,
) -> ShapelyPolygon:
    pts = np.squeeze(contour)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise MeshGenerationError("Contour has fewer than 3 points.")

    poly = ShapelyPolygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if not isinstance(poly, ShapelyPolygon):
            raise MeshGenerationError(f"Could not form a valid polygon (got {poly.geom_type}).")

    poly = poly.simplify(simplify_tolerance, preserve_topology=True)

    # Light morphological smoothing only for high-vertex contours (pixel masks).
    # Vision-traced polygons (few vertices) should keep their exact shape.
    if smooth_radius > 0 and len(poly.exterior.coords) > 12:
        smoothed = poly.buffer(smooth_radius, join_style=2).buffer(-smooth_radius, join_style=2)
        if not smoothed.is_empty and smoothed.area > poly.area * 0.7:
            if smoothed.geom_type == "MultiPolygon":
                smoothed = max(smoothed.geoms, key=lambda g: g.area)
            if isinstance(smoothed, ShapelyPolygon):
                poly = smoothed

    if poly.is_empty or poly.area < 1.0:
        raise MeshGenerationError("Polygon is empty or too small after simplification.")

    return poly


def _scale_polygon(poly: ShapelyPolygon, scale_factor: float) -> ShapelyPolygon:
    from shapely.affinity import scale as shapely_scale
    return shapely_scale(poly, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))


def _apply_chamfer(mesh: trimesh.Trimesh, chamfer_mm: float) -> trimesh.Trimesh:
    """
    Approximate a chamfer by slightly insetting and offsetting the top/bottom faces.
    Uses convex hull of the offset polygon for robustness.
    """
    if chamfer_mm <= 0:
        return mesh

    try:
        bounds = mesh.bounds
        z_min, z_max = bounds[0][2], bounds[1][2]
        height = z_max - z_min

        if chamfer_mm * 2 >= height:
            return mesh

        verts = mesh.vertices.copy()
        for i, v in enumerate(verts):
            if abs(v[2] - z_min) < 0.01 or abs(v[2] - z_max) < 0.01:
                cx = (bounds[0][0] + bounds[1][0]) / 2
                cy = (bounds[0][1] + bounds[1][1]) / 2
                dx, dy = v[0] - cx, v[1] - cy
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    shrink = min(chamfer_mm, dist * 0.3)
                    factor = (dist - shrink) / dist
                    verts[i][0] = cx + dx * factor
                    verts[i][1] = cy + dy * factor
                    if abs(v[2] - z_min) < 0.01:
                        verts[i][2] += chamfer_mm
                    else:
                        verts[i][2] -= chamfer_mm

        mesh.vertices = verts
        mesh.fix_normals()
    except Exception as e:
        logger.warning("Chamfer application failed (non-fatal): %s", e)

    return mesh


def generate_mesh(
    contour: np.ndarray,
    scale_factor: float,
    thickness_mm: float,
    output_path: Path,
    simplify_tolerance: float = 2.0,
    chamfer_mm: float = 0.0,
) -> MeshResult:
    n_pts = len(np.squeeze(contour))
    is_low_vertex = n_pts < 12

    sigma = 0.0 if is_low_vertex else 0.5
    simp_tol = 0.2 if is_low_vertex else simplify_tolerance

    smoothed_contour = _smooth_contour(contour, sigma=sigma)
    smoothed_pts = len(np.squeeze(smoothed_contour))
    smooth_radius = 0.0 if smoothed_pts <= 12 else 0.3

    poly_px = _contour_to_polygon(smoothed_contour, simp_tol, smooth_radius=smooth_radius)
    poly_mm = _scale_polygon(poly_px, scale_factor)

    try:
        mesh = trimesh.creation.extrude_polygon(poly_mm, height=thickness_mm)
    except Exception as e:
        raise MeshGenerationError(f"Trimesh extrusion failed: {e}") from e

    mesh.process(validate=True)

    if not mesh.is_volume:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            mesh.process(validate=True)
        except Exception as e:
            logger.warning("Mesh repair failed (non-fatal): %s", e)

    if chamfer_mm > 0:
        mesh = _apply_chamfer(mesh, chamfer_mm)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="stl")

    bbox = mesh.bounding_box.bounds.tolist()

    return MeshResult(
        file_path=str(output_path),
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        volume_mm3=round(float(mesh.volume), 2),
        surface_area_mm2=round(float(mesh.area), 2),
        is_watertight=mesh.is_watertight,
        bounding_box_mm=bbox,
    )
