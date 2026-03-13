"""
Prompt-to-mesh pipeline.

Parses a natural language description into structured shape parameters
via LLM, then generates a 3D mesh (STL) from the parsed shape.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon

from app.core.exceptions import MeshGenerationError
from app.models.job import MeshResult

logger = logging.getLogger("patchforge.prompt_to_mesh")

PARSE_SYSTEM_PROMPT = """\
You are a precise engineering assistant that converts natural language
descriptions of patches or 3D-printable parts into structured JSON.
The target printer is a Bambu Lab A1 with a maximum build volume of
256 x 256 x 256 mm. All dimensions must fit within this volume.

RULES:
- Extract the shape type, all dimensions, and thickness.
- Reject or warn if any dimension exceeds 256 mm.
- Resolve size references to millimetres using common objects:
    UK 1 pound coin = 23.43 mm diameter
    UK 2 pound coin = 28.4 mm diameter
    UK 1 penny = 20.3 mm diameter
    US quarter = 24.26 mm diameter
    US penny = 19.05 mm diameter
    Euro 1 coin = 23.25 mm diameter
    Euro 2 coin = 25.75 mm diameter
    ZAR R5 = 26 mm, ZAR R2 = 23 mm, ZAR R1 = 20 mm
    Credit card = 85.6 x 53.98 mm
    Tennis ball = 67 mm diameter
    Golf ball = 42.67 mm diameter
    AA battery = 14.5 mm diameter, 50.5 mm length
    Adult index finger width ≈ 17 mm
    Adult thumb width ≈ 20 mm
- If the user says "the size of X", use the real-world size of X.
- If no thickness is specified, default to 3.0 mm.
- shape_type must be one of: circle, rectangle, ellipse, triangle, hexagon,
  star, sphere, cylinder, cube.
- is_3d_primitive should be true ONLY for sphere, cylinder, or cube.

Respond with ONLY valid JSON (no markdown, no explanation):
{
    "shape_type": "<circle|rectangle|ellipse|triangle|hexagon|star|sphere|cylinder|cube>",
    "width_mm": <number>,
    "height_mm": <number>,
    "thickness_mm": <number>,
    "diameter_mm": <number or null>,
    "is_3d_primitive": <true|false>,
    "description": "<short description of what was requested>"
}
"""


async def parse_prompt(prompt: str) -> dict[str, Any]:
    """Parse a natural language prompt into structured shape parameters."""
    import asyncio
    from app.core.llm import call_llm, parse_json_response

    text, provider = await asyncio.to_thread(
        call_llm, PARSE_SYSTEM_PROMPT, prompt,
    )
    logger.info("Prompt parsed via %s", provider)

    parsed = parse_json_response(text)

    shape_type = parsed.get("shape_type", "circle")
    width = float(parsed.get("width_mm") or parsed.get("diameter_mm") or 20)
    height = float(parsed.get("height_mm") or width)
    thickness = float(parsed.get("thickness_mm") or 3.0)
    diameter = parsed.get("diameter_mm")
    is_3d = bool(parsed.get("is_3d_primitive", False))
    description = parsed.get("description", prompt)

    if diameter and shape_type in ("circle", "sphere", "cylinder"):
        width = float(diameter)
        height = float(diameter)

    width = max(1.0, min(width, 256.0))
    height = max(1.0, min(height, 256.0))
    thickness = max(0.5, min(thickness, 256.0))

    result = {
        "shape_type": shape_type,
        "width_mm": round(width, 2),
        "height_mm": round(height, 2),
        "thickness_mm": round(thickness, 2),
        "is_3d_primitive": is_3d,
        "description": description,
    }
    logger.info("Parsed shape: %s", result)
    return result


def _create_circle_polygon(diameter_mm: float, n_points: int = 64) -> ShapelyPolygon:
    radius = diameter_mm / 2.0
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    points = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    return ShapelyPolygon(points)


def _create_ellipse_polygon(
    width_mm: float, height_mm: float, n_points: int = 64,
) -> ShapelyPolygon:
    rx, ry = width_mm / 2.0, height_mm / 2.0
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    points = [(rx * math.cos(a), ry * math.sin(a)) for a in angles]
    return ShapelyPolygon(points)


def _create_rectangle_polygon(width_mm: float, height_mm: float) -> ShapelyPolygon:
    hw, hh = width_mm / 2.0, height_mm / 2.0
    return ShapelyPolygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])


def _create_triangle_polygon(width_mm: float, height_mm: float) -> ShapelyPolygon:
    hw = width_mm / 2.0
    return ShapelyPolygon([(-hw, 0), (hw, 0), (0, height_mm)])


def _create_hexagon_polygon(diameter_mm: float) -> ShapelyPolygon:
    radius = diameter_mm / 2.0
    angles = np.linspace(0, 2 * math.pi, 7, endpoint=True)
    points = [(radius * math.cos(a), radius * math.sin(a)) for a in angles[:6]]
    return ShapelyPolygon(points)


def _create_star_polygon(
    outer_diameter_mm: float, n_points: int = 5,
) -> ShapelyPolygon:
    outer_r = outer_diameter_mm / 2.0
    inner_r = outer_r * 0.4
    points = []
    for i in range(n_points * 2):
        angle = math.pi / 2 + i * math.pi / n_points
        r = outer_r if i % 2 == 0 else inner_r
        points.append((r * math.cos(angle), r * math.sin(angle)))
    return ShapelyPolygon(points)


def generate_mesh_from_shape(
    parsed_shape: dict[str, Any],
    output_path: Path,
    chamfer_mm: float = 0.0,
) -> MeshResult:
    """Generate an STL mesh from parsed shape parameters."""
    shape_type = parsed_shape["shape_type"]
    width = parsed_shape["width_mm"]
    height = parsed_shape["height_mm"]
    thickness = parsed_shape["thickness_mm"]
    is_3d = parsed_shape.get("is_3d_primitive", False)

    mesh: trimesh.Trimesh

    if is_3d and shape_type == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=width / 2.0)

    elif is_3d and shape_type == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=width / 2.0, height=thickness,
        )

    elif is_3d and shape_type == "cube":
        mesh = trimesh.creation.box(extents=[width, height, thickness])

    else:
        if shape_type == "circle":
            poly = _create_circle_polygon(width)
        elif shape_type == "ellipse":
            poly = _create_ellipse_polygon(width, height)
        elif shape_type == "rectangle":
            poly = _create_rectangle_polygon(width, height)
        elif shape_type == "triangle":
            poly = _create_triangle_polygon(width, height)
        elif shape_type == "hexagon":
            poly = _create_hexagon_polygon(width)
        elif shape_type == "star":
            poly = _create_star_polygon(width)
        else:
            poly = _create_rectangle_polygon(width, height)

        if poly.is_empty or not poly.is_valid:
            raise MeshGenerationError(f"Generated {shape_type} polygon is invalid.")

        try:
            mesh = trimesh.creation.extrude_polygon(poly, height=thickness)
        except Exception as e:
            raise MeshGenerationError(f"Extrusion failed for {shape_type}: {e}") from e

    mesh.process(validate=True)

    if not mesh.is_volume:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            mesh.process(validate=True)
        except Exception as e:
            logger.warning("Mesh repair failed (non-fatal): %s", e)

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
