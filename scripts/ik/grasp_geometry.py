"""Pure geometry construction for vertical sphere grasps."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from .kinematics_utils import finite_array, positive_scale


def normalized_vector(value: object, *, label: str) -> np.ndarray:
    """Return a finite unit three-vector."""
    vector = finite_array(value, shape=(3,), label=label)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} must have nonzero length")
    return vector / norm


def vertical_approach_waypoints(
    final_target_world_stage_units: object,
    *,
    meters_per_unit: float,
    hover_height_mm: float,
    descent_step_mm: float,
) -> list[np.ndarray]:
    """Return hover-to-contact waypoints on a world-vertical line."""
    final_target = finite_array(
        final_target_world_stage_units,
        shape=(3,),
        label="final top-down contact target",
    )
    scale = positive_scale(meters_per_unit, label="meters_per_unit")
    hover_height = positive_scale(hover_height_mm, label="hover_height_mm")
    descent_step = positive_scale(descent_step_mm, label="descent_step_mm")
    hover_height_stage_units = hover_height / 1000.0 / scale
    descent_step_stage_units = descent_step / 1000.0 / scale
    segment_count = max(
        1,
        math.ceil(hover_height_stage_units / descent_step_stage_units),
    )
    vertical_offsets = np.linspace(
        hover_height_stage_units,
        0.0,
        num=segment_count + 1,
        dtype=np.float64,
    )
    return [
        final_target + np.array([0.0, 0.0, offset], dtype=np.float64)
        for offset in vertical_offsets
    ]


def generate_sphere_grasp_candidates(
    sphere_center: object,
    sphere_radius: float,
    nominal_surface_point: object,
    *,
    meters_per_unit: float,
    surface_clearance_m: float,
    surface_offsets_deg: Sequence[float],
    approach_tilts_deg: Sequence[float],
    hover_height_mm: float,
    descent_step_mm: float,
) -> list[dict[str, Any]]:
    """Sample sphere-centered grasps and safe near-vertical tool axes."""
    center = finite_array(sphere_center, shape=(3,), label="sphere center")
    radius = positive_scale(sphere_radius, label="sphere_radius")
    scale = positive_scale(meters_per_unit, label="meters_per_unit")
    clearance_m = float(surface_clearance_m)
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("surface_clearance_m must be finite and nonnegative")
    clearance_stage_units = clearance_m / scale
    nominal = finite_array(
        nominal_surface_point,
        shape=(3,),
        label="nominal surface point",
    )
    nominal_radial = nominal - center
    nominal_radial[2] = 0.0
    nominal_radial = normalized_vector(
        nominal_radial,
        label="nominal grasp radial",
    )
    candidates: list[dict[str, Any]] = []
    for surface_offset_value in surface_offsets_deg:
        surface_offset_deg = float(surface_offset_value)
        if not math.isfinite(surface_offset_deg):
            raise ValueError("surface offsets must be finite")
        angle = math.radians(surface_offset_deg)
        radial = np.array(
            [
                math.cos(angle) * nominal_radial[0]
                - math.sin(angle) * nominal_radial[1],
                math.sin(angle) * nominal_radial[0]
                + math.cos(angle) * nominal_radial[1],
                0.0,
            ],
            dtype=np.float64,
        )
        inward = -radial
        surface_point = center + radius * radial
        approach_point = surface_point + clearance_stage_units * radial
        for approach_tilt_value in approach_tilts_deg:
            approach_tilt_deg = float(approach_tilt_value)
            if not math.isfinite(approach_tilt_deg):
                raise ValueError("approach tilts must be finite")
            tilt = math.radians(approach_tilt_deg)
            target_axis_world = normalized_vector(
                math.cos(tilt) * np.array([0.0, 0.0, -1.0])
                + math.sin(tilt) * inward,
                label="candidate approach axis",
            )
            candidate_id = (
                f"surface_{surface_offset_deg:+.0f}_tilt_"
                f"{approach_tilt_deg:.0f}"
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "surface_offset_deg": surface_offset_deg,
                    "approach_tilt_deg": approach_tilt_deg,
                    "target_kind": "fixed_finger_surface_vertical_approach",
                    "surface_point_world_stage_units": surface_point.copy(),
                    "approach_point_world_stage_units": approach_point.copy(),
                    "grasp_center_world_stage_units": approach_point.copy(),
                    "surface_clearance_m": clearance_m,
                    "target_axis_world": target_axis_world,
                    "target_closing_axis_world": inward,
                    "waypoints_world_stage_units": vertical_approach_waypoints(
                        approach_point,
                        meters_per_unit=meters_per_unit,
                        hover_height_mm=hover_height_mm,
                        descent_step_mm=descent_step_mm,
                    ),
                }
            )
    return candidates
