"""Pure geometry construction for progressively angled sphere grasps."""

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


def linear_approach_waypoints(
    final_target_world_stage_units: object,
    *,
    approach_axis_world: object,
    meters_per_unit: float,
    hover_height_mm: float,
    descent_step_mm: float,
) -> list[np.ndarray]:
    """Return hover-to-contact waypoints parallel to an approach axis.

    ``approach_axis_world`` points from hover toward the final target. The
    hover point is therefore offset in the opposite direction, which keeps a
    horizontal approach outside the sphere before it reaches the contact
    target.
    """
    final_target = finite_array(
        final_target_world_stage_units,
        shape=(3,),
        label="final approach contact target",
    )
    approach_axis = normalized_vector(
        approach_axis_world,
        label="approach axis",
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
    approach_offsets = np.linspace(
        hover_height_stage_units,
        0.0,
        num=segment_count + 1,
        dtype=np.float64,
    )
    return [
        final_target - approach_axis * offset
        for offset in approach_offsets
    ]


def vertical_approach_waypoints(
    final_target_world_stage_units: object,
    *,
    meters_per_unit: float,
    hover_height_mm: float,
    descent_step_mm: float,
) -> list[np.ndarray]:
    """Return vertical top-down waypoints via ``linear_approach_waypoints``."""
    return linear_approach_waypoints(
        final_target_world_stage_units,
        approach_axis_world=[0.0, 0.0, -1.0],
        meters_per_unit=meters_per_unit,
        hover_height_mm=hover_height_mm,
        descent_step_mm=descent_step_mm,
    )


def generate_sphere_grasp_candidates(
    sphere_center: object,
    sphere_radius: float,
    nominal_surface_point: object,
    *,
    meters_per_unit: float,
    surface_clearance_m: float,
    surface_offsets_deg: Sequence[float],
    hover_height_mm: float,
    descent_step_mm: float,
    approach_elevations_deg: Sequence[float] | None = None,
    approach_tilts_deg: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Sample sphere grasps over approach elevations above horizontal.

    ``90`` degrees is a vertical top-down approach and ``0`` degrees is a
    horizontal inward approach. ``approach_tilts_deg`` is retained temporarily
    for callers that still express the same orientation as a tilt away from
    vertical; callers must provide exactly one representation.
    """
    center = finite_array(sphere_center, shape=(3,), label="sphere center")
    radius = positive_scale(sphere_radius, label="sphere_radius")
    scale = positive_scale(meters_per_unit, label="meters_per_unit")
    clearance_m = float(surface_clearance_m)
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("surface_clearance_m must be finite and nonnegative")
    clearance_stage_units = clearance_m / scale
    if (
        approach_elevations_deg is None
        and approach_tilts_deg is None
    ) or (
        approach_elevations_deg is not None
        and approach_tilts_deg is not None
    ):
        raise ValueError(
            "Specify exactly one of approach_elevations_deg or "
            "approach_tilts_deg"
        )
    if approach_elevations_deg is not None:
        elevation_values = list(approach_elevations_deg)
    else:
        assert approach_tilts_deg is not None
        elevation_values = []
        for tilt_value in approach_tilts_deg:
            tilt_deg = float(tilt_value)
            if not math.isfinite(tilt_deg) or not 0.0 <= tilt_deg <= 90.0:
                raise ValueError("approach tilts must be finite values in [0, 90]")
            elevation_values.append(90.0 - tilt_deg)
    if not elevation_values:
        raise ValueError("at least one approach elevation is required")
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
        for approach_elevation_value in elevation_values:
            approach_elevation_deg = float(approach_elevation_value)
            if (
                not math.isfinite(approach_elevation_deg)
                or not 0.0 <= approach_elevation_deg <= 90.0
            ):
                raise ValueError(
                    "approach elevations must be finite values in [0, 90]"
                )
            elevation = math.radians(approach_elevation_deg)
            approach_tilt_deg = 90.0 - approach_elevation_deg
            target_axis_world = normalized_vector(
                math.sin(elevation) * np.array([0.0, 0.0, -1.0])
                + math.cos(elevation) * inward,
                label="candidate approach axis",
            )
            candidate_id = (
                f"surface_{surface_offset_deg:+.0f}_elevation_"
                f"{approach_elevation_deg:.0f}"
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "surface_offset_deg": surface_offset_deg,
                    "approach_elevation_deg": approach_elevation_deg,
                    "approach_tilt_deg": approach_tilt_deg,
                    "target_axis_world": target_axis_world,
                    "target_closing_axis_world": inward,
                    "waypoints_world_stage_units": linear_approach_waypoints(
                        approach_point,
                        approach_axis_world=target_axis_world,
                        meters_per_unit=meters_per_unit,
                        hover_height_mm=hover_height_mm,
                        descent_step_mm=descent_step_mm,
                    ),
                }
            )
    return candidates
