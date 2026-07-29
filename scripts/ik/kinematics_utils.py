"""Pure NumPy frame, unit, joint-limit, and FK helpers.

This module deliberately has no Isaac Sim or LeRobot imports.  Both runtimes
can therefore use exactly the same validation and coordinate conventions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import numpy as np


def finite_array(
    value: object,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """Return a float64 array with an exact shape and only finite values."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinity")
    return array


def positive_scale(value: float, *, label: str = "scale") -> float:
    """Validate a finite, strictly positive unit scale."""
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return scale


def stage_to_meters(value: object, meters_per_unit: float) -> np.ndarray:
    """Convert a position or displacement from USD stage units to meters."""
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("stage-unit value contains NaN or infinity")
    return array * positive_scale(meters_per_unit, label="meters_per_unit")


def meters_to_stage(value: object, meters_per_unit: float) -> np.ndarray:
    """Convert a position or displacement from meters to USD stage units."""
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("meter value contains NaN or infinity")
    return array / positive_scale(meters_per_unit, label="meters_per_unit")


def rotation_matrix_wxyz(quaternion: object) -> np.ndarray:
    """Convert an Isaac-order quaternion into a proper rotation matrix."""
    quaternion_array = finite_array(
        quaternion,
        shape=(4,),
        label="wxyz quaternion",
    )
    norm = float(np.linalg.norm(quaternion_array))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("wxyz quaternion norm must be nonzero")
    w, x, y, z = quaternion_array / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_rpy(rpy: object) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = finite_array(rpy, shape=(3,), label="rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64
    )
    rotation_y = np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64
    )
    rotation_z = np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64
    )
    return rotation_z @ rotation_y @ rotation_x


def require_rotation_matrix(value: object, *, label: str = "rotation") -> np.ndarray:
    """Return a finite proper 3x3 rotation or raise ValueError."""
    rotation = finite_array(value, shape=(3, 3), label=label)
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-8,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{label} must be a proper rotation matrix")
    return rotation


def make_transform(position: object, rotation: object) -> np.ndarray:
    """Construct a validated homogeneous parent-from-child transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = require_rotation_matrix(rotation)
    transform[:3, 3] = finite_array(position, shape=(3,), label="position")
    return transform


def require_transform(value: object, *, label: str = "transform") -> np.ndarray:
    """Return a finite rigid homogeneous transform or raise ValueError."""
    transform = finite_array(value, shape=(4, 4), label=label)
    require_rotation_matrix(transform[:3, :3], label=f"{label} rotation")
    if not np.allclose(
        transform[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{label} must have homogeneous bottom row [0, 0, 0, 1]")
    return transform


def transform_point(transform: object, point: object) -> np.ndarray:
    """Apply a parent-from-child transform to a three-dimensional point."""
    rigid_transform = require_transform(transform)
    point_array = finite_array(point, shape=(3,), label="point")
    return (rigid_transform @ np.append(point_array, 1.0))[:3]


def inverse_transform_point(transform: object, point: object) -> np.ndarray:
    """Apply the inverse of a parent-from-child transform to a point."""
    rigid_transform = require_transform(transform)
    point_array = finite_array(point, shape=(3,), label="point")
    return (np.linalg.inv(rigid_transform) @ np.append(point_array, 1.0))[:3]


def world_stage_point_in_frame_meters(
    point_world_stage_units: object,
    world_from_frame_meters: object,
    meters_per_unit: float,
) -> np.ndarray:
    """Express a USD world point in a metric local frame."""
    point_world_m = stage_to_meters(point_world_stage_units, meters_per_unit)
    return inverse_transform_point(world_from_frame_meters, point_world_m)


def frame_point_meters_in_world_stage(
    point_frame_meters: object,
    world_from_frame_meters: object,
    meters_per_unit: float,
) -> np.ndarray:
    """Express a metric local-frame point in USD world stage units."""
    point_world_m = transform_point(world_from_frame_meters, point_frame_meters)
    return meters_to_stage(point_world_m, meters_per_unit)


def resolve_named_indices(
    actual_names: Sequence[object],
    required_names: Sequence[str],
) -> list[int]:
    """Resolve a unique required joint order within a runtime joint list."""
    normalized_names = [str(name) for name in actual_names]
    duplicate_actual = sorted(
        {name for name in normalized_names if normalized_names.count(name) > 1}
    )
    if duplicate_actual:
        raise ValueError(f"Runtime joint names are not unique: {duplicate_actual}")
    if len(set(required_names)) != len(required_names):
        raise ValueError(f"Required joint names are not unique: {list(required_names)}")
    index_by_name = {name: index for index, name in enumerate(normalized_names)}
    missing = [name for name in required_names if name not in index_by_name]
    if missing:
        raise ValueError(
            f"Missing required joints {missing}; actual names={normalized_names}"
        )
    return [index_by_name[name] for name in required_names]


def joint_limit_violations(
    values: object,
    lower_limits: object,
    upper_limits: object,
    joint_names: Sequence[str],
    *,
    tolerance: float = 0.0,
) -> list[str]:
    """Return names of joints outside inclusive limits plus a tolerance."""
    shape = (len(joint_names),)
    joints = finite_array(values, shape=shape, label="joint values")
    lower = finite_array(lower_limits, shape=shape, label="lower joint limits")
    upper = finite_array(upper_limits, shape=shape, label="upper joint limits")
    if np.any(lower > upper):
        raise ValueError("lower joint limits exceed upper joint limits")
    limit_tolerance = float(tolerance)
    if not math.isfinite(limit_tolerance) or limit_tolerance < 0.0:
        raise ValueError("joint-limit tolerance must be finite and nonnegative")
    return [
        name
        for name, value, minimum, maximum in zip(
            joint_names,
            joints,
            lower,
            upper,
        )
        if value < minimum - limit_tolerance or value > maximum + limit_tolerance
    ]


def forward_kinematics_checked(
    forward_kinematics: Callable[[np.ndarray], object],
    joints: object,
    *,
    joint_count: int,
) -> np.ndarray:
    """Evaluate FK and validate its joint vector and rigid-pose result."""
    joint_values = finite_array(
        joints,
        shape=(joint_count,),
        label="FK joint vector",
    )
    pose = forward_kinematics(joint_values)
    return require_transform(pose, label="FK pose")


def rotation_error_rad(actual_rotation: object, desired_rotation: object) -> float:
    """Return the shortest SO(3) angular separation in radians."""
    actual = require_rotation_matrix(actual_rotation, label="actual rotation")
    desired = require_rotation_matrix(desired_rotation, label="desired rotation")
    relative_rotation = desired.T @ actual
    cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def axis_alignment_error_rad(actual_axis: object, desired_axis: object) -> float:
    """Return the unsigned angle between two nonzero three-dimensional axes."""
    actual = finite_array(actual_axis, shape=(3,), label="actual axis")
    desired = finite_array(desired_axis, shape=(3,), label="desired axis")
    actual_norm = float(np.linalg.norm(actual))
    desired_norm = float(np.linalg.norm(desired))
    if actual_norm <= np.finfo(np.float64).eps:
        raise ValueError("actual axis norm must be nonzero")
    if desired_norm <= np.finfo(np.float64).eps:
        raise ValueError("desired axis norm must be nonzero")
    cosine = float(
        np.dot(actual / actual_norm, desired / desired_norm)
    )
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def pose_residual(
    actual_pose: object,
    desired_pose: object,
) -> tuple[float, float]:
    """Return Cartesian position and SO(3) orientation residuals."""
    actual = require_transform(actual_pose, label="actual pose")
    desired = require_transform(desired_pose, label="desired pose")
    return (
        float(np.linalg.norm(actual[:3, 3] - desired[:3, 3])),
        rotation_error_rad(actual[:3, :3], desired[:3, :3]),
    )
