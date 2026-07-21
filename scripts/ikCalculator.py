#!/usr/bin/env python3
"""Align an SO-101 and select a perpendicular-axis sphere grasp point."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PROJECT_ROOT / "sim" / "worlds" / "blankworld.usd"

ROBOT_PRIM_PATH = "/so101_follower"
BASE_LINK_PRIM_PATH = f"{ROBOT_PRIM_PATH}/base"
BASE_REFERENCE_PRIM_PATH = (
    f"{BASE_LINK_PRIM_PATH}/visuals/base_motor_holder_so101_v1"
)
FIXED_FINGER_PRIM_PATH = f"{ROBOT_PRIM_PATH}/gripper"
SPHERE_PRIM_PATH = "/Sphere"
SHOULDER_PAN_DOF_NAME = "shoulder_pan"
BASE_ALIGNMENT_STEPS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align the SO-101 with the sphere, construct the perpendicular "
            "horizontal grasp axis, and select its point nearest the finger."
        )
    )
    parser.add_argument(
        "--world",
        type=Path,
        default=DEFAULT_WORLD,
        help="USD scene to load (default: sim/worlds/blankworld.usd)",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=120,
        help="Physics steps before measuring the scene (default: 120)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without opening a window",
    )
    args, _ = parser.parse_known_args()
    if args.settle_steps < 0:
        parser.error("--settle-steps cannot be negative")
    return args


ARGS = parse_args()
SIMULATION_APP = SimulationApp({"headless": ARGS.headless})


import numpy as np  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def as_numpy(value: object) -> np.ndarray:
    """Convert an Isaac tensor to a NumPy array."""
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def require_prim(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Required prim does not exist: {prim_path}")
    return prim


def visual_mesh_points_in_gripper_frame(
    gripper_prim: Usd.Prim,
) -> np.ndarray:
    """Return fixed-gripper visual vertices in the gripper's local frame."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    gripper_to_world = xform_cache.GetLocalToWorldTransform(gripper_prim)
    world_to_gripper = gripper_to_world.GetInverse()

    points_in_gripper: list[np.ndarray] = []
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    for prim in Usd.PrimRange(gripper_prim, predicate):
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
            continue

        mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not mesh_points:
            continue

        points = np.asarray(mesh_points, dtype=np.float64)
        mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
        mesh_to_gripper = np.asarray(
            mesh_to_world * world_to_gripper,
            dtype=np.float64,
        )
        homogeneous = np.column_stack((points, np.ones(len(points))))
        transformed = homogeneous @ mesh_to_gripper
        points_in_gripper.append(transformed[:, :3])

    if not points_in_gripper:
        raise RuntimeError(
            f"No visual mesh points found under {FIXED_FINGER_PRIM_PATH}"
        )

    return np.concatenate(points_in_gripper, axis=0)


def find_fixed_finger_endpoint_local(gripper_prim: Usd.Prim) -> np.ndarray:
    """Find the furthest fingertip face and return it in stage units."""
    points = visual_mesh_points_in_gripper_frame(gripper_prim)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    # The fixed finger is the longest extension from the gripper-frame origin.
    candidates = [
        (abs(float(minimum[axis])), axis, -1)
        for axis in range(3)
    ] + [
        (abs(float(maximum[axis])), axis, 1)
        for axis in range(3)
    ]
    _, tip_axis, tip_sign = max(candidates)
    tip_coordinate = (
        minimum[tip_axis] if tip_sign < 0 else maximum[tip_axis]
    )

    # Average the cross-section close to the extremity instead of selecting an
    # arbitrary mesh vertex on one edge of the fingertip.
    finger_length = maximum[tip_axis] - minimum[tip_axis]
    slice_depth = max(0.001, 0.05 * float(finger_length))
    if tip_sign < 0:
        tip_slice = points[points[:, tip_axis] <= tip_coordinate + slice_depth]
    else:
        tip_slice = points[points[:, tip_axis] >= tip_coordinate - slice_depth]

    endpoint = 0.5 * (tip_slice.min(axis=0) + tip_slice.max(axis=0))
    endpoint[tip_axis] = tip_coordinate
    return endpoint


def rotate_vector_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a quaternion stored in Isaac's wxyz order."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w = quaternion[0]
    xyz = quaternion[1:]
    return (
        vector * (w * w - np.dot(xyz, xyz))
        + 2.0 * xyz * np.dot(xyz, vector)
        + 2.0 * w * np.cross(xyz, vector)
    )


def align_base_with_sphere(
    robot: Articulation,
    base_reference: XformPrim,
    base_link: XformPrim,
    sphere_center: np.ndarray,
    steps: int = BASE_ALIGNMENT_STEPS,
) -> float:
    """Pan the SO-101 shoulder so its horizontal heading faces the sphere."""
    matches = [
        index
        for index, name in enumerate(robot.dof_names)
        if name == SHOULDER_PAN_DOF_NAME
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {SHOULDER_PAN_DOF_NAME!r} DOF, found {matches} in "
            f"{robot.dof_names}"
        )
    shoulder_pan_index = matches[0]

    base_positions, _ = base_reference.get_world_poses()
    _, base_orientations = base_link.get_world_poses()
    base_position = as_numpy(base_positions)[0]
    base_orientation = as_numpy(base_orientations)[0]

    direction_world = np.asarray(sphere_center, dtype=np.float64) - base_position
    direction_world[2] = 0.0
    if np.linalg.norm(direction_world[:2]) <= 1e-12:
        raise RuntimeError(
            "The sphere center is directly above the base reference; a "
            "horizontal alignment angle is not defined."
        )

    # Express the target direction in the rigid base frame. At zero shoulder
    # pan, a standard SO-101 points along local -Y.
    inverse_base_orientation = base_orientation.copy()
    inverse_base_orientation[1:] *= -1.0
    direction_local = rotate_vector_wxyz(
        inverse_base_orientation,
        direction_world,
    )
    # The imported SO-101 shoulder joint uses the opposite sign from the
    # conventional positive rotation about the base frame's +Z axis.
    desired_angle = -float(np.arctan2(direction_local[0], -direction_local[1]))

    lower_limits, upper_limits = robot.get_dof_limits()
    lower = float(as_numpy(lower_limits)[0, shoulder_pan_index])
    upper = float(as_numpy(upper_limits)[0, shoulder_pan_index])
    if not lower <= desired_angle <= upper:
        raise RuntimeError(
            "Sphere requires a shoulder-pan angle outside the robot limits: "
            f"target={np.degrees(desired_angle):.3f} deg, "
            f"limits=[{np.degrees(lower):.3f}, {np.degrees(upper):.3f}] deg"
        )

    for _ in range(steps):
        robot.set_dof_position_targets(
            [desired_angle],
            dof_indices=[shoulder_pan_index],
        )
        SIMULATION_APP.update()

    actual_angle = float(
        as_numpy(robot.get_dof_positions())[0, shoulder_pan_index]
    )
    print(
        "Base alignment (shoulder_pan): "
        f"target={np.degrees(desired_angle):.3f} deg, "
        f"actual={np.degrees(actual_angle):.3f} deg"
    )
    return actual_angle


def sphere_horizontal_radius(sphere_prim: Usd.Prim) -> float:
    """Return the sphere's horizontal radius in stage units."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    size = np.asarray(
        cache.ComputeWorldBound(sphere_prim).ComputeAlignedRange().GetSize(),
        dtype=np.float64,
    )
    radii = 0.5 * size[:2]
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise RuntimeError(f"Invalid sphere world bounds: {size.tolist()}")
    if not np.isclose(radii[0], radii[1], rtol=1e-3, atol=1e-6):
        raise RuntimeError(
            "The target prim is not horizontally spherical: "
            f"x radius={radii[0]}, y radius={radii[1]}"
        )
    return float(np.mean(radii))


def base_heading_world(
    base_orientation: np.ndarray,
    shoulder_pan_angle: float,
) -> np.ndarray:
    """Return the aligned arm's horizontal forward axis in world coordinates."""
    # A zero-pan SO-101 points along local -Y. Its imported shoulder joint has
    # the opposite sign from a conventional rotation about local +Z.
    heading_local = np.array(
        [
            -np.sin(shoulder_pan_angle),
            -np.cos(shoulder_pan_angle),
            0.0,
        ],
        dtype=np.float64,
    )
    heading_world = rotate_vector_wxyz(base_orientation, heading_local)
    heading_world[2] = 0.0
    horizontal_length = float(np.linalg.norm(heading_world[:2]))
    if horizontal_length <= 1e-12:
        raise RuntimeError("The aligned base heading has no horizontal component")
    return heading_world / horizontal_length


def perpendicular_sphere_grasp_points(
    sphere_center: np.ndarray,
    sphere_radius: float,
    base_heading: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect the sphere with its horizontal axis normal to the base."""
    perpendicular = np.array(
        [-base_heading[1], base_heading[0], 0.0],
        dtype=np.float64,
    )
    perpendicular /= np.linalg.norm(perpendicular)
    return (
        sphere_center + sphere_radius * perpendicular,
        sphere_center - sphere_radius * perpendicular,
    )


def choose_grasp_point_nearest_finger(
    grasp_points: tuple[np.ndarray, np.ndarray],
    finger_endpoint: np.ndarray,
) -> np.ndarray:
    """Choose the perpendicular-axis sphere point closest to the fingertip."""
    distances = [
        float(np.linalg.norm(point - finger_endpoint))
        for point in grasp_points
    ]
    return grasp_points[int(np.argmin(distances))]


def print_point(label: str, point_meters: np.ndarray) -> None:
    print(
        f"{label}: "
        f"x={point_meters[0]:.6f} m, "
        f"y={point_meters[1]:.6f} m, "
        f"z={point_meters[2]:.6f} m"
    )


def main() -> None:
    world_path = ARGS.world.expanduser().resolve()
    if not world_path.is_file():
        raise FileNotFoundError(f"World USD does not exist: {world_path}")

    opened, stage = stage_utils.open_stage(str(world_path))
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open world: {world_path}")

    gripper_prim = require_prim(stage, FIXED_FINGER_PRIM_PATH)
    require_prim(stage, BASE_LINK_PRIM_PATH)
    require_prim(stage, BASE_REFERENCE_PRIM_PATH)
    sphere_prim = require_prim(stage, SPHERE_PRIM_PATH)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit: {meters_per_unit}")

    endpoint_local_stage_units = find_fixed_finger_endpoint_local(gripper_prim)

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    robot = Articulation(ROBOT_PRIM_PATH)
    base_link = XformPrim(BASE_LINK_PRIM_PATH)
    base_reference = XformPrim(BASE_REFERENCE_PRIM_PATH)
    gripper = XformPrim(FIXED_FINGER_PRIM_PATH)
    sphere = XformPrim(SPHERE_PRIM_PATH)

    app_utils.play()
    SIMULATION_APP.update()
    completed_steps = 0
    while SIMULATION_APP.is_running() and completed_steps < ARGS.settle_steps:
        SIMULATION_APP.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            completed_steps += 1
    if completed_steps != ARGS.settle_steps:
        raise RuntimeError(
            f"Simulation stopped after {completed_steps}/{ARGS.settle_steps} steps"
        )

    sphere_positions, _ = sphere.get_world_poses()
    sphere_center = as_numpy(sphere_positions)[0]
    shoulder_pan_angle = align_base_with_sphere(
        robot,
        base_reference,
        base_link,
        sphere_center,
    )

    # Re-read all live poses because the alignment command advances physics.
    gripper_positions, gripper_orientations = gripper.get_world_poses()
    gripper_position = as_numpy(gripper_positions)[0]
    gripper_orientation = as_numpy(gripper_orientations)[0]
    finger_endpoint = gripper_position + rotate_vector_wxyz(
        gripper_orientation,
        endpoint_local_stage_units,
    )

    sphere_positions, _ = sphere.get_world_poses()
    sphere_center = as_numpy(sphere_positions)[0]
    sphere_radius = sphere_horizontal_radius(sphere_prim)
    _, base_orientations = base_link.get_world_poses()
    base_orientation = as_numpy(base_orientations)[0]
    heading = base_heading_world(base_orientation, shoulder_pan_angle)
    sphere_grasp_points = perpendicular_sphere_grasp_points(
        sphere_center,
        sphere_radius,
        heading,
    )
    sphere_grasp_point = choose_grasp_point_nearest_finger(
        sphere_grasp_points,
        finger_endpoint,
    )

    print()
    print(
        "Aligned base heading (world horizontal unit vector): "
        f"x={heading[0]:.6f}, y={heading[1]:.6f}, z={heading[2]:.6f}"
    )
    print_point("Fixed finger endpoint (world)", finger_endpoint * meters_per_unit)
    print_point(
        "Sphere perpendicular grasp point A (world)",
        sphere_grasp_points[0] * meters_per_unit,
    )
    print_point(
        "Sphere perpendicular grasp point B (world)",
        sphere_grasp_points[1] * meters_per_unit,
    )
    print_point(
        "Selected grasp point nearest fixed finger (world)",
        sphere_grasp_point * meters_per_unit,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        SIMULATION_APP.close()
