#!/usr/bin/env python3
"""Deterministic SO-101 scripted sphere grasp using exact simulation state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from isaacsim import SimulationApp


DEFAULT_WORLD = Path(__file__).resolve().parent / "worlds" / "blankworld.usd"
SPHERE_PRIM_PATH = "/Sphere"
ROBOT_PRIM_PATH = "/so101_follower"
TCP_LINK_PATH = f"{ROBOT_PRIM_PATH}/gripper"
FINGER_AXIS_LOCAL = (1.0, 0.0, 0.0)
VIRTUAL_GRASP_FRACTION = 0.8
PRE_GRASP_CLEARANCE_METERS = 0.10
LIFT_DISTANCE_METERS = 0.15
IK_MAX_STEPS = 300
IK_POSITION_TOLERANCE_METERS = 0.005
IK_ORIENTATION_TOLERANCE = 0.10
IK_DAMPING = 0.05
IK_MAX_JOINT_STEP = 0.05
GRIPPER_MOVE_STEPS = 90
GRIPPER_OPEN_PRESET_FRACTION = 1.0
GRIPPER_HOLD_STEPS = 30
POST_LIFT_HOLD_STEPS = 60
MIN_SUCCESSFUL_LIFT_METERS = 0.05
MAX_LIFT_COUPLING_ERROR_METERS = 0.03
MAX_TCP_SPHERE_DISTANCE_METERS = 0.08


@dataclass(frozen=True)
class CartesianTarget:
    name: str
    position: np.ndarray
    orientation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the deterministic SO-101 sphere grasp scene."
    )
    parser.add_argument(
        "--world",
        type=Path,
        default=DEFAULT_WORLD,
        help="USD world to load (default: sim/worlds/blankworld.usd)",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=120,
        help="Physics steps to run before reading scene state (default: 120)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Physics tensor device (default: cpu)",
    )
    parser.add_argument("--headless", action="store_true")
    args, _ = parser.parse_known_args()
    if args.settle_steps < 1:
        parser.error("--settle-steps must be at least 1")
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
    """Convert Warp, Torch, or NumPy-backed Isaac values to a NumPy array."""
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def require_prim(stage: object, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Required prim does not exist: {prim_path}")


def normalized_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def choose_unique_best(
    candidates: list[tuple[int, int, str]], description: str
) -> tuple[int, str]:
    if not candidates:
        raise RuntimeError(f"Could not find a {description} candidate")

    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    best = [candidate for candidate in candidates if candidate[0] == best_score]
    if len(best) != 1:
        names = [candidate[2] for candidate in best]
        raise RuntimeError(
            f"Ambiguous {description} candidates at score {best_score}: {names}"
        )
    _, index, name = best[0]
    return index, name


def resolve_gripper_dof(robot: Articulation) -> tuple[int, str]:
    exact_names = {
        "gripper",
        "gripper_joint",
        "jaw",
        "jaw_joint",
        "finger_joint",
    }
    candidates: list[tuple[int, int, str]] = []
    for index, name in enumerate(robot.dof_names):
        normalized = normalized_name(name)
        if normalized in exact_names:
            score = 100
        elif "gripper" in normalized or "jaw" in normalized:
            score = 80
        elif "finger" in normalized:
            score = 60
        else:
            continue
        candidates.append((score, index, name))

    index, name = choose_unique_best(candidates, "gripper DOF")
    print(
        f"[SELECT] gripper DOF: index={index} name={name!r} "
        f"path={robot.dof_paths[0][index]}"
    )
    return index, name


def link_index_for_path(robot: Articulation, path: str) -> int | None:
    for index, link_path in enumerate(robot.link_paths[0]):
        if str(link_path) == path:
            return index
    return None


def resolve_tcp_link(robot: Articulation) -> tuple[int, str, str]:
    """Require the fixed gripper body as TCP; the moving jaw is never valid."""
    index = link_index_for_path(robot, TCP_LINK_PATH)
    if index is None:
        raise RuntimeError(
            f"Required IK/TCP link {TCP_LINK_PATH!r} is not an articulation link; "
            f"available links are {[str(path) for path in robot.link_paths[0]]}"
        )
    name = robot.link_names[index]
    print(f"[SELECT] explicit IK/TCP link: index={index} name={name!r} path={TCP_LINK_PATH}")
    return index, name, TCP_LINK_PATH


def estimate_finger_length(stage: object, meters_per_unit: float) -> float:
    """Estimate fixed-finger reach from the TCP origin along local +X."""
    tcp_prim = stage.GetPrimAtPath(TCP_LINK_PATH)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    local_range = cache.ComputeLocalBound(tcp_prim).ComputeAlignedRange()
    minimum = np.asarray(local_range.GetMin(), dtype=np.float64)
    maximum = np.asarray(local_range.GetMax(), dtype=np.float64)
    finger_length = float(maximum[0])
    if not np.isfinite(finger_length) or finger_length <= 0.0:
        raise RuntimeError(
            f"Invalid fixed-finger +X reach from {TCP_LINK_PATH}: "
            f"bounds min={minimum.tolist()} max={maximum.tolist()}"
        )
    print(
        f"[FINGER] local geometry bounds: min={minimum.tolist()} "
        f"max={maximum.tolist()}"
    )
    print(
        f"[FINGER] estimated +X length={finger_length * meters_per_unit:.6f} m; "
        f"virtual grasp offset={VIRTUAL_GRASP_FRACTION * finger_length * meters_per_unit:.6f} m"
    )
    return finger_length


def sphere_world_radius(stage: object) -> float:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world_box = cache.ComputeWorldBound(stage.GetPrimAtPath(SPHERE_PRIM_PATH))
    size = np.asarray(world_box.ComputeAlignedRange().GetSize(), dtype=np.float64)
    radius = float(np.max(size) * 0.5)
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError(f"Invalid sphere world bounds {size.tolist()}")
    print(f"[SPHERE] world bounds size: {size.tolist()} radius={radius:.6f}")
    return radius


def top_down_orientation() -> np.ndarray:
    """Return wxyz orientation with the TCP local +X approach axis downward."""
    half_sqrt = np.sqrt(0.5)
    return np.array([half_sqrt, 0.0, half_sqrt, 0.0], dtype=np.float64)


def rotate_vector_wxyz(orientation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a local vector into world coordinates with a wxyz quaternion."""
    quaternion = np.asarray(orientation, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w = quaternion[0]
    xyz = quaternion[1:]
    return (
        vector * (w * w - np.dot(xyz, xyz))
        + 2.0 * xyz * np.dot(xyz, vector)
        + 2.0 * w * np.cross(xyz, vector)
    )


def build_grasp_targets(
    sphere_position: np.ndarray,
    sphere_radius: float,
    finger_length: float,
    meters_per_unit: float,
) -> list[CartesianTarget]:
    orientation = top_down_orientation()
    pre_grasp_clearance = PRE_GRASP_CLEARANCE_METERS / meters_per_unit
    lift_distance = LIFT_DISTANCE_METERS / meters_per_unit
    finger_offset_local = (
        np.asarray(FINGER_AXIS_LOCAL, dtype=np.float64)
        * VIRTUAL_GRASP_FRACTION
        * finger_length
    )
    finger_offset_world = rotate_vector_wxyz(orientation, finger_offset_local)
    # virtual_grasp_world = tcp_world + R_tcp_world @ finger_offset_local
    # Therefore the TCP target is the desired virtual point minus that offset.
    grasp = sphere_position - finger_offset_world
    pre_grasp = grasp + np.array(
        [0.0, 0.0, sphere_radius + pre_grasp_clearance], dtype=np.float64
    )
    lift = grasp + np.array([0.0, 0.0, lift_distance], dtype=np.float64)
    print(f"[PLAN] finger offset local: {finger_offset_local.tolist()}")
    print(f"[PLAN] finger offset world: {finger_offset_world.tolist()}")
    print(f"[PLAN] virtual grasp point at grasp: {(grasp + finger_offset_world).tolist()}")
    targets = [
        CartesianTarget("pre_grasp", pre_grasp, orientation),
        CartesianTarget("grasp", grasp, orientation),
        CartesianTarget("lift", lift, orientation),
    ]
    for target in targets:
        print(
            f"[PLAN] {target.name}: position={target.position.tolist()} "
            f"orientation_wxyz={target.orientation.tolist()}"
        )
    return targets


def quaternion_error(goal: np.ndarray, current: np.ndarray) -> np.ndarray:
    goal = goal / np.linalg.norm(goal)
    current = current / np.linalg.norm(current)
    cw, cx, cy, cz = current
    current_conjugate = np.array([cw, -cx, -cy, -cz])
    aw, ax, ay, az = goal
    bw, bx, by, bz = current_conjugate
    difference = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return difference[1:] * np.sign(difference[0] or 1.0)


def orientation_distance(goal: np.ndarray, current: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(goal, current)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def jacobian_layout(robot: Articulation, link_index: int) -> tuple[int, int]:
    link_rows, _, columns = robot.jacobian_matrix_shape
    if columns == robot.num_dofs and link_rows == len(robot.link_names) - 1:
        if link_index == 0:
            raise RuntimeError("The articulation root link cannot be used as the TCP")
        return link_index - 1, 0
    if columns == robot.num_dofs + 6 and link_rows == len(robot.link_names):
        # PhysX prepends six floating-root velocity columns. They are omitted
        # below so IK cannot translate or rotate the robot base.
        return link_index, 6
    raise RuntimeError(
        f"Unexpected Jacobian shape {robot.jacobian_matrix_shape} for "
        f"{len(robot.link_names)} links and {robot.num_dofs} DOFs"
    )


def drive_to_joint_positions(
    robot: Articulation,
    joint_positions: np.ndarray,
    dof_indices: list[int],
    max_steps: int = 240,
) -> None:
    for _ in range(max_steps):
        robot.set_dof_position_targets(
            joint_positions[dof_indices], dof_indices=dof_indices
        )
        SIMULATION_APP.update()
        current = as_numpy(robot.get_dof_positions())[0]
        if np.max(np.abs(current[dof_indices] - joint_positions[dof_indices])) < 0.01:
            return
    print("[IK] WARNING: timed out while restoring the IK seed state")


def gripper_targets_from_limits(
    gripper_dof_index: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> tuple[float, float]:
    closed = float(lower_limits[gripper_dof_index])
    upper = float(upper_limits[gripper_dof_index])
    if not np.isfinite(closed) or not np.isfinite(upper) or upper <= closed:
        raise RuntimeError(
            f"Invalid gripper limits: lower={closed} upper={upper}"
        )
    opened = closed + GRIPPER_OPEN_PRESET_FRACTION * (upper - closed)
    print(
        f"[GRIPPER] preset targets: open={opened:.6f} "
        f"({GRIPPER_OPEN_PRESET_FRACTION:.0%} of range) closed={closed:.6f}"
    )
    return opened, closed


def command_gripper(
    robot: Articulation,
    gripper_dof_index: int,
    target: float,
    label: str,
    steps: int = GRIPPER_MOVE_STEPS,
) -> float:
    print(f"[GRIPPER] {label}: commanding target={target:.6f} for {steps} steps")
    for _ in range(steps):
        robot.set_dof_position_targets([target], dof_indices=[gripper_dof_index])
        SIMULATION_APP.update()
    actual = float(as_numpy(robot.get_dof_positions())[0, gripper_dof_index])
    print(f"[GRIPPER] {label}: actual position={actual:.6f}")
    return actual


def hold_grasp_pose(
    robot: Articulation,
    arm_target: np.ndarray,
    arm_dof_indices: list[int],
    gripper_dof_index: int,
    gripper_target: float,
    steps: int,
) -> None:
    for _ in range(steps):
        robot.set_dof_position_targets(
            arm_target[arm_dof_indices], dof_indices=arm_dof_indices
        )
        robot.set_dof_position_targets(
            [gripper_target], dof_indices=[gripper_dof_index]
        )
        SIMULATION_APP.update()


def world_position(prim: XformPrim) -> np.ndarray:
    return as_numpy(prim.get_world_poses()[0])[0].copy()


def verify_lift(
    sphere_before_lift: np.ndarray,
    sphere_after_lift: np.ndarray,
    tcp_before_lift: np.ndarray,
    tcp_after_lift: np.ndarray,
    finger_offset_world: np.ndarray,
    meters_per_unit: float,
) -> bool:
    sphere_delta = (sphere_after_lift - sphere_before_lift) * meters_per_unit
    tcp_delta = (tcp_after_lift - tcp_before_lift) * meters_per_unit
    virtual_grasp_after_lift = tcp_after_lift + finger_offset_world
    grasp_sphere_distance = float(
        np.linalg.norm(sphere_after_lift - virtual_grasp_after_lift) * meters_per_unit
    )
    coupling_error = abs(float(sphere_delta[2] - tcp_delta[2]))

    rose_enough = float(sphere_delta[2]) >= MIN_SUCCESSFUL_LIFT_METERS
    moved_with_tcp = coupling_error <= MAX_LIFT_COUPLING_ERROR_METERS
    remains_near_grasp = grasp_sphere_distance <= MAX_TCP_SPHERE_DISTANCE_METERS
    success = rose_enough and moved_with_tcp and remains_near_grasp

    print(f"[VERIFY] sphere lift delta meters: {sphere_delta.tolist()}")
    print(f"[VERIFY] TCP lift delta meters: {tcp_delta.tolist()}")
    print(f"[VERIFY] vertical coupling error meters: {coupling_error:.6f}")
    print(
        "[VERIFY] final virtual-grasp/sphere distance meters: "
        f"{grasp_sphere_distance:.6f}"
    )
    print(
        f"[VERIFY] checks: rose_enough={rose_enough} "
        f"moved_with_tcp={moved_with_tcp} "
        f"remains_near_virtual_grasp={remains_near_grasp}"
    )
    print(f"[RESULT] grasp {'SUCCESS' if success else 'FAILURE'}")
    return success


def solve_cartesian_target(
    robot: Articulation,
    tcp: XformPrim,
    tcp_link_index: int,
    gripper_dof_index: int,
    target: CartesianTarget,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    meters_per_unit: float,
) -> tuple[bool, np.ndarray, str]:
    arm_dof_indices = [
        index for index in range(robot.num_dofs) if index != gripper_dof_index
    ]
    jacobian_row, joint_column_offset = jacobian_layout(robot, tcp_link_index)
    layout_name = "floating-base" if joint_column_offset else "fixed-base"
    print(
        f"[IK] {target.name}: Jacobian layout={layout_name} "
        f"link_row={jacobian_row} joint_column_offset={joint_column_offset}"
    )
    seed = as_numpy(robot.get_dof_positions())[0].copy()

    for orientation_mode, goal_orientation in (("top-down", target.orientation),):
        for step in range(1, IK_MAX_STEPS + 1):
            current_dofs = as_numpy(robot.get_dof_positions())[0]
            tcp_positions, tcp_orientations = tcp.get_world_poses()
            current_position = as_numpy(tcp_positions)[0]
            current_orientation = as_numpy(tcp_orientations)[0]
            position_error = target.position - current_position
            position_error_meters = position_error * meters_per_unit
            position_norm = float(np.linalg.norm(position_error_meters))
            angle_error = (
                orientation_distance(goal_orientation, current_orientation)
                if goal_orientation is not None
                else 0.0
            )

            if position_norm <= IK_POSITION_TOLERANCE_METERS and (
                goal_orientation is None
                or angle_error <= IK_ORIENTATION_TOLERANCE
            ):
                solved = as_numpy(robot.get_dof_positions())[0]
                print(
                    f"[IK] {target.name}: SUCCESS mode={orientation_mode} "
                    f"steps={step - 1} position_error={position_norm:.6f} "
                    f"orientation_error={angle_error:.6f}"
                )
                print(f"[IK] {target.name}: solved joints={solved.tolist()}")
                return True, solved, orientation_mode

            jacobian = as_numpy(robot.get_jacobian_matrices())[
                0, jacobian_row, :, :
            ][:, [joint_column_offset + index for index in arm_dof_indices]]
            jacobian[:3, :] *= meters_per_unit
            task_error = np.concatenate(
                [
                    position_error_meters,
                    0.35 * quaternion_error(goal_orientation, current_orientation),
                ]
            )
            task_jacobian = jacobian.copy()
            task_jacobian[3:, :] *= 0.35

            transpose = task_jacobian.T
            damping = np.eye(task_jacobian.shape[0]) * (IK_DAMPING**2)
            try:
                delta = transpose @ np.linalg.solve(
                    task_jacobian @ transpose + damping, task_error
                )
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(task_jacobian) @ task_error
            delta = np.clip(delta, -IK_MAX_JOINT_STEP, IK_MAX_JOINT_STEP)

            next_dofs = current_dofs.copy()
            next_dofs[arm_dof_indices] += delta
            next_dofs = np.clip(next_dofs, lower_limits, upper_limits)
            robot.set_dof_position_targets(
                next_dofs[arm_dof_indices], dof_indices=arm_dof_indices
            )
            SIMULATION_APP.update()

        print(
            f"[IK] {target.name}: FAILED mode={orientation_mode} "
            f"after {IK_MAX_STEPS} steps"
        )

    drive_to_joint_positions(robot, seed, arm_dof_indices)
    return False, seed, "failed"


def print_articulation_report(robot: Articulation) -> None:
    positions = as_numpy(robot.get_dof_positions())[0]
    lower, upper = robot.get_dof_limits()
    lower = as_numpy(lower)[0]
    upper = as_numpy(upper)[0]

    print("\n[SO101] Articulation inspection")
    print(f"[SO101] articulation prim: {ROBOT_PRIM_PATH}")
    print(f"[SO101] DOF count: {len(robot.dof_names)}")
    for index, name in enumerate(robot.dof_names):
        print(
            f"[SO101] DOF {index:02d}: name={name!r} "
            f"position={positions[index]: .6f} "
            f"limits=[{lower[index]: .6f}, {upper[index]: .6f}]"
        )

    print(f"[SO101] link count: {len(robot.link_names)}")
    link_paths = robot.link_paths[0]
    for index, (name, path) in enumerate(zip(robot.link_names, link_paths)):
        print(f"[SO101] LINK {index:02d}: name={name!r} path={path}")


def main() -> None:
    world_path = ARGS.world.expanduser().resolve()
    if not world_path.is_file():
        raise FileNotFoundError(f"World USD does not exist: {world_path}")

    print(f"[SCENE] Loading world: {world_path}")
    opened, stage = stage_utils.open_stage(str(world_path))
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open world: {world_path}")

    require_prim(stage, SPHERE_PRIM_PATH)
    require_prim(stage, ROBOT_PRIM_PATH)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit: {meters_per_unit}")
    print(f"[SCENE] Found sphere prim: {SPHERE_PRIM_PATH}")
    print(f"[SCENE] Found robot prim: {ROBOT_PRIM_PATH}")
    print(f"[SCENE] stage meters per unit: {meters_per_unit}")

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device=ARGS.device)
    sphere = XformPrim(SPHERE_PRIM_PATH)
    robot = Articulation(ROBOT_PRIM_PATH)

    app_utils.play()
    SIMULATION_APP.update()

    print(f"[SCENE] Settling physics for {ARGS.settle_steps} steps")
    completed_steps = 0
    while SIMULATION_APP.is_running() and completed_steps < ARGS.settle_steps:
        SIMULATION_APP.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            completed_steps += 1
    if completed_steps != ARGS.settle_steps:
        raise RuntimeError(
            f"Simulation stopped after {completed_steps}/{ARGS.settle_steps} settle steps"
        )

    sphere_positions, sphere_orientations = sphere.get_world_poses()
    sphere_position = as_numpy(sphere_positions)[0]
    sphere_orientation = as_numpy(sphere_orientations)[0]
    print(f"[SPHERE] world position after settle: {sphere_position.tolist()}")
    print(
        "[SPHERE] world orientation after settle (wxyz): "
        f"{sphere_orientation.tolist()}"
    )

    print_articulation_report(robot)
    gripper_dof_index, _ = resolve_gripper_dof(robot)
    tcp_link_index, tcp_link_name, tcp_link_path = resolve_tcp_link(robot)
    tcp = XformPrim(tcp_link_path)
    tcp_positions, tcp_orientations = tcp.get_world_poses()
    print(f"[TCP] selected link: {tcp_link_name!r}")
    print(f"[TCP] world position: {as_numpy(tcp_positions)[0].tolist()}")
    print(
        "[TCP] world orientation (wxyz): "
        f"{as_numpy(tcp_orientations)[0].tolist()}"
    )

    sphere_radius = sphere_world_radius(stage)
    finger_length = estimate_finger_length(stage, meters_per_unit)
    finger_offset_world = rotate_vector_wxyz(
        top_down_orientation(),
        np.asarray(FINGER_AXIS_LOCAL, dtype=np.float64)
        * VIRTUAL_GRASP_FRACTION
        * finger_length,
    )
    targets = build_grasp_targets(
        sphere_position, sphere_radius, finger_length, meters_per_unit
    )
    lower_limits, upper_limits = robot.get_dof_limits()
    lower_limits = as_numpy(lower_limits)[0]
    upper_limits = as_numpy(upper_limits)[0]
    arm_dof_indices = [
        index for index in range(robot.num_dofs) if index != gripper_dof_index
    ]
    open_target, closed_target = gripper_targets_from_limits(
        gripper_dof_index, lower_limits, upper_limits
    )

    command_gripper(robot, gripper_dof_index, open_target, "open")
    sphere_before_grasp = world_position(sphere)
    print(f"[SPHERE] position before grasp: {sphere_before_grasp.tolist()}")

    solved, _, _ = solve_cartesian_target(
        robot,
        tcp,
        tcp_link_index,
        gripper_dof_index,
        targets[0],
        lower_limits,
        upper_limits,
        meters_per_unit,
    )
    if not solved:
        print("[RESULT] grasp FAILURE: pre_grasp IK failed")
        raise RuntimeError("pre_grasp IK failed")

    solved, grasp_joints, _ = solve_cartesian_target(
        robot,
        tcp,
        tcp_link_index,
        gripper_dof_index,
        targets[1],
        lower_limits,
        upper_limits,
        meters_per_unit,
    )
    if not solved:
        print("[RESULT] grasp FAILURE: grasp IK failed")
        raise RuntimeError("grasp IK failed")

    command_gripper(robot, gripper_dof_index, closed_target, "close")
    hold_grasp_pose(
        robot,
        grasp_joints,
        arm_dof_indices,
        gripper_dof_index,
        closed_target,
        GRIPPER_HOLD_STEPS,
    )
    sphere_before_lift = world_position(sphere)
    tcp_before_lift = world_position(tcp)
    print(f"[SPHERE] position after close: {sphere_before_lift.tolist()}")

    solved, lift_joints, _ = solve_cartesian_target(
        robot,
        tcp,
        tcp_link_index,
        gripper_dof_index,
        targets[2],
        lower_limits,
        upper_limits,
        meters_per_unit,
    )
    if not solved:
        print("[RESULT] grasp FAILURE: lift IK failed")
        raise RuntimeError("lift IK failed")

    hold_grasp_pose(
        robot,
        lift_joints,
        arm_dof_indices,
        gripper_dof_index,
        closed_target,
        POST_LIFT_HOLD_STEPS,
    )
    sphere_after_lift = world_position(sphere)
    tcp_after_lift = world_position(tcp)
    print(f"[SPHERE] position after lift: {sphere_after_lift.tolist()}")

    if not verify_lift(
        sphere_before_lift,
        sphere_after_lift,
        tcp_before_lift,
        tcp_after_lift,
        finger_offset_world,
        meters_per_unit,
    ):
        raise RuntimeError("Sphere did not rise with the gripper")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SCENE] Interrupted")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        SIMULATION_APP.close()
