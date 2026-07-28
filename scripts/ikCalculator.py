#!/usr/bin/env python3
"""Align an SO-101, grasp a sphere, and lift it after grasp stabilization."""

from __future__ import annotations

import argparse
import builtins
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PROJECT_ROOT / "sim" / "worlds" / "blankworld.usd"
SO101_URDF_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.urdf"
)
DEFAULT_LEROBOT_PYTHON = Path(
    "/home/rog/miniconda3/envs/so101_ik/bin/python"
)
LEROBOT_TARGET_FRAME = "gripper_frame_link"
LEROBOT_FINGERTIP_FRAME = "ik_fixed_fingertip_link"
FIXED_TOOL_JOINT_NAME = "gripper_frame_joint"
FIXED_TOOL_PARENT_LINK = "gripper_link"
FIXED_TOOL_LINK = "gripper_frame_link"
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
LEROBOT_WORKER_MARKER = "LEROBOT_WORKER_RESULT="

ROBOT_PRIM_PATH = "/so101_follower"
BASE_LINK_PRIM_PATH = f"{ROBOT_PRIM_PATH}/base"
BASE_REFERENCE_PRIM_PATH = (
    f"{BASE_LINK_PRIM_PATH}/visuals/base_motor_holder_so101_v1"
)
FIXED_FINGER_PRIM_PATH = f"{ROBOT_PRIM_PATH}/gripper"
LEGACY_CONTACT_POINT_PRIM_PATH = f"{FIXED_FINGER_PRIM_PATH}/contactPoint"
MOVING_JAW_PRIM_PATH = f"{ROBOT_PRIM_PATH}/jaw"
SPHERE_PRIM_PATH = "/Sphere"
SHOULDER_PAN_DOF_NAME = "shoulder_pan"
WRIST_PRESET_DOF_NAME = "wrist_roll"
GRIPPER_PRESET_DOF_NAME = "gripper"
BASE_ALIGNMENT_STEPS = 300
PRESET_COMMAND_STEPS = 300
WRIST_PRESET_POSITION_RAD = -1.55
# Pre-shape the revolute jaw around the 50 mm test sphere. At 1.5 rad the
# moving fingertip is retracted roughly 63 mm above the fixed fingertip and
# cannot oppose it during closure; near 0.55 rad both tips share the grasp
# plane while retaining slightly more than one sphere diameter of clearance.
GRIPPER_PRESET_POSITION_RAD = 0.55
GRIPPER_DOF_INDEX = 5
GRIPPER_CLOSE_PROGRESS_INTERVAL_STEPS = 30
GRIPPER_MAX_CLOSE_STEPS = 600
GRASP_STABLE_CONTACT_FRAMES = 1
GRASP_STABLE_VELOCITY_THRESHOLD_RAD_S = 0.10
GRIPPER_CONTACT_SQUEEZE_DELTA_RAD = 0.08
GRIPPER_LIFT_SQUEEZE_DELTA_RAD = 0.08
GRIPPER_STIFFNESS_MULTIPLIER = 4.0
GRASP_STATIC_FRICTION = 1.5
GRASP_DYNAMIC_FRICTION = 1.5
GRIPPER_LOWER_LIMIT_TOLERANCE_RAD = 0.01
DEFAULT_LIFT_HEIGHT_MM = 100.0
DEFAULT_LIFT_STOP_DISTANCE_MM = 7.0
LIFT_COMMAND_STEPS = 600
LIFT_CONTACT_LOSS_TOLERANCE_FRAMES = 10
IK_MAX_ITERATIONS = 100
IK_POSITION_TOLERANCE_M = 0.0005
DEFAULT_TOP_DOWN_POSITION_TOLERANCE_MM = 2.0
DEFAULT_TOP_DOWN_AXIS_TOLERANCE_DEG = 10.0
DEFAULT_CLOSING_AXIS_TOLERANCE_DEG = 20.0
IK_SEED_REGULARIZATION_WEIGHT = 1e-4
IK_CLOSING_AXIS_WEIGHT = 0.5
DEFAULT_TOP_DOWN_HOVER_HEIGHT_MM = 70.0
DEFAULT_TOP_DOWN_DESCENT_STEP_MM = 10.0
DEFAULT_TOP_DOWN_YAW_OFFSETS_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
GRASP_SURFACE_OFFSETS_DEG = (
    0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0,
    120.0, -120.0, 150.0, -150.0, 180.0,
)
GRASP_APPROACH_TILTS_DEG = (0.0, 15.0)
IK_COMMAND_STEPS = 300
APPROACH_TRACKING_SETTLE_STEPS = 300
APPROACH_TRACKING_STABLE_FRAMES = 10
APPROACH_JOINT_TRACKING_TOLERANCE_RAD = math.radians(1.0)
APPROACH_TCP_TRACKING_TOLERANCE_M = 0.003
APPROACH_SOLVER_POSITION_ITERATIONS = 64
APPROACH_SOLVER_VELOCITY_ITERATIONS = 16
DEFAULT_ARM_STIFFNESS_MULTIPLIER = 4.0
DEFAULT_ARM_DAMPING_MULTIPLIER = 0.1
DEBUG_POINT_WAIT_STEPS = 45
DEBUG_IK_AXIS_LENGTH_MM = 50.0
DEBUG_IK_CANDIDATE_AXIS_LENGTH_MM = 35.0
# Keep the URDF fixed-finger tool point outside the object until jaw closure.
DEFAULT_SPHERE_SURFACE_CLEARANCE_MM = 5.0
TOOL_MODEL_POSITION_TOLERANCE_M = 0.001
# Isaac's imported wrist-roll zero omits the URDF's 0.0486795 rad calibration
# rotation, producing a fixed 2.789-degree frame discrepancy at the tool.
TOOL_MODEL_ORIENTATION_TOLERANCE_RAD = math.radians(3.0)


def lerobot_worker_main() -> int:
    """Run LeRobot/Placo requests without importing Isaac Sim."""
    import numpy as worker_np

    from ik.kinematics_utils import (
        axis_alignment_error_rad,
        finite_array,
        forward_kinematics_checked,
        joint_limit_violations,
    )
    import placo
    from lerobot.model.kinematics import RobotKinematics
    from ik.candidate_planning import evaluate_first_valid_path

    def request_array(
        name: str,
        shape: tuple[int, ...],
    ) -> worker_np.ndarray:
        return finite_array(request[name], shape=shape, label=name)

    def checked_fk(
        kinematics: RobotKinematics,
        joints_deg: worker_np.ndarray,
    ) -> worker_np.ndarray:
        return forward_kinematics_checked(
            kinematics.forward_kinematics,
            joints_deg,
            joint_count=len(ARM_JOINT_NAMES),
        )

    def make_fingertip_kinematics(
        fingertip_offset_m: worker_np.ndarray,
        temp_dir: str,
    ) -> RobotKinematics:
        """Create a kinematics model whose tip is Isaac's fixed contact point."""
        urdf_tree = ET.parse(SO101_URDF_PATH)
        urdf_root = urdf_tree.getroot()
        for mesh in urdf_root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.set(
                    "filename",
                    str((SO101_URDF_PATH.parent / filename).resolve()),
                )

        ET.SubElement(urdf_root, "link", {"name": LEROBOT_FINGERTIP_FRAME})
        fingertip_joint = ET.SubElement(
            urdf_root,
            "joint",
            {"name": "ik_fixed_fingertip_joint", "type": "fixed"},
        )
        ET.SubElement(fingertip_joint, "parent", {"link": "gripper_link"})
        ET.SubElement(
            fingertip_joint,
            "child",
            {"link": LEROBOT_FINGERTIP_FRAME},
        )
        ET.SubElement(
            fingertip_joint,
            "origin",
            {
                "xyz": " ".join(
                    f"{value:.17g}" for value in fingertip_offset_m
                ),
                "rpy": "0 0 0",
            },
        )
        fingertip_urdf = Path(temp_dir) / "so101_fingertip.urdf"
        urdf_tree.write(fingertip_urdf, encoding="utf-8", xml_declaration=True)
        return RobotKinematics(
            urdf_path=str(fingertip_urdf),
            target_frame_name=LEROBOT_FINGERTIP_FRAME,
            joint_names=ARM_JOINT_NAMES,
        )

    def solve_position_only(
        kinematics: RobotKinematics,
        seed_joints_deg: worker_np.ndarray,
        target_position_m: worker_np.ndarray,
        max_iterations: int,
        tolerance_m: float | None,
    ) -> dict[str, object]:
        """Solve a fingertip position target using the existing IK behavior."""
        joints_deg = seed_joints_deg.copy()
        target_pose = worker_np.asarray(
            checked_fk(kinematics, joints_deg),
            dtype=worker_np.float64,
        )
        target_pose[:3, 3] = target_position_m
        solved_pose = target_pose.copy()
        position_error_m = float("inf")
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            joints_deg = worker_np.asarray(
                kinematics.inverse_kinematics(
                    current_joint_pos=joints_deg,
                    desired_ee_pose=target_pose,
                    position_weight=1.0,
                    orientation_weight=0.0,
                ),
                dtype=worker_np.float64,
            )
            if not worker_np.all(worker_np.isfinite(joints_deg)):
                raise RuntimeError("Position IK returned NaN or infinity")
            solved_pose = worker_np.asarray(
                checked_fk(kinematics, joints_deg),
                dtype=worker_np.float64,
            )
            position_error_m = float(
                worker_np.linalg.norm(solved_pose[:3, 3] - target_position_m)
            )
            if tolerance_m is not None and position_error_m <= tolerance_m:
                break
        if tolerance_m is None:
            status = "best_effort"
        elif position_error_m <= tolerance_m:
            status = "converged"
        else:
            status = "not_converged"
        return {
            "status": status,
            "seed_joints_deg": seed_joints_deg.tolist(),
            "solved_joints_deg": joints_deg.tolist(),
            "target_position_m": target_position_m.tolist(),
            "solved_position_m": solved_pose[:3, 3].tolist(),
            "position_error_m": position_error_m,
            "iterations": iterations,
            "position_weight": 1.0,
            "orientation_weight": 0.0,
        }

    class PositionAxisSolverContext:
        """One resettable PlaCo solver shared by all candidates/waypoints."""

        def __init__(
            self,
            tool_axis_frame: worker_np.ndarray,
            tool_closing_axis_frame: worker_np.ndarray,
            seed_regularization_weight: float,
        ) -> None:
            self.robot = placo.RobotWrapper(str(SO101_URDF_PATH))
            self.solver = placo.KinematicsSolver(self.robot)
            self.solver.mask_fbase(True)
            self.position_task = self.solver.add_position_task(
                LEROBOT_TARGET_FRAME,
                worker_np.zeros(3, dtype=worker_np.float64),
            )
            self.position_task.configure("tcp_position", "hard")
            self.axis_task = self.solver.add_axisalign_task(
                LEROBOT_TARGET_FRAME,
                tool_axis_frame,
                worker_np.array([0.0, 0.0, 1.0]),
            )
            self.axis_task.configure("tool_approach_axis", "soft", 1.0)
            self.closing_task = self.solver.add_axisalign_task(
                LEROBOT_TARGET_FRAME,
                tool_closing_axis_frame,
                worker_np.array([1.0, 0.0, 0.0]),
            )
            self.seed_task = self.solver.add_joints_task()
            self.seed_regularization_weight = seed_regularization_weight
            self.context_builds = 1
            self.waypoint_solve_calls = 0
            self.solver_iterations = 0
            self.state_resets = 0
            self.last_solve_diagnostics: dict[str, object] = {}

        def reset(
            self,
            seed_joints_deg: worker_np.ndarray,
            target_position_m: worker_np.ndarray,
            target_axis_world: worker_np.ndarray,
            target_closing_axis_world: worker_np.ndarray,
            closing_axis_weight: float,
        ) -> None:
            for joint_name, joint_rad in zip(
                ARM_JOINT_NAMES,
                worker_np.deg2rad(seed_joints_deg),
            ):
                self.robot.set_joint(joint_name, float(joint_rad))
            self.robot.update_kinematics()
            self.position_task.target_world = target_position_m
            self.axis_task.targetAxis_world = target_axis_world
            self.closing_task.targetAxis_world = target_closing_axis_world
            self.closing_task.configure(
                "jaw_closing_axis",
                "soft",
                max(float(closing_axis_weight), 1e-12),
            )
            self.seed_task.set_joints(
                {
                    joint_name: float(joint_rad)
                    for joint_name, joint_rad in zip(
                        ARM_JOINT_NAMES,
                        worker_np.deg2rad(seed_joints_deg),
                    )
                }
            )
            self.seed_task.configure(
                "seed_regularization",
                "soft",
                self.seed_regularization_weight,
            )
            self.waypoint_solve_calls += 1
            self.state_resets += 1
            self.last_solve_diagnostics = {}

    def solve_position_axis_waypoint(
        seed_joints_deg: worker_np.ndarray,
        target_position_m: worker_np.ndarray,
        tool_axis_frame: worker_np.ndarray,
        target_axis_world: worker_np.ndarray,
        max_iterations: int,
        position_tolerance_m: float,
        axis_tolerance_rad: float,
        seed_regularization_weight: float,
        lower_joint_limits_deg: worker_np.ndarray,
        upper_joint_limits_deg: worker_np.ndarray,
        tool_closing_axis_frame: worker_np.ndarray | None = None,
        target_closing_axis_world: worker_np.ndarray | None = None,
        closing_axis_weight: float = 0.0,
        closing_axis_tolerance_rad: float = math.pi,
        solver_context: PositionAxisSolverContext | None = None,
    ) -> tuple[dict[str, object] | None, str | None]:
        """Solve hard position with soft approach and jaw-axis guidance."""
        if tool_closing_axis_frame is None:
            tool_closing_axis_frame = worker_np.array(
                [1.0, 0.0, 0.0], dtype=worker_np.float64
            )
        if target_closing_axis_world is None:
            target_closing_axis_world = worker_np.array(
                [1.0, 0.0, 0.0], dtype=worker_np.float64
            )
        context = solver_context or PositionAxisSolverContext(
            tool_axis_frame,
            tool_closing_axis_frame,
            seed_regularization_weight,
        )
        context.reset(
            seed_joints_deg,
            target_position_m,
            target_axis_world,
            target_closing_axis_world,
            closing_axis_weight,
        )
        robot = context.robot
        solver = context.solver

        best_state: dict[str, object] | None = None
        rejection_reason = "Position-axis IK produced no finite in-limit state"
        last_joint_limit_failures: list[str] = []
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            try:
                solver.solve(True)
                context.solver_iterations += 1
            except RuntimeError as error:
                rejection_reason = f"Position-axis QP failed: {error}"
                break
            robot.update_kinematics()
            joints_deg = worker_np.rad2deg(
                worker_np.asarray(
                    [robot.get_joint(name) for name in ARM_JOINT_NAMES],
                    dtype=worker_np.float64,
                )
            )
            if not worker_np.all(worker_np.isfinite(joints_deg)):
                rejection_reason = "Position-axis IK returned NaN or infinity"
                break
            violations = joint_limit_violations(
                joints_deg,
                lower_joint_limits_deg,
                upper_joint_limits_deg,
                ARM_JOINT_NAMES,
            )
            if violations:
                last_joint_limit_failures = list(violations)
                rejection_reason = (
                    "Position-axis IK violated arm joint limits: "
                    f"{violations}"
                )
                continue

            solved_pose = worker_np.asarray(
                robot.get_T_world_frame(LEROBOT_TARGET_FRAME),
                dtype=worker_np.float64,
            )
            if not worker_np.all(worker_np.isfinite(solved_pose)):
                rejection_reason = (
                    "Position-axis forward kinematics returned NaN or infinity"
                )
                continue
            position_error_m = float(
                worker_np.linalg.norm(solved_pose[:3, 3] - target_position_m)
            )
            solved_axis_world = solved_pose[:3, :3] @ tool_axis_frame
            axis_error_rad = axis_alignment_error_rad(
                solved_axis_world,
                target_axis_world,
            )
            closing_axis_error_rad = 0.0
            if (
                tool_closing_axis_frame is not None
                and target_closing_axis_world is not None
            ):
                solved_closing_axis_world = (
                    solved_pose[:3, :3] @ tool_closing_axis_frame
                )
                closing_axis_error_rad = axis_alignment_error_rad(
                    solved_closing_axis_world,
                    target_closing_axis_world,
                )
            candidate = {
                "joints_deg": joints_deg.tolist(),
                "solved_position_m": solved_pose[:3, 3].tolist(),
                "solved_rotation": solved_pose[:3, :3].tolist(),
                "solved_axis_world": solved_axis_world.tolist(),
                "position_error_m": position_error_m,
                "axis_error_rad": axis_error_rad,
                "closing_axis_error_rad": closing_axis_error_rad,
                "iterations": iteration,
            }
            score = (
                max(
                    position_error_m / position_tolerance_m,
                    axis_error_rad / axis_tolerance_rad,
                    closing_axis_error_rad / closing_axis_tolerance_rad,
                ),
                position_error_m,
                axis_error_rad,
                closing_axis_error_rad,
                iteration,
            )
            if best_state is None or score < (
                max(
                    float(best_state["position_error_m"])
                    / position_tolerance_m,
                    float(best_state["axis_error_rad"])
                    / axis_tolerance_rad,
                    float(best_state["closing_axis_error_rad"])
                    / closing_axis_tolerance_rad,
                ),
                float(best_state["position_error_m"]),
                float(best_state["axis_error_rad"]),
                float(best_state["closing_axis_error_rad"]),
                int(best_state["iterations"]),
            ):
                best_state = candidate
            if (
                position_error_m <= position_tolerance_m
                and axis_error_rad <= axis_tolerance_rad
                and closing_axis_error_rad <= closing_axis_tolerance_rad
            ):
                context.last_solve_diagnostics = {
                    "iterations": iteration,
                    "position_error_m": position_error_m,
                    "axis_error_rad": axis_error_rad,
                    "closing_axis_error_rad": closing_axis_error_rad,
                    "joint_limit_failures": [],
                }
                return candidate, None

        if best_state is not None:
            rejection_reason = (
                "Position-axis residual gate failed: "
                f"position={float(best_state['position_error_m']) * 1000.0:.3f} mm "
                f"(limit={position_tolerance_m * 1000.0:.3f} mm), "
                f"axis={math.degrees(float(best_state['axis_error_rad'])):.3f} deg "
                f"(limit={math.degrees(axis_tolerance_rad):.3f} deg), "
                f"closing-axis={math.degrees(float(best_state['closing_axis_error_rad'])):.3f} deg "
                f"(limit={math.degrees(closing_axis_tolerance_rad):.3f} deg)"
            )
        context.last_solve_diagnostics = {
            "iterations": iteration,
            "position_error_m": (
                float(best_state["position_error_m"])
                if best_state is not None
                else None
            ),
            "axis_error_rad": (
                float(best_state["axis_error_rad"])
                if best_state is not None
                else None
            ),
            "closing_axis_error_rad": (
                float(best_state["closing_axis_error_rad"])
                if best_state is not None
                else None
            ),
            "joint_limit_failures": last_joint_limit_failures,
        }
        return None, rejection_reason

    def solve_position_waypoint(
        kinematics: RobotKinematics,
        seed_joints_deg: worker_np.ndarray,
        target_position_m: worker_np.ndarray,
        max_iterations: int,
        lower_joint_limits_deg: worker_np.ndarray,
        upper_joint_limits_deg: worker_np.ndarray,
    ) -> tuple[dict[str, object] | None, str | None]:
        """Return the best position-only state at one fallback waypoint."""
        target_pose = worker_np.asarray(
            checked_fk(kinematics, seed_joints_deg),
            dtype=worker_np.float64,
        )
        target_pose[:3, 3] = target_position_m
        joints_deg = seed_joints_deg.copy()
        best_state: dict[str, object] | None = None
        rejection_reason = "No valid position-only state was produced"

        for iteration in range(1, max_iterations + 1):
            joints_deg = worker_np.asarray(
                kinematics.inverse_kinematics(
                    current_joint_pos=joints_deg,
                    desired_ee_pose=target_pose,
                    position_weight=1.0,
                    orientation_weight=0.0,
                ),
                dtype=worker_np.float64,
            )
            if not worker_np.all(worker_np.isfinite(joints_deg)):
                rejection_reason = "Position-only IK returned NaN or infinity"
                break
            if joint_limit_violations(
                joints_deg,
                lower_joint_limits_deg,
                upper_joint_limits_deg,
                ARM_JOINT_NAMES,
            ):
                rejection_reason = "Position-only IK violated arm joint limits"
                continue
            solved_pose = worker_np.asarray(
                checked_fk(kinematics, joints_deg),
                dtype=worker_np.float64,
            )
            if not worker_np.all(worker_np.isfinite(solved_pose)):
                rejection_reason = (
                    "Position-only forward kinematics returned NaN or infinity"
                )
                continue
            position_error_m = float(
                worker_np.linalg.norm(solved_pose[:3, 3] - target_position_m)
            )
            candidate = {
                "joints_deg": joints_deg.tolist(),
                "solved_position_m": solved_pose[:3, 3].tolist(),
                "position_error_m": position_error_m,
                "iterations": iteration,
            }
            score = (position_error_m, iteration)
            if best_state is None or score < (
                float(best_state["position_error_m"]),
                int(best_state["iterations"]),
            ):
                best_state = candidate

        if best_state is not None:
            return best_state, None
        return None, rejection_reason

    def evaluate_position_axis_path(
        waypoint_positions_m: worker_np.ndarray,
        tool_axis_frame: worker_np.ndarray,
        target_axis_world: worker_np.ndarray,
        max_iterations: int,
        position_tolerance_m: float,
        axis_tolerance_rad: float,
        seed_regularization_weight: float,
        lower_joint_limits_deg: worker_np.ndarray,
        upper_joint_limits_deg: worker_np.ndarray,
        metadata: dict[str, object],
        tool_closing_axis_frame: worker_np.ndarray,
        target_closing_axis_world: worker_np.ndarray,
        closing_axis_weight: float,
        closing_axis_tolerance_rad: float,
        endpoint_state: dict[str, object],
        solver_context: PositionAxisSolverContext,
    ) -> dict[str, object]:
        """Expand one feasible endpoint without solving it a second time."""
        candidate = dict(metadata)
        candidate.update(
            {
                "constraint_mode": "position_hard_axis_gated",
                "position_tolerance_m": position_tolerance_m,
                "axis_tolerance_rad": axis_tolerance_rad,
                "target_axis_world": target_axis_world.tolist(),
                "waypoint_target_positions_m": waypoint_positions_m.tolist(),
            }
        )
        reverse_states: list[dict[str, object]] = [dict(endpoint_state)]
        waypoint_seed = worker_np.asarray(
            endpoint_state["joints_deg"],
            dtype=worker_np.float64,
        )
        for reverse_index, waypoint_position_m in enumerate(
            waypoint_positions_m[-2::-1],
            start=1,
        ):
            closing_progress = 1.0 - (
                reverse_index / max(1, len(waypoint_positions_m) - 1)
            )
            state, reason = solve_position_axis_waypoint(
                waypoint_seed,
                waypoint_position_m,
                tool_axis_frame,
                target_axis_world,
                max_iterations,
                position_tolerance_m,
                axis_tolerance_rad,
                seed_regularization_weight,
                lower_joint_limits_deg,
                upper_joint_limits_deg,
                tool_closing_axis_frame,
                target_closing_axis_world,
                closing_axis_weight * closing_progress,
                (
                    closing_axis_tolerance_rad
                    if reverse_index == 0
                    else math.pi
                ),
                solver_context,
            )
            if state is None:
                candidate.update(
                    {
                        "status": "rejected",
                        "rejection_stage": "path",
                        "failed_waypoint_index": (
                            len(waypoint_positions_m) - 1 - reverse_index
                        ),
                        "reason": reason,
                        "failed_waypoint_diagnostics": dict(
                            solver_context.last_solve_diagnostics
                        ),
                    }
                )
                return candidate
            reverse_states.append(state)
            waypoint_seed = worker_np.asarray(
                state["joints_deg"],
                dtype=worker_np.float64,
            )

        states = list(reversed(reverse_states))
        joints = worker_np.asarray(
            [state["joints_deg"] for state in states],
            dtype=worker_np.float64,
        )
        position_errors = [float(state["position_error_m"]) for state in states]
        axis_errors = [float(state["axis_error_rad"]) for state in states]
        closing_axis_errors = [
            float(state["closing_axis_error_rad"]) for state in states
        ]
        waypoint_iterations = [int(state["iterations"]) for state in states]
        candidate.update(
            {
                "status": "valid",
                "rejection_stage": None,
                "waypoint_joints_deg": joints.tolist(),
                "waypoint_solved_positions_m": [
                    state["solved_position_m"] for state in states
                ],
                "waypoint_solved_rotations": [
                    state["solved_rotation"] for state in states
                ],
                "waypoint_solved_axes_world": [
                    state["solved_axis_world"] for state in states
                ],
                "waypoint_position_errors_m": position_errors,
                "waypoint_axis_errors_rad": axis_errors,
                "waypoint_closing_axis_errors_rad": closing_axis_errors,
                "waypoint_iterations": waypoint_iterations,
                "max_position_error_m": max(position_errors),
                "terminal_position_error_m": position_errors[-1],
                "max_axis_error_rad": max(axis_errors),
                "terminal_axis_error_rad": axis_errors[-1],
                "total_joint_travel_deg": float(
                    worker_np.sum(
                        worker_np.linalg.norm(
                            worker_np.diff(joints, axis=0),
                            axis=1,
                        )
                    )
                ),
            }
        )
        return candidate

    request = json.load(sys.stdin)
    operation = request.get("operation")
    if operation == "setup":
        # Constructing RobotKinematics loads Placo, parses the SO-101 URDF,
        # fixes the model base, and creates the end-effector frame task.
        kinematics = RobotKinematics(
            urdf_path=str(SO101_URDF_PATH),
            target_frame_name=LEROBOT_TARGET_FRAME,
            joint_names=ARM_JOINT_NAMES,
        )
        result = {
            "status": "ready",
            "lerobot_version": importlib.metadata.version("lerobot"),
            "placo_version": importlib.metadata.version("placo"),
            "urdf_path": str(SO101_URDF_PATH),
            "target_frame": kinematics.target_frame_name,
            "joint_names": kinematics.joint_names,
        }
    elif operation == "forward_kinematics":
        joints_deg = request_array(
            "joints_deg",
            (len(ARM_JOINT_NAMES),),
        )
        kinematics = RobotKinematics(
            urdf_path=str(SO101_URDF_PATH),
            target_frame_name=LEROBOT_TARGET_FRAME,
            joint_names=ARM_JOINT_NAMES,
        )
        pose = checked_fk(kinematics, joints_deg)
        result = {
            "status": "valid",
            "joint_names": ARM_JOINT_NAMES,
            "joints_deg": joints_deg.tolist(),
            "pose": pose.tolist(),
        }
    elif operation == "position_ik":
        fingertip_offset_m = request_array("fingertip_offset_m", (3,))
        seed_joints_deg = request_array(
            "seed_joints_deg",
            (len(ARM_JOINT_NAMES),),
        )
        target_position_m = request_array("target_position_m", (3,))
        max_iterations = int(request["max_iterations"])
        tolerance_m = float(request["position_tolerance_m"])
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive and finite")

        with tempfile.TemporaryDirectory(prefix="so101_position_ik_") as temp_dir:
            kinematics = make_fingertip_kinematics(
                fingertip_offset_m,
                temp_dir,
            )
            result = solve_position_only(
                kinematics,
                seed_joints_deg,
                target_position_m,
                max_iterations,
                tolerance_m,
            )
        result["joint_names"] = ARM_JOINT_NAMES
    elif operation == "grasp_candidates":
        seed_joints_deg = request_array(
            "seed_joints_deg",
            (len(ARM_JOINT_NAMES),),
        )
        fingertip_offset_m = request_array("fingertip_offset_m", (3,))
        candidate_waypoints = worker_np.asarray(
            request["candidate_waypoint_positions_m"],
            dtype=worker_np.float64,
        )
        candidate_axes = worker_np.asarray(
            request["candidate_target_axes_world"],
            dtype=worker_np.float64,
        )
        candidate_closing_axes = worker_np.asarray(
            request["candidate_target_closing_axes_world"],
            dtype=worker_np.float64,
        )
        metadata = request["candidate_metadata"]
        if (
            candidate_waypoints.ndim != 3
            or candidate_waypoints.shape[0] == 0
            or candidate_waypoints.shape[1] < 2
            or candidate_waypoints.shape[2] != 3
            or not worker_np.all(worker_np.isfinite(candidate_waypoints))
            or candidate_axes.shape != (candidate_waypoints.shape[0], 3)
            or candidate_closing_axes.shape
            != (candidate_waypoints.shape[0], 3)
            or not worker_np.all(worker_np.isfinite(candidate_axes))
            or not worker_np.all(worker_np.isfinite(candidate_closing_axes))
            or not isinstance(metadata, list)
            or len(metadata) != candidate_waypoints.shape[0]
            or not all(isinstance(item, dict) for item in metadata)
        ):
            raise ValueError("Candidate waypoint, axis, and metadata arrays are invalid")
        axis_norms = worker_np.linalg.norm(candidate_axes, axis=1)
        if worker_np.any(axis_norms <= 1e-12):
            raise ValueError("Candidate target axes must have nonzero length")
        candidate_axes = candidate_axes / axis_norms[:, None]
        closing_norms = worker_np.linalg.norm(candidate_closing_axes, axis=1)
        if worker_np.any(closing_norms <= 1e-12):
            raise ValueError("Candidate closing axes must have nonzero length")
        candidate_closing_axes = candidate_closing_axes / closing_norms[:, None]
        tool_axis_frame = request_array("tool_axis_frame", (3,))
        tool_axis_norm = float(worker_np.linalg.norm(tool_axis_frame))
        if tool_axis_norm <= 1e-12:
            raise ValueError("Tool axis must have nonzero length")
        tool_axis_frame = tool_axis_frame / tool_axis_norm
        tool_closing_axis_frame = request_array(
            "tool_closing_axis_frame", (3,)
        )
        tool_closing_axis_frame /= worker_np.linalg.norm(
            tool_closing_axis_frame
        )
        closing_axis_weight = float(request["closing_axis_weight"])
        closing_axis_tolerance_rad = float(
            request["closing_axis_tolerance_rad"]
        )
        lower_joint_limits_deg = request_array(
            "lower_joint_limits_deg", (len(ARM_JOINT_NAMES),)
        )
        upper_joint_limits_deg = request_array(
            "upper_joint_limits_deg", (len(ARM_JOINT_NAMES),)
        )
        joint_limit_violations(
            lower_joint_limits_deg,
            lower_joint_limits_deg,
            upper_joint_limits_deg,
            ARM_JOINT_NAMES,
        )
        max_iterations = int(request["max_iterations"])
        position_tolerance_m = float(request["position_tolerance_m"])
        axis_tolerance_rad = float(request["axis_tolerance_rad"])
        seed_regularization_weight = float(request["seed_regularization_weight"])
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if position_tolerance_m <= 0.0 or axis_tolerance_rad <= 0.0:
            raise ValueError("Candidate residual tolerances must be positive")

        planning_started = time.perf_counter()
        solver_context = PositionAxisSolverContext(
            tool_axis_frame,
            tool_closing_axis_frame,
            seed_regularization_weight,
        )
        endpoint_states: dict[int, dict[str, object]] = {}
        candidates: list[dict[str, object]] = []
        position_seed_solver_calls = 0
        position_seed_iterations = 0
        endpoint_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="so101_grasp_candidates_") as temp_dir:
            kinematics = make_fingertip_kinematics(fingertip_offset_m, temp_dir)
            for candidate_index, (
                waypoints,
                target_axis,
                target_closing_axis,
                candidate_metadata,
            ) in enumerate(
                zip(
                    candidate_waypoints,
                    candidate_axes,
                    candidate_closing_axes,
                    metadata,
                )
            ):
                candidate = dict(candidate_metadata)
                candidate.update(
                    {
                        "candidate_index": candidate_index,
                        "constraint_mode": "position_hard_axis_gated",
                        "position_tolerance_m": position_tolerance_m,
                        "axis_tolerance_rad": axis_tolerance_rad,
                        "target_axis_world": target_axis.tolist(),
                        "waypoint_target_positions_m": waypoints.tolist(),
                    }
                )
                position_seed_solver_calls += 1
                position_seed, seed_reason = solve_position_waypoint(
                    kinematics,
                    seed_joints_deg,
                    waypoints[-1],
                    max_iterations,
                    lower_joint_limits_deg,
                    upper_joint_limits_deg,
                )
                if position_seed is None:
                    candidate.update(
                        {
                            "status": "rejected",
                            "rejection_stage": "endpoint",
                            "reason": seed_reason,
                        }
                    )
                    candidates.append(candidate)
                    continue
                position_seed_iterations += int(position_seed["iterations"])
                endpoint_state, endpoint_reason = solve_position_axis_waypoint(
                    worker_np.asarray(
                        position_seed["joints_deg"], dtype=worker_np.float64
                    ),
                    waypoints[-1],
                    tool_axis_frame,
                    target_axis,
                    max_iterations,
                    position_tolerance_m,
                    axis_tolerance_rad,
                    seed_regularization_weight,
                    lower_joint_limits_deg,
                    upper_joint_limits_deg,
                    tool_closing_axis_frame,
                    target_closing_axis,
                    closing_axis_weight,
                    closing_axis_tolerance_rad,
                    solver_context,
                )
                if endpoint_state is None:
                    candidate.update(
                        {
                            "status": "rejected",
                            "rejection_stage": "endpoint",
                            "reason": endpoint_reason,
                            "position_seed_iterations": position_seed[
                                "iterations"
                            ],
                            "endpoint_diagnostics": dict(
                                solver_context.last_solve_diagnostics
                            ),
                        }
                    )
                    candidates.append(candidate)
                    continue
                endpoint_states[candidate_index] = endpoint_state
                candidate.update(
                    {
                        "status": "endpoint_valid",
                        "rejection_stage": None,
                        "position_seed_iterations": position_seed["iterations"],
                        "endpoint_iterations": endpoint_state["iterations"],
                        "endpoint_joints_deg": endpoint_state["joints_deg"],
                        "endpoint_position_error_m": endpoint_state[
                            "position_error_m"
                        ],
                        "endpoint_axis_error_rad": endpoint_state[
                            "axis_error_rad"
                        ],
                        "endpoint_closing_axis_error_rad": endpoint_state[
                            "closing_axis_error_rad"
                        ],
                        "endpoint_joint_travel_deg": float(
                            worker_np.linalg.norm(
                                worker_np.asarray(
                                    endpoint_state["joints_deg"],
                                    dtype=worker_np.float64,
                                )
                                - seed_joints_deg
                            )
                        ),
                    }
                )
                candidates.append(candidate)

            endpoint_duration_seconds = time.perf_counter() - endpoint_started
            full_path_started = time.perf_counter()

            def evaluate_ranked_path(
                candidate: dict[str, object],
            ) -> dict[str, object]:
                candidate_index = int(candidate["candidate_index"])
                return evaluate_position_axis_path(
                    candidate_waypoints[candidate_index],
                    tool_axis_frame,
                    candidate_axes[candidate_index],
                    max_iterations,
                    position_tolerance_m,
                    axis_tolerance_rad,
                    seed_regularization_weight,
                    lower_joint_limits_deg,
                    upper_joint_limits_deg,
                    candidate,
                    tool_closing_axis_frame,
                    candidate_closing_axes[candidate_index],
                    closing_axis_weight,
                    closing_axis_tolerance_rad,
                    endpoint_states[candidate_index],
                    solver_context,
                )

            selected, full_paths_evaluated, full_paths_skipped = (
                evaluate_first_valid_path(candidates, evaluate_ranked_path)
            )
            full_path_duration_seconds = (
                time.perf_counter() - full_path_started
            )
        planning_duration_seconds = time.perf_counter() - planning_started
        result = {
            "status": "generated",
            "joint_names": ARM_JOINT_NAMES,
            "candidate_count": len(candidates),
            "selected_candidate_id": (
                selected.get("candidate_id") if selected is not None else None
            ),
            "candidates": candidates,
            "planning_metrics": {
                "strategy": "endpoint_first_first_valid",
                "endpoint_candidates": len(candidates),
                "endpoint_feasible": len(endpoint_states),
                "full_paths_evaluated": full_paths_evaluated,
                "full_paths_skipped": full_paths_skipped,
                "solver_calls": solver_context.waypoint_solve_calls,
                "solver_iterations": solver_context.solver_iterations,
                "position_seed_solver_calls": position_seed_solver_calls,
                "position_seed_iterations": position_seed_iterations,
                "solver_context_builds": solver_context.context_builds,
                "solver_state_resets": solver_context.state_resets,
                "endpoint_duration_seconds": endpoint_duration_seconds,
                "full_path_duration_seconds": full_path_duration_seconds,
                "total_planning_duration_seconds": planning_duration_seconds,
            },
        }
    elif operation == "top_down_candidates":
        fingertip_offset_m = request_array("fingertip_offset_m", (3,))
        seed_joints_deg = request_array(
            "seed_joints_deg",
            (len(ARM_JOINT_NAMES),),
        )
        waypoint_positions_m = worker_np.asarray(
            request["waypoint_positions_m"],
            dtype=worker_np.float64,
        )
        if (
            waypoint_positions_m.ndim != 2
            or waypoint_positions_m.shape[0] < 2
            or waypoint_positions_m.shape[1] != 3
            or not worker_np.all(worker_np.isfinite(waypoint_positions_m))
        ):
            raise ValueError(
                "waypoint_positions_m must be at least two finite three-vectors"
            )
        tool_axis_frame = request_array("tool_axis_frame", (3,))
        target_axis_world = request_array("target_axis_world", (3,))
        tool_axis_norm = float(worker_np.linalg.norm(tool_axis_frame))
        target_axis_norm = float(worker_np.linalg.norm(target_axis_world))
        if tool_axis_norm <= 1e-12 or target_axis_norm <= 1e-12:
            raise ValueError("Tool and target axes must have nonzero length")
        tool_axis_frame = tool_axis_frame / tool_axis_norm
        target_axis_world = target_axis_world / target_axis_norm
        lower_joint_limits_deg = request_array(
            "lower_joint_limits_deg",
            (len(ARM_JOINT_NAMES),),
        )
        upper_joint_limits_deg = request_array(
            "upper_joint_limits_deg",
            (len(ARM_JOINT_NAMES),),
        )
        joint_limit_violations(
            lower_joint_limits_deg,
            lower_joint_limits_deg,
            upper_joint_limits_deg,
            ARM_JOINT_NAMES,
        )
        max_iterations = int(request["max_iterations"])
        position_tolerance_m = float(request["position_tolerance_m"])
        axis_tolerance_rad = float(request["axis_tolerance_rad"])
        seed_regularization_weight = float(
            request["seed_regularization_weight"]
        )
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive and finite")
        if not math.isfinite(axis_tolerance_rad) or not (
            0.0 < axis_tolerance_rad < math.pi
        ):
            raise ValueError("axis_tolerance_rad must be in the range (0, pi)")
        if not math.isfinite(seed_regularization_weight) or not (
            0.0 < seed_regularization_weight <= 1.0
        ):
            raise ValueError(
                "seed_regularization_weight must be in the range (0, 1]"
            )

        final_target_position_m = waypoint_positions_m[-1]
        with tempfile.TemporaryDirectory(prefix="so101_top_down_ik_") as temp_dir:
            kinematics = make_fingertip_kinematics(
                fingertip_offset_m,
                temp_dir,
            )
            position_seed = solve_position_only(
                kinematics,
                seed_joints_deg,
                final_target_position_m,
                max_iterations,
                None,
            )
            position_seed_joints_deg = worker_np.asarray(
                position_seed["solved_joints_deg"],
                dtype=worker_np.float64,
            )
            if joint_limit_violations(
                position_seed_joints_deg,
                lower_joint_limits_deg,
                upper_joint_limits_deg,
                ARM_JOINT_NAMES,
            ):
                raise RuntimeError("Position-only seed violates arm joint limits")

            reverse_waypoint_states: list[dict[str, object]] = []
            candidate_reason: str | None = None
            waypoint_seed_joints_deg = position_seed_joints_deg.copy()
            for waypoint_position_m in waypoint_positions_m[::-1]:
                waypoint_state, rejection_reason = solve_position_axis_waypoint(
                    waypoint_seed_joints_deg,
                    waypoint_position_m,
                    tool_axis_frame,
                    target_axis_world,
                    max_iterations,
                    position_tolerance_m,
                    axis_tolerance_rad,
                    seed_regularization_weight,
                    lower_joint_limits_deg,
                    upper_joint_limits_deg,
                )
                if waypoint_state is None:
                    candidate_reason = rejection_reason
                    break
                reverse_waypoint_states.append(waypoint_state)
                waypoint_seed_joints_deg = worker_np.asarray(
                    waypoint_state["joints_deg"],
                    dtype=worker_np.float64,
                )

            constrained_candidate: dict[str, object] = {
                "constraint_mode": "position_hard_axis_gated",
                "position_tolerance_m": position_tolerance_m,
                "axis_tolerance_rad": axis_tolerance_rad,
            }
            if candidate_reason is not None:
                constrained_candidate.update(
                    {
                        "status": "rejected",
                        "reason": candidate_reason,
                    }
                )
                candidates: list[dict[str, object]] = [constrained_candidate]
            else:
                waypoint_states = list(reversed(reverse_waypoint_states))
                waypoint_position_errors_m = [
                    float(state["position_error_m"])
                    for state in waypoint_states
                ]
                waypoint_axis_errors_rad = [
                    float(state["axis_error_rad"])
                    for state in waypoint_states
                ]
                waypoint_solved_positions_m = [
                    state["solved_position_m"] for state in waypoint_states
                ]
                waypoint_solved_axes_world = [
                    state["solved_axis_world"] for state in waypoint_states
                ]
                waypoint_joints_deg = worker_np.asarray(
                    [state["joints_deg"] for state in waypoint_states],
                    dtype=worker_np.float64,
                )
                constrained_candidate.update(
                    {
                        "status": "valid",
                        "waypoint_joints_deg": waypoint_joints_deg.tolist(),
                        "waypoint_solved_positions_m": waypoint_solved_positions_m,
                        "waypoint_solved_axes_world": waypoint_solved_axes_world,
                        "waypoint_position_errors_m": waypoint_position_errors_m,
                        "waypoint_axis_errors_rad": waypoint_axis_errors_rad,
                        "max_position_error_m": max(waypoint_position_errors_m),
                        "terminal_position_error_m": waypoint_position_errors_m[-1],
                        "max_axis_error_rad": max(waypoint_axis_errors_rad),
                        "terminal_axis_error_rad": waypoint_axis_errors_rad[-1],
                        "total_joint_travel_deg": float(
                            worker_np.sum(
                                worker_np.linalg.norm(
                                    worker_np.diff(waypoint_joints_deg, axis=0),
                                    axis=1,
                                )
                            )
                        ),
                    }
                )
                candidates = [constrained_candidate]

            # The fallback follows the identical hover-to-contact Cartesian
            # path but deliberately leaves orientation unconstrained. Build it
            # here so it uses the same validated temporary fingertip model,
            # seed and limits as the oriented branches.
            reverse_fallback_states: list[dict[str, object]] = [
                {
                    "joints_deg": position_seed_joints_deg.tolist(),
                    "position_error_m": float(
                        position_seed["position_error_m"]
                    ),
                    "solved_position_m": position_seed["solved_position_m"],
                    "iterations": int(position_seed["iterations"]),
                }
            ]
            fallback_reason: str | None = None
            fallback_seed_joints_deg = position_seed_joints_deg.copy()
            for waypoint_position_m in waypoint_positions_m[-2::-1]:
                fallback_state, rejection_reason = solve_position_waypoint(
                    kinematics,
                    fallback_seed_joints_deg,
                    waypoint_position_m,
                    max_iterations,
                    lower_joint_limits_deg,
                    upper_joint_limits_deg,
                )
                if fallback_state is None:
                    fallback_reason = rejection_reason
                    break
                reverse_fallback_states.append(fallback_state)
                fallback_seed_joints_deg = worker_np.asarray(
                    fallback_state["joints_deg"],
                    dtype=worker_np.float64,
                )

            if fallback_reason is not None:
                position_only_fallback: dict[str, object] = {
                    "status": "rejected",
                    "reason": fallback_reason,
                }
            else:
                fallback_states = list(reversed(reverse_fallback_states))
                fallback_waypoint_joints_deg = worker_np.asarray(
                    [state["joints_deg"] for state in fallback_states],
                    dtype=worker_np.float64,
                )
                fallback_position_errors_m = [
                    float(state["position_error_m"])
                    for state in fallback_states
                ]
                fallback_solved_positions_m = [
                    state["solved_position_m"] for state in fallback_states
                ]
                position_only_fallback = {
                    "status": "valid",
                    "waypoint_joints_deg": (
                        fallback_waypoint_joints_deg.tolist()
                    ),
                    "waypoint_solved_positions_m": fallback_solved_positions_m,
                    "waypoint_position_errors_m": fallback_position_errors_m,
                    "max_position_error_m": max(fallback_position_errors_m),
                    "terminal_position_error_m": fallback_position_errors_m[-1],
                    "total_joint_travel_deg": float(
                        worker_np.sum(
                            worker_np.linalg.norm(
                                worker_np.diff(
                                    fallback_waypoint_joints_deg,
                                    axis=0,
                                ),
                                axis=1,
                            )
                        )
                    ),
                }

        result = {
            "status": "generated",
            "joint_names": ARM_JOINT_NAMES,
            "position_seed": position_seed,
            "waypoint_positions_m": waypoint_positions_m.tolist(),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "position_only_fallback": position_only_fallback,
        }
    else:
        raise ValueError(
            f"Unsupported LeRobot worker operation: {operation!r}"
        )
    print(LEROBOT_WORKER_MARKER + json.dumps(result, allow_nan=False))
    return 0


# The LeRobot environment uses Python 3.11 while Isaac Sim embeds Python 3.12.
# Dispatch the worker before importing Isaac so the two native environments
# never share an interpreter.
if "--lerobot-worker" in sys.argv:
    raise SystemExit(lerobot_worker_main())


from isaacsim import SimulationApp  # noqa: E402


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
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help=(
            "Write a structured run result to this path, including failure "
            "stage and measured grasp/lift metrics"
        ),
    )
    parser.add_argument(
        "--disable-self-collisions-diagnostic",
        action="store_true",
        help=(
            "Temporarily disable articulation self-collisions to distinguish "
            "collision-blocked motion from controller tracking failure; "
            "diagnostic only and disabled by default"
        ),
    )
    parser.add_argument(
        "--arm-damping-multiplier",
        type=float,
        default=DEFAULT_ARM_DAMPING_MULTIPLIER,
        help=(
            "Multiply the five arm-joint damping gains at runtime for a "
            "controlled tracking adjustment "
            f"(default: {DEFAULT_ARM_DAMPING_MULTIPLIER:.1f})"
        ),
    )
    parser.add_argument(
        "--arm-stiffness-multiplier",
        type=float,
        default=DEFAULT_ARM_STIFFNESS_MULTIPLIER,
        help=(
            "Multiply the five arm-joint stiffness gains at runtime for a "
            "controlled tracking adjustment "
            f"(default: {DEFAULT_ARM_STIFFNESS_MULTIPLIER:.1f})"
        ),
    )
    parser.add_argument(
        "--ik-debug-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Draw the interpreted grasp geometry, target approach, yaw "
            "directions, and final residual in the Isaac viewport "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--lerobot-python",
        type=Path,
        default=DEFAULT_LEROBOT_PYTHON,
        help=(
            "Python interpreter containing LeRobot and Placo "
            f"(default: {DEFAULT_LEROBOT_PYTHON})"
        ),
    )
    parser.add_argument(
        "--lift-height-mm",
        type=float,
        default=DEFAULT_LIFT_HEIGHT_MM,
        help=(
            "Vertical distance to lift the grasped sphere "
            f"(default: {DEFAULT_LIFT_HEIGHT_MM:.1f} mm)"
        ),
    )
    parser.add_argument(
        "--lift-stop-distance-mm",
        type=float,
        default=DEFAULT_LIFT_STOP_DISTANCE_MM,
        help=(
            "Stop the lift when the authored gripper contact point is within "
            "this distance of the lift target "
            f"(default: {DEFAULT_LIFT_STOP_DISTANCE_MM:.1f} mm)"
        ),
    )
    parser.add_argument(
        "--top-down-hover-height-mm",
        type=float,
        default=DEFAULT_TOP_DOWN_HOVER_HEIGHT_MM,
        help=(
            "Height above the final fixed-finger target for the "
            "top-down hover pose "
            f"(default: {DEFAULT_TOP_DOWN_HOVER_HEIGHT_MM:.1f} mm)"
        ),
    )
    parser.add_argument(
        "--top-down-descent-step-mm",
        type=float,
        default=DEFAULT_TOP_DOWN_DESCENT_STEP_MM,
        help=(
            "Maximum spacing between vertical top-down waypoints "
            f"(default: {DEFAULT_TOP_DOWN_DESCENT_STEP_MM:.1f} mm)"
        ),
    )
    parser.add_argument(
        "--sphere-surface-clearance-mm",
        type=float,
        default=DEFAULT_SPHERE_SURFACE_CLEARANCE_MM,
        help=(
            "Radial stand-off between the sphere surface and the URDF "
            "fixed-finger tool point before jaw closure "
            f"(default: {DEFAULT_SPHERE_SURFACE_CLEARANCE_MM:.1f} mm)"
        ),
    )
    parser.add_argument(
        "--top-down-yaw-offsets-deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOP_DOWN_YAW_OFFSETS_DEG),
        help=(
            "Yaw targets, in degrees around the geometry-derived nominal "
            "top-down pose "
            f"(default: {' '.join(str(value) for value in DEFAULT_TOP_DOWN_YAW_OFFSETS_DEG)})"
        ),
    )
    parser.add_argument(
        "--top-down-position-tolerance-mm",
        type=float,
        default=DEFAULT_TOP_DOWN_POSITION_TOLERANCE_MM,
        help=(
            "Maximum accepted TCP residual for constrained top-down IK "
            f"(default: {DEFAULT_TOP_DOWN_POSITION_TOLERANCE_MM:.3f} mm)"
        ),
    )
    parser.add_argument(
        "--top-down-axis-tolerance-deg",
        type=float,
        default=DEFAULT_TOP_DOWN_AXIS_TOLERANCE_DEG,
        help=(
            "Maximum accepted tool-approach-axis error for constrained IK "
            f"(default: {DEFAULT_TOP_DOWN_AXIS_TOLERANCE_DEG:.3f} deg)"
        ),
    )
    args, _ = parser.parse_known_args()
    if args.settle_steps < 0:
        parser.error("--settle-steps cannot be negative")
    if (
        not math.isfinite(args.arm_damping_multiplier)
        or args.arm_damping_multiplier <= 0.0
    ):
        parser.error("--arm-damping-multiplier must be positive")
    if (
        not math.isfinite(args.arm_stiffness_multiplier)
        or args.arm_stiffness_multiplier <= 0.0
    ):
        parser.error("--arm-stiffness-multiplier must be positive")
    if (
        not math.isfinite(args.lift_height_mm)
        or args.lift_height_mm <= 0.0
    ):
        parser.error("--lift-height-mm must be a positive number")
    if (
        not math.isfinite(args.lift_stop_distance_mm)
        or args.lift_stop_distance_mm <= 0.0
    ):
        parser.error("--lift-stop-distance-mm must be a positive number")
    if (
        not math.isfinite(args.top_down_hover_height_mm)
        or args.top_down_hover_height_mm <= 0.0
    ):
        parser.error("--top-down-hover-height-mm must be a positive number")
    if (
        not math.isfinite(args.top_down_descent_step_mm)
        or args.top_down_descent_step_mm <= 0.0
    ):
        parser.error("--top-down-descent-step-mm must be a positive number")
    if args.top_down_descent_step_mm > args.top_down_hover_height_mm:
        parser.error(
            "--top-down-descent-step-mm cannot exceed "
            "--top-down-hover-height-mm"
        )
    if (
        not math.isfinite(args.sphere_surface_clearance_mm)
        or args.sphere_surface_clearance_mm < 0.0
    ):
        parser.error("--sphere-surface-clearance-mm must be nonnegative")
    if not args.top_down_yaw_offsets_deg or not all(
        math.isfinite(value) for value in args.top_down_yaw_offsets_deg
    ):
        parser.error("--top-down-yaw-offsets-deg must contain finite values")
    if (
        not math.isfinite(args.top_down_position_tolerance_mm)
        or args.top_down_position_tolerance_mm <= 0.0
    ):
        parser.error("--top-down-position-tolerance-mm must be positive")
    if (
        not math.isfinite(args.top_down_axis_tolerance_deg)
        or not 0.0 < args.top_down_axis_tolerance_deg < 180.0
    ):
        parser.error(
            "--top-down-axis-tolerance-deg must be in the range (0, 180)"
        )
    return args


ARGS = parse_args()
RUN_STARTED_MONOTONIC = time.monotonic()
CURRENT_RUN_STAGE = "startup"
RUN_SUMMARY: dict[str, object] = {
    "schema_version": 1,
    "status": "running",
    "success": False,
    "failure_stage": None,
    "failure_reason": None,
    "configuration": {
        "world": str(ARGS.world.expanduser().resolve()),
        "headless": ARGS.headless,
        "settle_steps": ARGS.settle_steps,
        "ik_debug_overlay": ARGS.ik_debug_overlay,
        "lerobot_python": str(ARGS.lerobot_python.expanduser().resolve()),
        "lift_height_mm": ARGS.lift_height_mm,
        "lift_stop_distance_mm": ARGS.lift_stop_distance_mm,
        "top_down_hover_height_mm": ARGS.top_down_hover_height_mm,
        "top_down_descent_step_mm": ARGS.top_down_descent_step_mm,
        "sphere_surface_clearance_mm": ARGS.sphere_surface_clearance_mm,
        "top_down_yaw_offsets_deg": ARGS.top_down_yaw_offsets_deg,
        "top_down_position_tolerance_mm": (
            ARGS.top_down_position_tolerance_mm
        ),
        "top_down_axis_tolerance_deg": ARGS.top_down_axis_tolerance_deg,
        "disable_self_collisions_diagnostic": (
            ARGS.disable_self_collisions_diagnostic
        ),
        "arm_damping_multiplier": ARGS.arm_damping_multiplier,
        "arm_stiffness_multiplier": ARGS.arm_stiffness_multiplier,
    },
}
SIMULATION_APP = SimulationApp({"headless": ARGS.headless})


import numpy as np  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.util.debug_draw import _debug_draw  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from omni.physx.bindings._physx import ContactEventType  # noqa: E402
from pxr import (  # noqa: E402
    PhysxSchema,
    PhysicsSchemaTools,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)
from ik.kinematics_utils import (  # noqa: E402
    frame_point_meters_in_world_stage as lerobot_base_point_in_world,
    joint_limit_violations,
    make_transform,
    meters_to_stage,
    pose_residual,
    resolve_named_indices,
    rotation_matrix_rpy,
    rotation_matrix_wxyz,
    stage_to_meters,
    world_stage_point_in_frame_meters as world_point_in_lerobot_base,
)
from ik.execution_policy import (  # noqa: E402
    DescentContactDiagnostics,
    distributed_command_steps,
)
from ik.grasp_geometry import (  # noqa: E402
    generate_sphere_grasp_candidates,
    vertical_approach_waypoints,
)
from ik.tool_model import (  # noqa: E402
    FixedToolModel,
    fixed_tool_model_from_urdf,
)


class FixedToolPoint:
    """Runtime point backed by the URDF tool joint, not a USD marker prim."""

    def __init__(
        self,
        parent_link: XformPrim,
        tool_model: FixedToolModel,
        meters_per_unit: float,
    ) -> None:
        self.parent_link = parent_link
        self.tool_model = tool_model
        self.meters_per_unit = meters_per_unit

    def world_position_stage_units(self) -> np.ndarray:
        positions, orientations = self.parent_link.get_world_poses()
        parent_position_m = stage_to_meters(
            as_numpy(positions)[0],
            self.meters_per_unit,
        )
        parent_rotation = rotation_matrix_wxyz(as_numpy(orientations)[0])
        tool_position_m = (
            parent_position_m
            + parent_rotation @ self.tool_model.position_in_parent_m
        )
        return meters_to_stage(tool_position_m, self.meters_per_unit)


def json_compatible(value: object) -> object:
    """Convert NumPy, path, and nested values into strict JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def set_run_stage(stage: str) -> None:
    """Identify the stage that owns any subsequent run failure."""
    global CURRENT_RUN_STAGE
    CURRENT_RUN_STAGE = stage
    RUN_SUMMARY["current_stage"] = stage


def record_run_section(name: str, values: dict[str, object]) -> None:
    """Merge structured measurements into one top-level result section."""
    existing = RUN_SUMMARY.setdefault(name, {})
    if not isinstance(existing, dict):
        raise RuntimeError(f"Run-summary section {name!r} is not a mapping")
    existing.update(json_compatible(values))


def write_run_summary() -> None:
    """Atomically write the run summary when --result-json was requested."""
    if ARGS.result_json is None:
        return
    result_path = ARGS.result_json.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(json_compatible(RUN_SUMMARY), indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(result_path)
    print(f"Structured run result written to: {result_path}")


def run_lerobot_worker(request: dict[str, object]) -> dict[str, object]:
    """Run one request in the external LeRobot Python environment."""
    python_path = ARGS.lerobot_python.expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(
            f"LeRobot Python interpreter does not exist: {python_path}"
        )
    if not SO101_URDF_PATH.is_file():
        raise FileNotFoundError(f"SO-101 URDF does not exist: {SO101_URDF_PATH}")

    # SimulationApp sets launcher variables for Isaac's embedded Python 3.12.
    # They must not leak into the Conda Python 3.11 worker.
    worker_environment = os.environ.copy()
    worker_environment.pop("PYTHONHOME", None)
    worker_environment.pop("PYTHONPATH", None)
    worker_environment.pop("VIRTUAL_ENV", None)
    worker_environment["PATH"] = (
        f"{python_path.parent}{os.pathsep}"
        f"{worker_environment.get('PATH', '')}"
    )

    completed = subprocess.run(
        [str(python_path), str(Path(__file__).resolve()), "--lerobot-worker"],
        input=json.dumps(request, allow_nan=False),
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        env=worker_environment,
        check=False,
    )
    marker_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(LEROBOT_WORKER_MARKER)
    ]
    if completed.returncode != 0 or not marker_lines:
        raise RuntimeError(
            "LeRobot setup worker failed: "
            f"returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )

    return json.loads(marker_lines[-1][len(LEROBOT_WORKER_MARKER) :])


def initialize_lerobot() -> dict[str, object]:
    """Verify the external LeRobot runtime and SO-101 model are loadable."""
    setup = run_lerobot_worker({"operation": "setup"})
    print(
        "LeRobot IK environment ready: "
        f"lerobot={setup['lerobot_version']}, "
        f"placo={setup['placo_version']}"
    )
    print(f"SO-101 IK URDF: {setup['urdf_path']}")
    print(
        f"IK target frame: {setup['target_frame']}; "
        f"arm joints: {setup['joint_names']}"
    )
    return setup


def world_transform_meters(
    prim: XformPrim, meters_per_unit: float
) -> np.ndarray:
    positions, orientations = prim.get_world_poses()
    return make_transform(
        stage_to_meters(as_numpy(positions)[0], meters_per_unit),
        rotation_matrix_wxyz(as_numpy(orientations)[0]),
    )


def urdf_joint_origin(joint_name: str) -> np.ndarray:
    root = ET.parse(SO101_URDF_PATH).getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        origin = joint.find("origin")
        if origin is None:
            return np.eye(4, dtype=np.float64)
        position = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        return make_transform(position, rotation_matrix_rpy(rpy))
    raise RuntimeError(f"URDF joint does not exist: {joint_name}")


def usd_joint_zero_transform(stage: Usd.Stage, joint_name: str) -> np.ndarray:
    """Return the imported USD parent-to-child joint transform at q=0."""
    joint_path = f"{ROBOT_PRIM_PATH}/joints/{joint_name}"
    joint = UsdPhysics.RevoluteJoint(require_prim(stage, joint_path))

    def local_transform(position_attr: object, rotation_attr: object) -> np.ndarray:
        # These imported joint positions are authored in SI meters even though
        # the containing stage may use centimeters.
        position = np.asarray(position_attr.Get(), dtype=np.float64)
        quaternion = rotation_attr.Get()
        quaternion_wxyz = np.array(
            [quaternion.GetReal(), *quaternion.GetImaginary()],
            dtype=np.float64,
        )
        return make_transform(position, rotation_matrix_wxyz(quaternion_wxyz))

    parent_to_joint = local_transform(
        joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()
    )
    child_to_joint = local_transform(
        joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()
    )
    return parent_to_joint @ np.linalg.inv(child_to_joint)


def world_from_lerobot_base_transform(
    stage: Usd.Stage,
    base_link: XformPrim,
    meters_per_unit: float,
) -> np.ndarray:
    """Return the fixed transform from the LeRobot base into world meters."""
    # The imported USD base and the LeRobot URDF base use different coordinate
    # frames. Derive their fixed bridge from independently authored
    # shoulder-pan joint frames rather than fitting it from the end effector.
    usd_base_to_shoulder_zero = usd_joint_zero_transform(stage, "shoulder_pan")
    lerobot_base_to_shoulder_zero = urdf_joint_origin("shoulder_pan")
    usd_base_from_lerobot_base = (
        usd_base_to_shoulder_zero
        @ np.linalg.inv(lerobot_base_to_shoulder_zero)
    )
    world_from_usd_base = world_transform_meters(base_link, meters_per_unit)
    return world_from_usd_base @ usd_base_from_lerobot_base


def validate_tool_model_alignment(
    stage: Usd.Stage,
    robot: Articulation,
    base_link: XformPrim,
    gripper: XformPrim,
    tool_model: FixedToolModel,
    meters_per_unit: float,
) -> dict[str, float]:
    """Cross-check the Isaac tool pose against official-URDF LeRobot FK."""
    arm_indices = resolve_arm_dof_indices(robot)
    joint_positions = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    joints_deg = np.rad2deg(joint_positions[arm_indices])
    fk_result = run_lerobot_worker(
        {
            "operation": "forward_kinematics",
            "joints_deg": joints_deg.tolist(),
        }
    )
    lerobot_base_from_tool = np.asarray(fk_result["pose"], dtype=np.float64)
    world_from_lerobot_base = world_from_lerobot_base_transform(
        stage,
        base_link,
        meters_per_unit,
    )
    world_from_tool_fk = world_from_lerobot_base @ lerobot_base_from_tool
    world_from_gripper_isaac = world_transform_meters(
        gripper,
        meters_per_unit,
    )
    world_from_tool_isaac = (
        world_from_gripper_isaac @ tool_model.parent_from_tool
    )
    position_error_m, orientation_error_rad = pose_residual(
        world_from_tool_isaac,
        world_from_tool_fk,
    )
    metrics = {
        "fk_position_error_mm": position_error_m * 1000.0,
        "fk_orientation_error_deg": math.degrees(orientation_error_rad),
    }
    record_run_section("tool_model", metrics)
    print()
    print("Fixed tool-model validation:")
    print(
        "  Isaac vs LeRobot FK position error: "
        f"{metrics['fk_position_error_mm']:.3f} mm"
    )
    print(
        "  Isaac vs LeRobot FK orientation error: "
        f"{metrics['fk_orientation_error_deg']:.3f} deg"
    )
    if (
        position_error_m > TOOL_MODEL_POSITION_TOLERANCE_M
        or orientation_error_rad > TOOL_MODEL_ORIENTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(
            "Isaac and LeRobot disagree on the fixed tool pose: "
            f"position_error={position_error_m * 1000.0:.3f} mm, "
            f"orientation_error={math.degrees(orientation_error_rad):.3f} deg"
        )
    return metrics


def resolve_arm_dof_indices(robot: Articulation) -> list[int]:
    try:
        return resolve_named_indices(robot.dof_names, ARM_JOINT_NAMES)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def command_ik_seed_presets(robot: Articulation) -> None:
    """Place the wrist and gripper in the requested pre-IK configuration."""
    index_by_name = {str(name): index for index, name in enumerate(robot.dof_names)}
    required_names = (WRIST_PRESET_DOF_NAME, GRIPPER_PRESET_DOF_NAME)
    missing = [name for name in required_names if name not in index_by_name]
    if missing:
        raise RuntimeError(
            f"Missing pre-IK preset DOFs {missing}; actual names={robot.dof_names}"
        )

    preset_indices = [index_by_name[name] for name in required_names]
    preset_targets = np.array(
        [WRIST_PRESET_POSITION_RAD, GRIPPER_PRESET_POSITION_RAD],
        dtype=np.float64,
    )
    lower_limits, upper_limits = robot.get_dof_limits()
    lower_limits = as_numpy(lower_limits)[0]
    upper_limits = as_numpy(upper_limits)[0]
    for name, target, index in zip(
        required_names, preset_targets, preset_indices
    ):
        if target < lower_limits[index] or target > upper_limits[index]:
            raise RuntimeError(
                f"Pre-IK target for {name}={target:.6f} rad violates "
                f"[{lower_limits[index]:.6f}, {upper_limits[index]:.6f}]"
            )

    current_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    start_positions = current_all[preset_indices]
    print()
    print("Applying pre-IK wrist/gripper configuration:")
    print(f"  DOFs: {list(required_names)}")
    print(f"  Start positions (rad): {start_positions.tolist()}")
    print(f"  Target positions (rad): {preset_targets.tolist()}")

    for step in range(1, PRESET_COMMAND_STEPS + 1):
        fraction = step / PRESET_COMMAND_STEPS
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        command = start_positions + blend * (
            preset_targets - start_positions
        )
        robot.set_dof_position_targets(command, dof_indices=preset_indices)
        SIMULATION_APP.update()

    actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    print(
        "  Actual positions before IK (rad): "
        f"{actual_all[preset_indices].tolist()}"
    )


def calculate_position_only_ik(
    stage: Usd.Stage,
    robot: Articulation,
    base_link: XformPrim,
    fingertip_offset_stage_units: np.ndarray,
    target_world_stage_units: np.ndarray,
    meters_per_unit: float,
) -> dict[str, object]:
    """Calculate, but do not command, a fixed-fingertip position IK result."""
    arm_indices = resolve_arm_dof_indices(robot)
    all_joint_positions = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    seed_joints_rad = all_joint_positions[arm_indices]
    seed_joints_deg = np.rad2deg(seed_joints_rad)

    world_from_lerobot_base = world_from_lerobot_base_transform(
        stage,
        base_link,
        meters_per_unit,
    )
    target_world_m = stage_to_meters(
        target_world_stage_units,
        meters_per_unit,
    )
    target_lerobot_m = world_point_in_lerobot_base(
        target_world_stage_units,
        world_from_lerobot_base,
        meters_per_unit,
    )
    fingertip_offset_m = stage_to_meters(
        fingertip_offset_stage_units,
        meters_per_unit,
    )

    print()
    print("Position-only IK inputs:")
    print(f"  Joint order: {ARM_JOINT_NAMES}")
    print(f"  Seed joints (rad): {seed_joints_rad.tolist()}")
    print(f"  Seed joints (deg): {seed_joints_deg.tolist()}")
    print(f"  Fixed fingertip offset in gripper frame (m): {fingertip_offset_m.tolist()}")
    print(f"  Target point in world frame (m): {target_world_m.tolist()}")
    print(f"  Target point in LeRobot base frame (m): {target_lerobot_m.tolist()}")

    result = run_lerobot_worker(
        {
            "operation": "position_ik",
            "seed_joints_deg": seed_joints_deg.tolist(),
            "fingertip_offset_m": fingertip_offset_m.tolist(),
            "target_position_m": target_lerobot_m.tolist(),
            "max_iterations": IK_MAX_ITERATIONS,
            "position_tolerance_m": IK_POSITION_TOLERANCE_M,
        }
    )
    solved_joints_deg = np.asarray(
        result["solved_joints_deg"], dtype=np.float64
    )
    solved_joints_rad = np.deg2rad(solved_joints_deg)
    if solved_joints_deg.shape != (len(ARM_JOINT_NAMES),):
        raise RuntimeError(
            f"LeRobot returned an invalid joint vector: {solved_joints_deg}"
        )
    if not np.all(np.isfinite(solved_joints_deg)):
        raise RuntimeError("LeRobot returned NaN or infinite joint values")

    lower_limits, upper_limits = robot.get_dof_limits()
    lower_limits = as_numpy(lower_limits)[0]
    upper_limits = as_numpy(upper_limits)[0]
    violations = joint_limit_violations(
        solved_joints_rad,
        lower_limits[arm_indices],
        upper_limits[arm_indices],
        ARM_JOINT_NAMES,
    )
    if violations:
        raise RuntimeError(f"IK solution violates joint limits: {violations}")

    print()
    print("Position-only IK result:")
    print(f"  Status: {result['status']}")
    print(f"  Solved joints (deg): {solved_joints_deg.tolist()}")
    print(f"  Solved joints (rad): {solved_joints_rad.tolist()}")
    print(f"  Solved fingertip position in LeRobot base (m): {result['solved_position_m']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Position error: {float(result['position_error_m']) * 1000.0:.3f} mm")
    print(
        "  Task weights: "
        f"position={result['position_weight']}, "
        f"orientation={result['orientation_weight']}"
    )
    if result["status"] != "converged":
        raise RuntimeError(
            "Position-only IK did not converge within "
            f"{IK_MAX_ITERATIONS} iterations"
        )
    return result


def calculate_top_down_candidates(
    stage: Usd.Stage,
    robot: Articulation,
    base_link: XformPrim,
    fingertip_offset_stage_units: np.ndarray,
    grasp_candidates_world: list[dict[str, object]],
    meters_per_unit: float,
) -> dict[str, object]:
    """Evaluate constrained grasp-point and approach-axis candidates."""
    arm_indices = resolve_arm_dof_indices(robot)
    all_joint_positions = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    seed_joints_deg = np.rad2deg(all_joint_positions[arm_indices])
    lower_limits, upper_limits = robot.get_dof_limits()
    lower_limits_deg = np.rad2deg(as_numpy(lower_limits)[0, arm_indices])
    upper_limits_deg = np.rad2deg(as_numpy(upper_limits)[0, arm_indices])
    world_from_lerobot_base = world_from_lerobot_base_transform(
        stage,
        base_link,
        meters_per_unit,
    )
    candidate_waypoint_positions_m = []
    candidate_target_axes_lerobot = []
    candidate_target_closing_axes_lerobot = []
    candidate_metadata = []
    for candidate in grasp_candidates_world:
        waypoints = candidate["waypoints_world_stage_units"]
        assert isinstance(waypoints, list)
        candidate_waypoint_positions_m.append(
            [
                world_point_in_lerobot_base(
                    waypoint,
                    world_from_lerobot_base,
                    meters_per_unit,
                ).tolist()
                for waypoint in waypoints
            ]
        )
        target_axis_world = np.asarray(
            candidate["target_axis_world"], dtype=np.float64
        )
        candidate_target_axes_lerobot.append(
            (world_from_lerobot_base[:3, :3].T @ target_axis_world).tolist()
        )
        target_closing_axis_world = np.asarray(
            candidate["target_closing_axis_world"], dtype=np.float64
        )
        candidate_target_closing_axes_lerobot.append(
            (
                world_from_lerobot_base[:3, :3].T
                @ target_closing_axis_world
            ).tolist()
        )
        candidate_metadata.append(
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "target_kind",
                    "surface_offset_deg",
                    "approach_tilt_deg",
                    "surface_clearance_m",
                )
            }
        )
    # The official gripper_frame_link +Z axis is the tool approach axis.
    # Align it with each sampled approach axis while intentionally leaving yaw
    # about that axis unconstrained; a 5-DOF arm can satisfy these five task
    # dimensions.
    tool_axis_frame = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fingertip_offset_m = stage_to_meters(
        fingertip_offset_stage_units,
        meters_per_unit,
    )
    print()
    print("Constrained top-down IK inputs:")
    print(f"  Position-only seed joints (deg): {seed_joints_deg.tolist()}")
    print(f"  Candidate count: {len(grasp_candidates_world)}")
    print(
        "  Surface offsets (deg): "
        f"{list(GRASP_SURFACE_OFFSETS_DEG)}"
    )
    print(f"  Approach tilts (deg): {list(GRASP_APPROACH_TILTS_DEG)}")
    print("  Constraints: TCP position (hard), tool +Z aligned to candidate axis")
    print("  Rotation about tool approach axis: unconstrained")
    print(
        "  Residual gates: "
        f"position <= {ARGS.top_down_position_tolerance_mm:.3f} mm, "
        f"axis <= {ARGS.top_down_axis_tolerance_deg:.3f} deg"
    )
    result = run_lerobot_worker(
        {
            "operation": "grasp_candidates",
            "seed_joints_deg": seed_joints_deg.tolist(),
            "fingertip_offset_m": fingertip_offset_m.tolist(),
            "candidate_waypoint_positions_m": candidate_waypoint_positions_m,
            "candidate_target_axes_world": candidate_target_axes_lerobot,
            "candidate_target_closing_axes_world": (
                candidate_target_closing_axes_lerobot
            ),
            "candidate_metadata": candidate_metadata,
            "tool_axis_frame": tool_axis_frame.tolist(),
            "tool_closing_axis_frame": [-1.0, 0.0, 0.0],
            "closing_axis_weight": IK_CLOSING_AXIS_WEIGHT,
            "closing_axis_tolerance_rad": math.radians(
                DEFAULT_CLOSING_AXIS_TOLERANCE_DEG
            ),
            "lower_joint_limits_deg": lower_limits_deg.tolist(),
            "upper_joint_limits_deg": upper_limits_deg.tolist(),
            "max_iterations": IK_MAX_ITERATIONS,
            "position_tolerance_m": (
                ARGS.top_down_position_tolerance_mm / 1000.0
            ),
            "axis_tolerance_rad": math.radians(
                ARGS.top_down_axis_tolerance_deg
            ),
            "seed_regularization_weight": IK_SEED_REGULARIZATION_WEIGHT,
        }
    )
    paths: list[object] = []
    candidates = result.get("candidates")
    if isinstance(candidates, list):
        paths.extend(candidates)
    for path in paths:
        if not isinstance(path, dict) or path.get("status") != "valid":
            continue
        solved_positions_m = np.asarray(
            path.get("waypoint_solved_positions_m"),
            dtype=np.float64,
        )
        waypoint_targets_m = np.asarray(
            path.get("waypoint_target_positions_m"), dtype=np.float64
        )
        expected_shape = (len(candidate_waypoint_positions_m[0]), 3)
        if (
            solved_positions_m.shape != expected_shape
            or not np.all(np.isfinite(solved_positions_m))
            or waypoint_targets_m.shape != expected_shape
            or not np.all(np.isfinite(waypoint_targets_m))
        ):
            raise RuntimeError(
                "LeRobot candidate is missing valid solved waypoint positions"
            )
        path["waypoint_solved_positions_world_stage_units"] = [
            lerobot_base_point_in_world(
                solved_position_m,
                world_from_lerobot_base,
                meters_per_unit,
            ).tolist()
            for solved_position_m in solved_positions_m
        ]
        path["waypoint_target_positions_world_stage_units"] = [
            lerobot_base_point_in_world(
                target_position_m,
                world_from_lerobot_base,
                meters_per_unit,
            ).tolist()
            for target_position_m in waypoint_targets_m
        ]
        solved_rotations = np.asarray(
            path.get("waypoint_solved_rotations"), dtype=np.float64
        )
        if solved_rotations.shape != (expected_shape[0], 3, 3):
            raise RuntimeError(
                "LeRobot candidate is missing solved waypoint rotations"
            )
        candidate_id = path.get("candidate_id")
        source_candidate = next(
            (
                candidate
                for candidate in grasp_candidates_world
                if candidate.get("candidate_id") == candidate_id
            ),
            None,
        )
        if source_candidate is None:
            raise RuntimeError(f"Missing source geometry for {candidate_id}")
        desired_closing_world = normalized_vector(
            np.asarray(
                source_candidate["target_closing_axis_world"],
                dtype=np.float64,
            ),
            "Desired candidate closing axis",
        )
        closing_axis_frame = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        solved_closing_axes_world = np.asarray(
            [
                world_from_lerobot_base[:3, :3]
                @ solved_rotation
                @ closing_axis_frame
                for solved_rotation in solved_rotations
            ]
        )
        closing_errors_rad = [
            math.acos(
                float(
                    np.clip(
                        np.dot(
                            normalized_vector(axis, "Solved closing axis"),
                            desired_closing_world,
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
            for axis in solved_closing_axes_world
        ]
        path["waypoint_solved_closing_axes_world"] = (
            solved_closing_axes_world.tolist()
        )
        path["waypoint_closing_axis_errors_rad"] = closing_errors_rad
        path["terminal_closing_axis_error_rad"] = closing_errors_rad[-1]
        if closing_errors_rad[-1] > math.radians(
            DEFAULT_CLOSING_AXIS_TOLERANCE_DEG
        ):
            path["status"] = "rejected"
            path["rejection_stage"] = "path"
            path["reason"] = (
                "Jaw-closing axis residual gate failed: "
                f"error={math.degrees(closing_errors_rad[-1]):.3f} deg "
                f"(limit={DEFAULT_CLOSING_AXIS_TOLERANCE_DEG:.3f} deg)"
            )
    return result


def select_top_down_path(
    candidate_result: dict[str, object],
) -> dict[str, object]:
    """Return the worker's first valid ranked path."""
    candidates = candidate_result.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("Top-down worker did not return a candidate list")

    eligible = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("status") == "valid"
    ]
    if eligible:
        selected_candidate_id = candidate_result.get("selected_candidate_id")
        selected = next(
            (
                candidate
                for candidate in eligible
                if candidate.get("candidate_id") == selected_candidate_id
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(
                "Top-down worker selected candidate is missing or invalid: "
                f"selected_candidate_id={selected_candidate_id!r}"
            )
        return {
            "mode": "top_down_constrained",
            "selection_reason": "endpoint_ranked_first_valid_path",
            "path": selected,
        }

    rejected_candidates = sum(
        1
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("status") == "rejected"
    )
    rejection_reasons = [
        str(candidate.get("reason"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("status") == "rejected"
    ]
    raise RuntimeError(
        "No sampled grasp candidate passes the position, approach-axis, "
        "jaw-axis, and joint-limit gates before motion: "
        f"rejected_candidates={rejected_candidates}, "
        f"reasons={rejection_reasons}"
    )


def print_top_down_candidate_selection(
    candidate_result: dict[str, object],
    selection: dict[str, object],
) -> None:
    """Print compact branch diagnostics and the path selected for Step 4."""
    candidates = candidate_result["candidates"]
    assert isinstance(candidates, list)
    print()
    print("Constrained top-down IK result:")
    for index, candidate in enumerate(candidates, start=1):
        assert isinstance(candidate, dict)
        status = candidate["status"]
        if status == "rejected":
            print(
                f"  #{index:02d}: id={candidate['candidate_id']}, "
                f"rejected-at={candidate.get('rejection_stage', 'unknown')}, "
                f"reason={candidate['reason']}"
            )
            continue
        if status == "skipped":
            print(
                f"  #{index:02d}: id={candidate['candidate_id']}, "
                f"status=skipped, rank={candidate.get('endpoint_rank')}, "
                f"reason={candidate.get('skip_reason')}"
            )
            continue
        print(
            f"  #{index:02d}: id={candidate['candidate_id']}, "
            "status=valid, "
            f"terminal-position={float(candidate['terminal_position_error_m']) * 1000.0:.3f} mm, "
            f"terminal-axis={math.degrees(float(candidate['terminal_axis_error_rad'])):.3f} deg, "
            f"max-position={float(candidate['max_position_error_m']) * 1000.0:.3f} mm, "
            f"max-axis={math.degrees(float(candidate['max_axis_error_rad'])):.3f} deg, "
            f"closing-axis={math.degrees(float(candidate['terminal_closing_axis_error_rad'])):.3f} deg, "
            f"travel={float(candidate['total_joint_travel_deg']):.3f} deg"
        )

    planning_metrics = candidate_result.get("planning_metrics")
    if isinstance(planning_metrics, dict):
        print(
            "  Planning metrics: "
            f"strategy={planning_metrics.get('strategy')}, "
            f"endpoints={planning_metrics.get('endpoint_feasible')}/"
            f"{planning_metrics.get('endpoint_candidates')}, "
            f"full-paths={planning_metrics.get('full_paths_evaluated')}, "
            f"skipped={planning_metrics.get('full_paths_skipped')}, "
            f"solver-calls={planning_metrics.get('solver_calls')}, "
            f"contexts={planning_metrics.get('solver_context_builds')}, "
            f"duration={float(planning_metrics.get('total_planning_duration_seconds', 0.0)):.3f} s"
        )

    fallback = candidate_result.get("position_only_fallback")
    if isinstance(fallback, dict):
        if fallback["status"] == "rejected":
            print(f"  Position-only fallback: rejected={fallback['reason']}")
        else:
            print(
                "  Position-only fallback: status=valid, "
                f"max-position={float(fallback['max_position_error_m']) * 1000.0:.3f} mm"
            )

    selected_path = selection["path"]
    assert isinstance(selected_path, dict)
    if selection["mode"] == "top_down_constrained":
        print(
            "[CONSTRAINED TOP-DOWN PATH SELECTED] "
            "selection=endpoint-ranked-first-valid-path, "
            f"id={selected_path['candidate_id']}, "
            f"terminal-position={float(selected_path['terminal_position_error_m']) * 1000.0:.3f} mm, "
            f"terminal-axis={math.degrees(float(selected_path['terminal_axis_error_rad'])):.3f} deg"
        )


def draw_ik_debug_overlay(
    sphere_center_world: np.ndarray,
    sphere_grasp_points_world: tuple[np.ndarray, np.ndarray],
    selected_sphere_grasp_point_world: np.ndarray,
    contact_point_world: np.ndarray,
    base_heading_world: np.ndarray,
    top_down_waypoints_world: list[np.ndarray],
    target_approach_axis_world: np.ndarray,
    local_tool_approach: np.ndarray,
    local_jaw_closing: np.ndarray,
    yaw_candidates_world: list[tuple[float, np.ndarray]],
    candidate_result: dict[str, object],
    selection: dict[str, object],
    meters_per_unit: float,
) -> None:
    """Draw the grasp interpretation, target path, and requested tool pose."""
    if not ARGS.ik_debug_overlay:
        return
    if len(top_down_waypoints_world) < 2:
        raise RuntimeError("IK debug overlay requires at least two waypoints")

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    draw.clear_lines()
    points: list[list[float]] = []
    point_colors: list[tuple[float, float, float, float]] = []
    point_sizes: list[float] = []
    line_starts: list[list[float]] = []
    line_ends: list[list[float]] = []
    line_colors: list[tuple[float, float, float, float]] = []
    line_sizes: list[float] = []

    def add_point(
        point: np.ndarray,
        color: tuple[float, float, float, float],
        size: float,
    ) -> None:
        points.append(np.asarray(point, dtype=np.float64).tolist())
        point_colors.append(color)
        point_sizes.append(size)

    def add_line(
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float, float],
        size: float,
    ) -> None:
        line_starts.append(np.asarray(start, dtype=np.float64).tolist())
        line_ends.append(np.asarray(end, dtype=np.float64).tolist())
        line_colors.append(color)
        line_sizes.append(size)

    sphere_center_world = np.asarray(sphere_center_world, dtype=np.float64)
    selected_sphere_grasp_point_world = np.asarray(
        selected_sphere_grasp_point_world,
        dtype=np.float64,
    )
    contact_point_world = np.asarray(contact_point_world, dtype=np.float64)
    top_down_waypoints = [
        np.asarray(waypoint, dtype=np.float64)
        for waypoint in top_down_waypoints_world
    ]
    local_approach_direction = normalized_vector(
        local_tool_approach,
        "IK debug local tool approach",
    )
    local_jaw_direction = unit_perpendicular_to(
        local_jaw_closing,
        local_approach_direction,
        "IK debug local jaw closing",
    )
    axis_length = DEBUG_IK_AXIS_LENGTH_MM / 1000.0 / meters_per_unit
    candidate_axis_length = (
        DEBUG_IK_CANDIDATE_AXIS_LENGTH_MM / 1000.0 / meters_per_unit
    )

    yellow = (1.0, 1.0, 0.0, 1.0)
    cyan = (0.0, 1.0, 1.0, 1.0)
    red = (1.0, 0.0, 0.0, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)
    green = (0.0, 1.0, 0.0, 1.0)
    blue = (0.2, 0.5, 1.0, 1.0)
    purple = (0.55, 0.2, 1.0, 1.0)
    magenta = (1.0, 0.0, 1.0, 1.0)
    orange = (1.0, 0.55, 0.0, 1.0)
    gray = (0.35, 0.35, 0.35, 1.0)

    add_point(sphere_center_world, yellow, 18.0)
    add_point(contact_point_world, cyan, 16.0)
    for grasp_point in sphere_grasp_points_world:
        grasp_point = np.asarray(grasp_point, dtype=np.float64)
        is_selected = np.allclose(
            grasp_point,
            selected_sphere_grasp_point_world,
            rtol=0.0,
            atol=1e-9,
        )
        add_point(grasp_point, red if is_selected else white, 18.0)
        add_line(
            sphere_center_world,
            grasp_point,
            red if is_selected else white,
            2.0,
        )

    # The lateral grasp points are chosen perpendicular to this horizontal
    # base-facing direction, so draw it at the sphere for direct inspection.
    heading_direction = normalized_vector(
        base_heading_world,
        "IK debug base heading",
    )
    add_line(
        sphere_center_world,
        sphere_center_world + axis_length * heading_direction,
        cyan,
        3.0,
    )

    for index, waypoint in enumerate(top_down_waypoints):
        add_point(waypoint, green if index else blue, 11.0 if index else 18.0)
        if index:
            add_line(top_down_waypoints[index - 1], waypoint, green, 3.0)
    add_point(selected_sphere_grasp_point_world, red, 22.0)

    selected_path = selection.get("path")
    predicted_waypoints = None
    if isinstance(selected_path, dict):
        predicted_waypoints = selected_path.get(
            "waypoint_solved_positions_world_stage_units"
        )
    if predicted_waypoints is not None:
        predicted_waypoints_array = np.asarray(
            predicted_waypoints,
            dtype=np.float64,
        )
        if predicted_waypoints_array.shape != (len(top_down_waypoints), 3) or not np.all(
            np.isfinite(predicted_waypoints_array)
        ):
            raise RuntimeError(
                "Selected IK path has invalid solved waypoint positions for debug draw"
            )
        for index, predicted_waypoint in enumerate(predicted_waypoints_array):
            add_point(predicted_waypoint, purple, 14.0)
            if index:
                add_line(
                    predicted_waypoints_array[index - 1],
                    predicted_waypoint,
                    purple,
                    3.0,
                )

    candidate_status_by_yaw: dict[float, str] = {}
    candidates = candidate_result.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if "yaw_offset_deg" not in candidate:
                continue
            yaw_offset_deg = float(candidate["yaw_offset_deg"])
            if candidate.get("status") == "valid":
                candidate_status_by_yaw[yaw_offset_deg] = "valid"
            else:
                candidate_status_by_yaw.setdefault(yaw_offset_deg, "rejected")

    selected_yaw_deg: float | None = None
    if (
        selection.get("mode") == "top_down_constrained"
        and isinstance(selected_path, dict)
        and "yaw_offset_deg" in selected_path
    ):
        selected_yaw_deg = float(selected_path["yaw_offset_deg"])
    selected_rotation: np.ndarray | None = None
    hover_waypoint = top_down_waypoints[0]
    for yaw_offset_deg, rotation in yaw_candidates_world:
        rotation = np.asarray(rotation, dtype=np.float64)
        desired_closing_direction = normalized_vector(
            rotation @ local_jaw_direction,
            f"IK debug yaw {yaw_offset_deg:.1f} closing direction",
        )
        is_selected = (
            selected_yaw_deg is not None
            and math.isclose(yaw_offset_deg, selected_yaw_deg, abs_tol=1e-9)
        )
        if is_selected:
            color, width = magenta, 5.0
            selected_rotation = rotation
        elif candidate_status_by_yaw.get(yaw_offset_deg) == "valid":
            color, width = orange, 3.0
        else:
            color, width = gray, 1.0
        add_line(
            hover_waypoint - 0.5 * candidate_axis_length * desired_closing_direction,
            hover_waypoint + 0.5 * candidate_axis_length * desired_closing_direction,
            color,
            width,
        )

    # BLUE is the selected constrained tool approach axis. Rotation about it is free,
    # so no selected jaw-closing/yaw axis exists for the constrained solver.
    target_approach_direction = normalized_vector(
        target_approach_axis_world,
        "IK debug selected target approach axis",
    )
    add_line(
        hover_waypoint,
        hover_waypoint + axis_length * target_approach_direction,
        blue,
        5.0,
    )
    if selected_rotation is not None:
        selected_approach_direction = normalized_vector(
            selected_rotation @ local_approach_direction,
            "IK debug selected approach direction",
        )
        selected_closing_direction = normalized_vector(
            selected_rotation @ local_jaw_direction,
            "IK debug selected closing direction",
        )
        add_line(
            hover_waypoint,
            hover_waypoint + axis_length * selected_approach_direction,
            blue,
            5.0,
        )
        add_line(
            hover_waypoint - 0.5 * axis_length * selected_closing_direction,
            hover_waypoint + 0.5 * axis_length * selected_closing_direction,
            magenta,
            5.0,
        )

    draw.draw_points(points, point_colors, point_sizes)
    draw.draw_lines(line_starts, line_ends, line_colors, line_sizes)

    print()
    print("IK debug overlay legend (viewport world coordinates):")
    print("  YELLOW sphere center; WHITE/RED alternate/selected surface grasp points")
    print("  CYAN initial authored fixed-finger contact and base-facing heading")
    print("  BLUE hover target and selected tool approach axis")
    print("  GREEN vertical desired contact-point path from hover to final target")
    print("  PURPLE forward-kinematics contact path predicted by the selected IK joints")
    print("  MAGENTA selected yaw jaw-closing axis; ORANGE valid nonselected yaw; GRAY rejected yaw")
    for index, waypoint in enumerate(top_down_waypoints):
        print_point(
            f"  Overlay waypoint {index} ({'hover' if index == 0 else 'descent'})",
            waypoint * meters_per_unit,
        )
    if predicted_waypoints is not None:
        for index, predicted_waypoint in enumerate(predicted_waypoints_array):
            print_point(
                f"  Selected IK FK waypoint {index}",
                predicted_waypoint * meters_per_unit,
            )


def draw_ik_execution_residual(
    actual_contact_world: np.ndarray,
    target_contact_world: np.ndarray,
) -> None:
    """Add a persistent actual-versus-target contact residual to the overlay."""
    if not ARGS.ik_debug_overlay:
        return
    draw = _debug_draw.acquire_debug_draw_interface()
    actual_contact_world = np.asarray(actual_contact_world, dtype=np.float64)
    target_contact_world = np.asarray(target_contact_world, dtype=np.float64)
    magenta = (1.0, 0.0, 1.0, 1.0)
    draw.draw_points([actual_contact_world.tolist()], [magenta], [24.0])
    draw.draw_lines(
        [target_contact_world.tolist()],
        [actual_contact_world.tolist()],
        [magenta],
        [5.0],
    )
    print("  MAGENTA point/line now also shows actual final contact and residual")


class SphereFingerContactTracker:
    """Track sphere/finger and whole-robot contacts from PhysX callbacks."""

    def __init__(self, sphere_prim: Usd.Prim) -> None:
        self._sphere_path = SPHERE_PRIM_PATH
        self._finger_paths = {
            FIXED_FINGER_PRIM_PATH: "fixed",
            f"{ROBOT_PRIM_PATH}/jaw": "moving",
        }
        self._active_fingers: set[str] = set()
        self._active_robot_contact_pairs: set[tuple[str, str]] = set()
        self._robot_body_paths: set[str] = {
            f"{ROBOT_PRIM_PATH}/{link_name}"
            for link_name in (
                "base",
                "shoulder",
                "upper_arm",
                "lower_arm",
                "wrist",
                "gripper",
                "jaw",
            )
        }

        # This applies only to the opened stage in this process. Reporting on
        # the sphere identifies every robot link that touches the target while
        # avoiding a destructive PhysX articulation rebuild.
        contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(sphere_prim)
        contact_report_api.CreateThresholdAttr().Set(0.0)
        self._subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report
            )
        )
        print(
            "Sphere contact reporting enabled: "
            f"sphere={self._sphere_path}, "
            f"fixed={FIXED_FINGER_PRIM_PATH}, "
            f"moving={ROBOT_PRIM_PATH}/jaw"
        )
        print(
            "Sphere-to-robot contact classification enabled for: "
            f"{sorted(self._robot_body_paths)}"
        )

    def _tracked_finger(self, actor0: str, actor1: str) -> str | None:
        if actor0 == self._sphere_path:
            return self._finger_paths.get(actor1)
        if actor1 == self._sphere_path:
            return self._finger_paths.get(actor0)
        return None

    def _on_contact_report(self, contact_headers: object, _: object) -> None:
        for header in contact_headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            pair = tuple(sorted((actor0, actor1)))
            involves_robot = bool(
                {actor0, actor1}.intersection(self._robot_body_paths)
            )
            if involves_robot:
                if header.type == ContactEventType.CONTACT_LOST:
                    self._active_robot_contact_pairs.discard(pair)
                elif header.type in (
                    ContactEventType.CONTACT_FOUND,
                    ContactEventType.CONTACT_PERSIST,
                ):
                    self._active_robot_contact_pairs.add(pair)
            finger = self._tracked_finger(actor0, actor1)
            if finger is None:
                continue

            if header.type == ContactEventType.CONTACT_LOST:
                if finger in self._active_fingers:
                    self._active_fingers.remove(finger)
                    print(f"Sphere {finger}-finger contact lost")
                continue

            if header.type in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            ) and finger not in self._active_fingers:
                self._active_fingers.add(finger)
                print(f"Sphere {finger}-finger contact detected")

    @property
    def fixed_finger_in_contact(self) -> bool:
        return "fixed" in self._active_fingers

    @property
    def moving_finger_in_contact(self) -> bool:
        return "moving" in self._active_fingers

    @property
    def active_robot_contact_pairs(self) -> set[tuple[str, str]]:
        """Return a copy of active robot self/object/environment contacts."""
        return set(self._active_robot_contact_pairs)

    def print_status(self) -> None:
        print(
            "Sphere finger-contact status: "
            f"fixed={self.fixed_finger_in_contact}, "
            f"moving={self.moving_finger_in_contact}"
        )
        print(
            "Active robot contact pairs: "
            f"{sorted(self._active_robot_contact_pairs)}"
        )


def apply_selected_approach_path(
    robot: Articulation,
    contact_point: FixedToolPoint,
    contact_tracker: SphereFingerContactTracker,
    waypoint_targets_world_stage_units: list[np.ndarray],
    meters_per_unit: float,
    selection: dict[str, object],
) -> np.ndarray:
    """Command the full selected hover-to-contact trajectory."""
    arm_indices = resolve_arm_dof_indices(robot)
    initial_position_iterations, initial_velocity_iterations = (
        robot.get_solver_iteration_counts()
    )
    initial_self_collisions_enabled = robot.get_enabled_self_collisions()
    if ARGS.disable_self_collisions_diagnostic:
        robot.set_enabled_self_collisions([False])
    effective_self_collisions_enabled = robot.get_enabled_self_collisions()
    robot.set_solver_iteration_counts(
        [APPROACH_SOLVER_POSITION_ITERATIONS],
        [APPROACH_SOLVER_VELOCITY_ITERATIONS],
    )
    effective_position_iterations, effective_velocity_iterations = (
        robot.get_solver_iteration_counts()
    )
    initial_controller_kps, initial_controller_kds = robot.get_dof_gains()
    initial_arm_stiffnesses = as_numpy(initial_controller_kps)[0, arm_indices]
    initial_arm_dampings = as_numpy(initial_controller_kds)[0, arm_indices]
    if (
        not math.isclose(ARGS.arm_stiffness_multiplier, 1.0)
        or not math.isclose(ARGS.arm_damping_multiplier, 1.0)
    ):
        robot.set_dof_gains(
            stiffnesses=(
                initial_arm_stiffnesses * ARGS.arm_stiffness_multiplier
            ).tolist(),
            dampings=(
                initial_arm_dampings * ARGS.arm_damping_multiplier
            ).tolist(),
            dof_indices=arm_indices,
        )
    controller_kps, controller_kds = robot.get_dof_gains()
    controller_max_efforts = robot.get_dof_max_efforts()
    record_run_section(
        "approach_controller",
        {
            "arm_joint_names": ARM_JOINT_NAMES,
            "initial_position_gains": initial_arm_stiffnesses,
            "position_gains": as_numpy(controller_kps)[0, arm_indices],
            "stiffness_multiplier": ARGS.arm_stiffness_multiplier,
            "initial_velocity_gains": initial_arm_dampings,
            "velocity_gains": as_numpy(controller_kds)[0, arm_indices],
            "damping_multiplier": ARGS.arm_damping_multiplier,
            "max_efforts": as_numpy(controller_max_efforts)[0, arm_indices],
            "joint_tracking_tolerance_rad": (
                APPROACH_JOINT_TRACKING_TOLERANCE_RAD
            ),
            "tcp_tracking_tolerance_m": APPROACH_TCP_TRACKING_TOLERANCE_M,
            "stable_frames_required": APPROACH_TRACKING_STABLE_FRAMES,
            "maximum_settle_steps": APPROACH_TRACKING_SETTLE_STEPS,
            "initial_solver_position_iterations": as_numpy(
                initial_position_iterations
            ),
            "initial_solver_velocity_iterations": as_numpy(
                initial_velocity_iterations
            ),
            "effective_solver_position_iterations": as_numpy(
                effective_position_iterations
            ),
            "effective_solver_velocity_iterations": as_numpy(
                effective_velocity_iterations
            ),
            "initial_self_collisions_enabled": as_numpy(
                initial_self_collisions_enabled
            ),
            "effective_self_collisions_enabled": as_numpy(
                effective_self_collisions_enabled
            ),
            "self_collisions_diagnostic_override": (
                ARGS.disable_self_collisions_diagnostic
            ),
        },
    )
    selected_path = selection.get("path")
    if not isinstance(selected_path, dict):
        raise RuntimeError("Selected top-down path is missing its trajectory")
    waypoint_joints_deg = np.asarray(
        selected_path.get("waypoint_joints_deg"),
        dtype=np.float64,
    )
    waypoint_targets_world_m = np.asarray(
        waypoint_targets_world_stage_units,
        dtype=np.float64,
    ) * meters_per_unit
    expected_shape = (len(waypoint_targets_world_m), len(ARM_JOINT_NAMES))
    if (
        len(waypoint_targets_world_m) < 2
        or waypoint_joints_deg.shape != expected_shape
        or not np.all(np.isfinite(waypoint_joints_deg))
        or not np.all(np.isfinite(waypoint_targets_world_m))
    ):
        raise RuntimeError(
            "Selected approach path has invalid waypoint positions or joint vectors"
        )
    waypoint_joints_rad = np.deg2rad(waypoint_joints_deg)
    lower_limits, upper_limits = robot.get_dof_limits()
    lower_arm_limits = as_numpy(lower_limits)[0, arm_indices]
    upper_arm_limits = as_numpy(upper_limits)[0, arm_indices]
    violating_waypoints = [
        waypoint_index
        for waypoint_index, waypoint_joints in enumerate(waypoint_joints_rad)
        if joint_limit_violations(
            waypoint_joints,
            lower_arm_limits,
            upper_arm_limits,
            ARM_JOINT_NAMES,
        )
    ]
    if violating_waypoints:
        raise RuntimeError(
            "Selected approach path violates Isaac arm joint limits at "
            f"waypoints {violating_waypoints}"
        )

    descent_steps = distributed_command_steps(
        IK_COMMAND_STEPS,
        len(waypoint_joints_rad) - 1,
    )

    def live_arm_joints() -> np.ndarray:
        positions = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
        return positions[arm_indices].copy()

    def contact_error_m(target_world_m: np.ndarray) -> tuple[np.ndarray, float]:
        actual_contact_m = stage_to_meters(
            contact_point.world_position_stage_units(),
            meters_per_unit,
        )
        return actual_contact_m, float(
            np.linalg.norm(actual_contact_m - target_world_m)
        )

    baseline_contact_pairs = contact_tracker.active_robot_contact_pairs

    def unexpected_contact_pairs(
        *,
        allow_sphere_fingers: bool = False,
    ) -> list[tuple[str, str]]:
        pairs = contact_tracker.active_robot_contact_pairs - baseline_contact_pairs
        if allow_sphere_fingers:
            permitted = {
                tuple(sorted((SPHERE_PRIM_PATH, FIXED_FINGER_PRIM_PATH))),
                tuple(sorted((SPHERE_PRIM_PATH, MOVING_JAW_PRIM_PATH))),
            }
            pairs -= permitted
        return sorted(pairs)

    def reject_unexpected_contacts(
        *,
        phase: str,
        command_step: int,
        target_world_m: np.ndarray,
        allow_sphere_fingers: bool = False,
    ) -> None:
        pairs = unexpected_contact_pairs(
            allow_sphere_fingers=allow_sphere_fingers
        )
        if not pairs:
            return
        actual_contact_m, tcp_error_m = contact_error_m(target_world_m)
        record_run_section(
            "approach_execution",
            {
                "status": "rejected_collision",
                "failed_phase": phase,
                "failed_command_step": command_step,
                "unexpected_contact_pairs": pairs,
                "baseline_contact_pairs": sorted(baseline_contact_pairs),
                "tcp_tracking_error_mm": tcp_error_m * 1000.0,
                "actual_fingertip_world_m": actual_contact_m,
            },
        )
        raise RuntimeError(
            f"Unexpected contact during {phase} at command step "
            f"{command_step}: {pairs}"
        )

    def wait_for_tracking(
        target_joints_rad: np.ndarray,
        target_world_m: np.ndarray,
        *,
        phase: str,
        allow_sphere_fingers: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Hold a target until both joint and TCP residuals are stable."""
        stable_frames = 0
        tracking_trace: list[dict[str, object]] = []
        actual_joints_rad = live_arm_joints()
        actual_contact_m, tcp_error_m = contact_error_m(target_world_m)
        for settle_step in range(APPROACH_TRACKING_SETTLE_STEPS + 1):
            joint_errors_rad = actual_joints_rad - target_joints_rad
            if settle_step % 30 == 0:
                velocities = as_numpy(robot.get_dof_velocities())[0].astype(
                    np.float64
                )[arm_indices]
                tracking_trace.append(
                    {
                        "step": settle_step,
                        "joint_errors_rad": joint_errors_rad.copy(),
                        "joint_velocities_rad_s": velocities,
                        "tcp_error_mm": tcp_error_m * 1000.0,
                    }
                )
            within_gate = (
                float(np.max(np.abs(joint_errors_rad)))
                <= APPROACH_JOINT_TRACKING_TOLERANCE_RAD
                and tcp_error_m <= APPROACH_TCP_TRACKING_TOLERANCE_M
            )
            stable_frames = stable_frames + 1 if within_gate else 0
            if stable_frames >= APPROACH_TRACKING_STABLE_FRAMES:
                return (
                    actual_joints_rad,
                    actual_contact_m,
                    tcp_error_m,
                    settle_step,
                )
            if settle_step == APPROACH_TRACKING_SETTLE_STEPS:
                break
            robot.set_dof_position_targets(target_joints_rad, dof_indices=arm_indices)
            SIMULATION_APP.update()
            reject_unexpected_contacts(
                phase=phase,
                command_step=settle_step + 1,
                target_world_m=target_world_m,
                allow_sphere_fingers=allow_sphere_fingers,
            )
            actual_joints_rad = live_arm_joints()
            actual_contact_m, tcp_error_m = contact_error_m(target_world_m)
        joint_errors_rad = actual_joints_rad - target_joints_rad
        all_velocities = as_numpy(robot.get_dof_velocities())[0].astype(np.float64)
        all_efforts = as_numpy(robot.get_dof_efforts())[0].astype(np.float64)
        record_run_section(
            "approach_execution",
            {
                "status": "rejected_tracking",
                "failed_phase": phase,
                "tracking_settle_steps": APPROACH_TRACKING_SETTLE_STEPS,
                "joint_tracking_errors_rad": joint_errors_rad,
                "actual_joint_velocities_rad_s": all_velocities[arm_indices],
                "measured_joint_efforts": all_efforts[arm_indices],
                "tcp_tracking_error_mm": tcp_error_m * 1000.0,
                "tracking_trace": tracking_trace,
            },
        )
        raise RuntimeError(
            f"Approach {phase} failed execution tracking gates: "
            f"max joint error={math.degrees(float(np.max(np.abs(joint_errors_rad)))):.3f} deg "
            f"(limit={math.degrees(APPROACH_JOINT_TRACKING_TOLERANCE_RAD):.3f} deg), "
            f"TCP error={tcp_error_m * 1000.0:.3f} mm "
            f"(limit={APPROACH_TCP_TRACKING_TOLERANCE_M * 1000.0:.3f} mm)"
        )

    print(
        "\nApplying selected approach path: "
        f"mode={selection['mode']}, "
        f"selection={selection['selection_reason']}"
    )
    print(f"  Hover joints (rad): {waypoint_joints_rad[0].tolist()}")
    print(f"  Final joints (rad): {waypoint_joints_rad[-1].tolist()}")
    print(f"  Vertical descent segments: {len(descent_steps)}")
    print(f"  Descent command steps: {descent_steps}")
    print(
        "  Completion policy: complete the planned grasp depth with stable "
        "joint/TCP tracking; finger-sphere contact is allowed"
    )
    print(f"  Baseline robot contacts: {sorted(baseline_contact_pairs)}")

    # Move from the live pre-IK configuration to the selected hover solution.
    # Hover is a complete segment before the vertical Cartesian descent.
    hover_start_joints_rad = live_arm_joints()
    for step in range(1, IK_COMMAND_STEPS + 1):
        fraction = step / IK_COMMAND_STEPS
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        command = hover_start_joints_rad + blend * (
            waypoint_joints_rad[0] - hover_start_joints_rad
        )
        robot.set_dof_position_targets(command, dof_indices=arm_indices)
        SIMULATION_APP.update()
        reject_unexpected_contacts(
            phase="transition_to_hover",
            command_step=step,
            target_world_m=waypoint_targets_world_m[0],
        )
    _, hover_contact_m, hover_error_m, hover_settle_steps = wait_for_tracking(
        waypoint_joints_rad[0],
        waypoint_targets_world_m[0],
        phase="hover",
    )
    print(
        "  Hover contact residual after commanded motion: "
        f"{hover_error_m * 1000.0:.3f} mm"
    )
    print_point("  Reached hover contact point (world)", hover_contact_m)

    final_target_world_m = waypoint_targets_world_m[-1]
    applied_descent_steps = 0
    # Finger contact on the upper hemisphere is expected during a tangential
    # vertical insertion. It must not turn the current, still-high arm state
    # into the final grasp pose; only unexpected robot contacts remain fatal.
    contact_diagnostics = DescentContactDiagnostics()
    for segment_index, segment_steps in enumerate(descent_steps, start=1):
        segment_start_joints_rad = live_arm_joints()
        segment_target_joints_rad = waypoint_joints_rad[segment_index]
        segment_target_world_m = waypoint_targets_world_m[segment_index]
        for step in range(1, segment_steps + 1):
            fraction = step / segment_steps
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            command = segment_start_joints_rad + blend * (
                segment_target_joints_rad - segment_start_joints_rad
            )
            robot.set_dof_position_targets(command, dof_indices=arm_indices)
            SIMULATION_APP.update()
            applied_descent_steps += 1
            reject_unexpected_contacts(
                phase=f"descent_segment_{segment_index}",
                command_step=step,
                target_world_m=segment_target_world_m,
                allow_sphere_fingers=True,
            )
            fixed_contact = contact_tracker.fixed_finger_in_contact
            moving_contact = contact_tracker.moving_finger_in_contact
            first_any_contact, first_two_finger_contact = (
                contact_diagnostics.observe(
                    segment=segment_index,
                    step=step,
                    fixed_contact=fixed_contact,
                    moving_contact=moving_contact,
                )
            )
            if first_any_contact:
                print(
                    "  Tangential vertical contact started: "
                    f"segment={segment_index}, step={step}, "
                    f"fixed={fixed_contact}, moving={moving_contact}"
                )
            if first_two_finger_contact:
                print(
                    "  Two-finger contact during vertical insertion; "
                    "continuing toward the planned grasp depth: "
                    f"segment={segment_index}, step={step}, "
                    f"fixed={fixed_contact}, moving={moving_contact}"
                )
        wait_for_tracking(
            segment_target_joints_rad,
            segment_target_world_m,
            phase=f"descent_segment_{segment_index}",
            allow_sphere_fingers=True,
        )

    # Keep commanding the selected final solution while the gripper closes;
    # it is the minimum-residual target even when the simulated arm lags.
    commanded_arm_target_rad = waypoint_joints_rad[-1].copy()
    robot.set_dof_position_targets(commanded_arm_target_rad, dof_indices=arm_indices)
    actual_joints_rad = live_arm_joints()
    actual_fingertip_m, fingertip_error_m = contact_error_m(
        final_target_world_m
    )
    joint_errors_rad = actual_joints_rad - commanded_arm_target_rad
    print(f"  Applied hover steps: {IK_COMMAND_STEPS}")
    print(f"  Applied descent steps: {applied_descent_steps}")
    print(f"  Commanded final joints (rad): {commanded_arm_target_rad.tolist()}")
    print(f"  Actual joints (rad): {actual_joints_rad.tolist()}")
    print(f"  Joint tracking errors (rad): {joint_errors_rad.tolist()}")
    print_point("Applied descent fingertip (world)", actual_fingertip_m)
    print_point("Applied descent target (world)", final_target_world_m)
    print(f"  Measured fingertip position error: {fingertip_error_m * 1000.0:.3f} mm")
    draw_ik_execution_residual(
        actual_contact_world=actual_fingertip_m / meters_per_unit,
        target_contact_world=final_target_world_m / meters_per_unit,
    )
    record_run_section(
        "approach_execution",
        {
            "hover_command_steps": IK_COMMAND_STEPS,
            "hover_tracking_settle_steps": hover_settle_steps,
            "descent_command_steps": applied_descent_steps,
            "hover_position_error_mm": hover_error_m * 1000.0,
            "terminal_position_error_mm": fingertip_error_m * 1000.0,
            "hover_fingertip_world_m": hover_contact_m,
            "terminal_fingertip_world_m": actual_fingertip_m,
            "terminal_target_world_m": final_target_world_m,
            "commanded_arm_joints_rad": commanded_arm_target_rad,
            "actual_arm_joints_rad": actual_joints_rad,
            "joint_tracking_errors_rad": joint_errors_rad,
            "contact_stopped_final_insertion": False,
            "contact_stop_step": None,
            **contact_diagnostics.as_dict(),
            "terminal_fixed_finger_contact": (
                contact_tracker.fixed_finger_in_contact
            ),
            "terminal_moving_finger_contact": (
                contact_tracker.moving_finger_in_contact
            ),
        },
    )

    return commanded_arm_target_rad


def close_gripper_until_stable_grasp(
    robot: Articulation,
    arm_hold_target_rad: np.ndarray,
    contact_tracker: SphereFingerContactTracker,
    sphere: XformPrim,
    meters_per_unit: float,
) -> float:
    """Close until stable and keep the fully closed target active."""
    arm_indices = resolve_arm_dof_indices(robot)
    arm_target_rad = np.asarray(arm_hold_target_rad, dtype=np.float64)
    lower_limits, _ = robot.get_dof_limits()
    closed_target_rad = float(
        as_numpy(lower_limits)[0, GRIPPER_DOF_INDEX]
    )
    initial_kps, initial_kds = robot.get_dof_gains()
    initial_gripper_kp = float(
        as_numpy(initial_kps)[0, GRIPPER_DOF_INDEX]
    )
    initial_gripper_kd = float(
        as_numpy(initial_kds)[0, GRIPPER_DOF_INDEX]
    )
    robot.set_dof_gains(
        stiffnesses=[initial_gripper_kp * GRIPPER_STIFFNESS_MULTIPLIER],
        dampings=[initial_gripper_kd],
        dof_indices=[GRIPPER_DOF_INDEX],
    )
    effective_kps, effective_kds = robot.get_dof_gains()
    record_run_section(
        "grasp_controller",
        {
            "initial_stiffness": initial_gripper_kp,
            "effective_stiffness": float(
                as_numpy(effective_kps)[0, GRIPPER_DOF_INDEX]
            ),
            "stiffness_multiplier": GRIPPER_STIFFNESS_MULTIPLIER,
            "initial_damping": initial_gripper_kd,
            "effective_damping": float(
                as_numpy(effective_kds)[0, GRIPPER_DOF_INDEX]
            ),
        },
    )
    active_close_target_rad = closed_target_rad
    contact_squeeze_engaged = False

    def sphere_world_m() -> np.ndarray:
        positions, _ = sphere.get_world_poses()
        return as_numpy(positions)[0].astype(np.float64) * meters_per_unit

    print()
    print("Closing gripper until stable two-finger grasp:")
    print(f"  Input index: {GRIPPER_DOF_INDEX}")
    print(f"  Fully closed target (rad): {closed_target_rad:.6f}")
    print(
        "  Stable grasp requirements: "
        f"{GRASP_STABLE_CONTACT_FRAMES} consecutive physics frames, "
        f"|jaw velocity| <= {GRASP_STABLE_VELOCITY_THRESHOLD_RAD_S:.3f} rad/s"
    )

    qualifying_frames = 0
    step = 0
    while SIMULATION_APP.is_running():
        step += 1
        robot.set_dof_position_targets(arm_target_rad, dof_indices=arm_indices)
        robot.set_dof_position_targets(
            [active_close_target_rad], dof_indices=[GRIPPER_DOF_INDEX]
        )
        SIMULATION_APP.update()

        positions = as_numpy(robot.get_dof_positions())[0]
        velocities = as_numpy(robot.get_dof_velocities())[0]
        position_targets = as_numpy(robot.get_dof_position_targets())[0]
        actual_position_rad = float(positions[GRIPPER_DOF_INDEX])
        actual_velocity_rad_s = float(velocities[GRIPPER_DOF_INDEX])
        commanded_target_rad = float(position_targets[GRIPPER_DOF_INDEX])
        target_gap_rad = actual_position_rad - commanded_target_rad
        fixed_contact = contact_tracker.fixed_finger_in_contact
        moving_contact = contact_tracker.moving_finger_in_contact
        if fixed_contact and moving_contact and not contact_squeeze_engaged:
            active_close_target_rad = max(
                closed_target_rad,
                actual_position_rad - GRIPPER_CONTACT_SQUEEZE_DELTA_RAD,
            )
            contact_squeeze_engaged = True
            robot.set_dof_position_targets(
                [active_close_target_rad],
                dof_indices=[GRIPPER_DOF_INDEX],
            )
            print(
                "  Two-finger contact: limiting closure to squeeze target "
                f"{active_close_target_rad:.6f} rad"
            )
        qualifies = (
            fixed_contact
            and moving_contact
            and abs(actual_velocity_rad_s)
            <= GRASP_STABLE_VELOCITY_THRESHOLD_RAD_S
        )

        if qualifies:
            qualifying_frames += 1
            if qualifying_frames == 1:
                print("  Stable-grasp counter started")
            if qualifying_frames >= GRASP_STABLE_CONTACT_FRAMES:
                maintenance_target_rad = max(
                    closed_target_rad,
                    actual_position_rad - GRIPPER_LIFT_SQUEEZE_DELTA_RAD,
                )
                robot.set_dof_position_targets(
                    [maintenance_target_rad],
                    dof_indices=[GRIPPER_DOF_INDEX],
                )
                commanded_target_rad = maintenance_target_rad
                target_gap_rad = actual_position_rad - maintenance_target_rad
                print()
                print("Stable two-finger grasp detected:")
                print(f"  Physics steps in closure: {step}")
                print(f"  Actual position (rad): {actual_position_rad:.6f}")
                print(f"  Actual velocity (rad/s): {actual_velocity_rad_s:.6f}")
                print(f"  Commanded target (rad): {commanded_target_rad:.6f}")
                print(f"  Remaining close gap (rad): {target_gap_rad:.6f}")
                print(
                    f"  Consecutive qualifying frames: {qualifying_frames}"
                )

                builtins.print(
                    "[GRASP COMPLETED] Stable two-finger grasp finalized; "
                    f"keeping squeeze target={commanded_target_rad:.6f} rad "
                    "to maintain squeeze force during lift.",
                    flush=True,
                )
                record_run_section(
                    "grasp",
                    {
                        "stable": True,
                        "closure_steps": step,
                        "actual_position_rad": actual_position_rad,
                        "actual_velocity_rad_s": actual_velocity_rad_s,
                        "commanded_target_rad": commanded_target_rad,
                        "remaining_close_gap_rad": target_gap_rad,
                        "fixed_finger_contact": fixed_contact,
                        "moving_finger_contact": moving_contact,
                        "consecutive_qualifying_frames": qualifying_frames,
                        "velocity_threshold_rad_s": (
                            GRASP_STABLE_VELOCITY_THRESHOLD_RAD_S
                        ),
                        "contact_squeeze_engaged": contact_squeeze_engaged,
                        "contact_squeeze_delta_rad": (
                            GRIPPER_CONTACT_SQUEEZE_DELTA_RAD
                        ),
                        "lift_squeeze_delta_rad": (
                            GRIPPER_LIFT_SQUEEZE_DELTA_RAD
                        ),
                        "sphere_world_m": sphere_world_m(),
                    },
                )
                return commanded_target_rad
        elif qualifying_frames:
            print(
                "  Stable-grasp counter reset after "
                f"{qualifying_frames} qualifying frame(s)"
            )
            qualifying_frames = 0

        if (
            actual_position_rad - closed_target_rad
            <= GRIPPER_LOWER_LIMIT_TOLERANCE_RAD
        ):
            record_run_section(
                "grasp",
                {
                    "stable": False,
                    "closure_steps": step,
                    "actual_position_rad": actual_position_rad,
                    "actual_velocity_rad_s": actual_velocity_rad_s,
                    "commanded_target_rad": commanded_target_rad,
                    "remaining_close_gap_rad": target_gap_rad,
                    "fixed_finger_contact": fixed_contact,
                    "moving_finger_contact": moving_contact,
                    "consecutive_qualifying_frames": qualifying_frames,
                    "sphere_world_m": sphere_world_m(),
                },
            )
            raise RuntimeError(
                "Gripper reached the fully closed lower limit without a "
                "stable two-finger grasp: "
                f"position={actual_position_rad:.6f} rad, "
                f"velocity={actual_velocity_rad_s:.6f} rad/s, "
                f"target_gap={target_gap_rad:.6f} rad, "
                f"fixed_contact={fixed_contact}, "
                f"moving_contact={moving_contact}"
            )

        if step % GRIPPER_CLOSE_PROGRESS_INTERVAL_STEPS == 0:
            print(
                f"  Step {step}: position={actual_position_rad:.6f} rad, "
                f"velocity={actual_velocity_rad_s:.6f} rad/s, "
                f"close-gap={target_gap_rad:.6f} rad, "
                f"fixed-contact={contact_tracker.fixed_finger_in_contact}, "
                f"moving-contact={contact_tracker.moving_finger_in_contact}"
            )
        if step >= GRIPPER_MAX_CLOSE_STEPS:
            raise RuntimeError(
                "Gripper closure did not settle before the step limit: "
                f"steps={step}, fixed_contact={fixed_contact}, "
                f"moving_contact={moving_contact}, "
                f"velocity={actual_velocity_rad_s:.6f} rad/s"
            )

    raise RuntimeError("Simulation stopped before a stable grasp was detected")


def lift_grasped_sphere(
    robot: Articulation,
    contact_point: FixedToolPoint,
    sphere: XformPrim,
    sphere_start_world_stage_units: np.ndarray,
    target_world_stage_units: np.ndarray,
    meters_per_unit: float,
    ik_result: dict[str, object],
    gripper_target_rad: float,
    contact_tracker: SphereFingerContactTracker,
) -> np.ndarray:
    """Lift to a vertical fingertip target while holding the finalized grasp."""
    arm_indices = resolve_arm_dof_indices(robot)
    solved_joints_deg = np.asarray(
        ik_result["solved_joints_deg"], dtype=np.float64
    )
    solved_joints_rad = np.deg2rad(solved_joints_deg)
    current_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    start_joints_rad = current_all[arm_indices].copy()
    target_world_m = (
        np.asarray(target_world_stage_units, dtype=np.float64) * meters_per_unit
    )
    stop_distance_m = ARGS.lift_stop_distance_mm / 1000.0
    contact_loss_frames = 0

    def sphere_displacement_m() -> tuple[np.ndarray, float]:
        sphere_positions, _ = sphere.get_world_poses()
        current_sphere_world = as_numpy(sphere_positions)[0].astype(np.float64)
        displacement = (
            current_sphere_world
            - np.asarray(sphere_start_world_stage_units, dtype=np.float64)
        ) * meters_per_unit
        return displacement, float(np.linalg.norm(displacement))

    print()
    print("Lifting grasped sphere:")
    print(f"  Start joints (rad): {start_joints_rad.tolist()}")
    print(f"  Target joints (rad): {solved_joints_rad.tolist()}")
    print(f"  Constant gripper target (rad): {gripper_target_rad:.6f}")
    print(f"  Lift height: {ARGS.lift_height_mm:.3f} mm")
    print(f"  Lift stop distance: {ARGS.lift_stop_distance_mm:.3f} mm")

    for step in range(1, LIFT_COMMAND_STEPS + 1):
        fraction = step / LIFT_COMMAND_STEPS
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        arm_command = start_joints_rad + blend * (
            solved_joints_rad - start_joints_rad
        )
        robot.set_dof_position_targets(
            arm_command,
            dof_indices=arm_indices,
        )
        robot.set_dof_position_targets(
            [gripper_target_rad],
            dof_indices=[GRIPPER_DOF_INDEX],
        )
        SIMULATION_APP.update()

        fixed_contact = contact_tracker.fixed_finger_in_contact
        moving_contact = contact_tracker.moving_finger_in_contact
        if fixed_contact and moving_contact:
            contact_loss_frames = 0
        else:
            contact_loss_frames += 1
            if contact_loss_frames >= LIFT_CONTACT_LOSS_TOLERANCE_FRAMES:
                actual_all = as_numpy(
                    robot.get_dof_positions()
                )[0].astype(np.float64)
                robot.set_dof_position_targets(
                    actual_all[arm_indices],
                    dof_indices=arm_indices,
                )
                robot.set_dof_position_targets(
                    [gripper_target_rad],
                    dof_indices=[GRIPPER_DOF_INDEX],
                )
                sphere_displacement, sphere_distance = sphere_displacement_m()
                record_run_section(
                    "lift",
                    {
                        "completed": False,
                        "termination": "contact_lost",
                        "command_steps": step,
                        "fixed_finger_contact": fixed_contact,
                        "moving_finger_contact": moving_contact,
                        "sphere_displacement_m": sphere_displacement,
                        "sphere_displacement_norm_mm": sphere_distance * 1000.0,
                    },
                )
                raise RuntimeError(
                    "Stable grasp was lost during lift: "
                    f"step={step}, fixed_contact={fixed_contact}, "
                    f"moving_contact={moving_contact}"
                )

        actual_fingertip_stage_units = (
            contact_point.world_position_stage_units()
        )
        fingertip_error_m = float(
            np.linalg.norm(
                actual_fingertip_stage_units * meters_per_unit - target_world_m
            )
        )
        if fingertip_error_m <= stop_distance_m:
            actual_all = as_numpy(
                robot.get_dof_positions()
            )[0].astype(np.float64)
            lifted_arm_hold_rad = actual_all[arm_indices].copy()
            robot.set_dof_position_targets(
                lifted_arm_hold_rad,
                dof_indices=arm_indices,
            )
            robot.set_dof_position_targets(
                [gripper_target_rad],
                dof_indices=[GRIPPER_DOF_INDEX],
            )
            builtins.print(
                "[LIFT COMPLETED] "
                f"step={step}, "
                f"distance={fingertip_error_m * 1000.0:.3f} mm, "
                f"threshold={stop_distance_m * 1000.0:.3f} mm",
                flush=True,
            )
            sphere_displacement, sphere_distance = sphere_displacement_m()
            record_run_section(
                "lift",
                {
                    "completed": True,
                    "termination": "within_stop_distance",
                    "command_steps": step,
                    "terminal_fingertip_error_mm": fingertip_error_m * 1000.0,
                    "stop_distance_mm": stop_distance_m * 1000.0,
                    "arm_hold_joints_rad": lifted_arm_hold_rad,
                    "fixed_finger_contact": fixed_contact,
                    "moving_finger_contact": moving_contact,
                    "sphere_displacement_m": sphere_displacement,
                    "sphere_displacement_norm_mm": sphere_distance * 1000.0,
                },
            )
            return lifted_arm_hold_rad

    actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    robot.set_dof_position_targets(
        actual_all[arm_indices],
        dof_indices=arm_indices,
    )
    robot.set_dof_position_targets(
        [gripper_target_rad],
        dof_indices=[GRIPPER_DOF_INDEX],
    )
    actual_fingertip_stage_units = contact_point.world_position_stage_units()
    fingertip_error_m = float(
        np.linalg.norm(
            actual_fingertip_stage_units * meters_per_unit - target_world_m
        )
    )
    sphere_displacement, sphere_distance = sphere_displacement_m()
    record_run_section(
        "lift",
        {
            "completed": False,
            "termination": "command_step_limit",
            "command_steps": LIFT_COMMAND_STEPS,
            "terminal_fingertip_error_mm": fingertip_error_m * 1000.0,
            "stop_distance_mm": stop_distance_m * 1000.0,
            "sphere_displacement_m": sphere_displacement,
            "sphere_displacement_norm_mm": sphere_distance * 1000.0,
        },
    )
    raise RuntimeError(
        "Lift did not reach its target before the command-step limit: "
        f"steps={LIFT_COMMAND_STEPS}, "
        f"remaining_distance={fingertip_error_m * 1000.0:.3f} mm"
    )


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


def inverse_rotate_vector_wxyz(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Express a world vector in the local frame of an Isaac quaternion."""
    inverse_quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    inverse_quaternion[1:] *= -1.0
    return rotate_vector_wxyz(inverse_quaternion, vector)


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


def choose_grasp_point_nearest_contact(
    grasp_points: tuple[np.ndarray, np.ndarray],
    contact_point_world: np.ndarray,
) -> np.ndarray:
    """Choose the perpendicular-axis sphere point closest to the contact point."""
    distances = [
        float(np.linalg.norm(point - contact_point_world))
        for point in grasp_points
    ]
    return grasp_points[int(np.argmin(distances))]


def offset_outward_from_sphere(
    sphere_center: np.ndarray,
    sphere_surface_point: np.ndarray,
    clearance_stage_units: float,
) -> np.ndarray:
    """Move a point on the sphere surface outward along its surface normal."""
    outward_normal = sphere_surface_point - sphere_center
    normal_length = float(np.linalg.norm(outward_normal))
    if normal_length <= 1e-12:
        raise RuntimeError("Cannot calculate a sphere-surface normal at its center")
    return sphere_surface_point + clearance_stage_units * outward_normal / normal_length


def normalized_vector(vector: np.ndarray, label: str) -> np.ndarray:
    """Return a finite non-zero 3-vector normalized to unit length."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"{label} must be a finite three-vector: {vector}")
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise RuntimeError(f"{label} must have non-zero length")
    return vector / length


def unit_perpendicular_to(
    vector: np.ndarray,
    normal: np.ndarray,
    label: str,
) -> np.ndarray:
    """Project ``vector`` onto the plane normal to ``normal`` and normalize."""
    normal = normalized_vector(normal, f"{label} normal")
    vector = np.asarray(vector, dtype=np.float64)
    return normalized_vector(
        vector - np.dot(vector, normal) * normal,
        f"{label} after projection perpendicular to its approach axis",
    )


def make_top_down_target_rotation(
    local_approach: np.ndarray,
    local_closing: np.ndarray,
    world_closing: np.ndarray,
) -> np.ndarray:
    """Map live tool geometry onto a down-facing, sphere-closing pose.

    The temporary IK fingertip frame added in the LeRobot worker has the same
    orientation as ``gripper_link``.  Its local axes therefore remain the
    correct source axes for this rotation, even though its origin is moved to
    the authored fixed-finger contact point.
    """
    source_z = normalized_vector(local_approach, "Local tool approach axis")
    source_y = unit_perpendicular_to(
        local_closing,
        source_z,
        "Local jaw-closing axis",
    )
    source_x = normalized_vector(
        np.cross(source_y, source_z),
        "Local top-down frame x axis",
    )

    target_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    target_y = unit_perpendicular_to(
        world_closing,
        target_z,
        "World jaw-closing axis",
    )
    target_x = normalized_vector(
        np.cross(target_y, target_z),
        "World top-down frame x axis",
    )

    source_basis = np.column_stack((source_x, source_y, source_z))
    target_basis = np.column_stack((target_x, target_y, target_z))
    rotation = target_basis @ source_basis.T
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-8,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-8):
        raise RuntimeError("Constructed top-down target is not a proper rotation")
    return rotation


def top_down_yaw_candidates(
    nominal_rotation: np.ndarray,
    yaw_offsets_deg: list[float],
) -> list[tuple[float, np.ndarray]]:
    """Return world-Z yaw variations of a nominal top-down rotation."""
    candidates: list[tuple[float, np.ndarray]] = []
    for yaw_offset_deg in yaw_offsets_deg:
        yaw_offset_rad = math.radians(yaw_offset_deg)
        cosine = math.cos(yaw_offset_rad)
        sine = math.sin(yaw_offset_rad)
        world_z_rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        candidates.append((yaw_offset_deg, world_z_rotation @ nominal_rotation))
    return candidates


def print_point(label: str, point_meters: np.ndarray) -> None:
    print(
        f"{label}: "
        f"x={point_meters[0]:.6f} m, "
        f"y={point_meters[1]:.6f} m, "
        f"z={point_meters[2]:.6f} m"
    )


def apply_grasp_physics_material(stage: Usd.Stage) -> None:
    """Bind high friction to fingers without increasing table drag."""
    material = UsdShade.Material.Define(
        stage, "/PhysicsMaterials/GraspHighFriction"
    )
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(GRASP_STATIC_FRICTION)
    physics_material.CreateDynamicFrictionAttr().Set(GRASP_DYNAMIC_FRICTION)
    physics_material.CreateRestitutionAttr().Set(0.0)
    for prim_path in (
        FIXED_FINGER_PRIM_PATH,
        MOVING_JAW_PRIM_PATH,
    ):
        prim = require_prim(stage, prim_path)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
    record_run_section(
        "grasp_material",
        {
            "static_friction": GRASP_STATIC_FRICTION,
            "dynamic_friction": GRASP_DYNAMIC_FRICTION,
            "restitution": 0.0,
            "bound_prims": [
                FIXED_FINGER_PRIM_PATH,
                MOVING_JAW_PRIM_PATH,
            ],
        },
    )
    print(
        "High-friction grasp material enabled: "
        f"static={GRASP_STATIC_FRICTION:.2f}, "
        f"dynamic={GRASP_DYNAMIC_FRICTION:.2f}"
    )


def main() -> None:
    set_run_stage("initialize_lerobot")
    initialize_lerobot()

    set_run_stage("open_world")
    world_path = ARGS.world.expanduser().resolve()
    if not world_path.is_file():
        raise FileNotFoundError(f"World USD does not exist: {world_path}")

    opened, stage = stage_utils.open_stage(str(world_path))
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open world: {world_path}")

    require_prim(stage, FIXED_FINGER_PRIM_PATH)
    require_prim(stage, MOVING_JAW_PRIM_PATH)
    require_prim(stage, BASE_LINK_PRIM_PATH)
    require_prim(stage, BASE_REFERENCE_PRIM_PATH)
    sphere_prim = require_prim(stage, SPHERE_PRIM_PATH)
    apply_grasp_physics_material(stage)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit: {meters_per_unit}")
    tool_model = fixed_tool_model_from_urdf(
        SO101_URDF_PATH,
        joint_name=FIXED_TOOL_JOINT_NAME,
        expected_parent_link=FIXED_TOOL_PARENT_LINK,
        expected_tool_link=FIXED_TOOL_LINK,
    )
    legacy_contact_prim = stage.GetPrimAtPath(LEGACY_CONTACT_POINT_PRIM_PATH)
    legacy_contact_offset_m: list[float] | None = None
    if legacy_contact_prim.IsValid():
        legacy_translation = legacy_contact_prim.GetAttribute(
            "xformOp:translate"
        ).Get()
        if legacy_translation is not None:
            legacy_contact_offset_m = np.asarray(
                legacy_translation,
                dtype=np.float64,
            ).tolist()
    record_run_section(
        "scene",
        {
            "world": world_path,
            "meters_per_unit": meters_per_unit,
            "sphere_prim": SPHERE_PRIM_PATH,
        },
    )
    record_run_section(
        "tool_model",
        {
            "source": "urdf_fixed_joint",
            "joint_name": tool_model.joint_name,
            "parent_link": tool_model.parent_link,
            "tool_link": tool_model.tool_link,
            "position_in_parent_m": tool_model.position_in_parent_m,
            "approach_axis_in_parent": tool_model.approach_axis_in_parent,
            "closing_axis_in_parent": tool_model.closing_axis_in_parent,
            "legacy_marker_path": LEGACY_CONTACT_POINT_PRIM_PATH,
            "legacy_marker_present": legacy_contact_prim.IsValid(),
            "legacy_marker_active": (
                legacy_contact_prim.IsActive()
                if legacy_contact_prim.IsValid()
                else None
            ),
            "legacy_marker_offset_m": legacy_contact_offset_m,
            "legacy_marker_offset_error_mm": (
                float(
                    np.linalg.norm(
                        np.asarray(legacy_contact_offset_m, dtype=np.float64)
                        - tool_model.position_in_parent_m
                    )
                    * 1000.0
                )
                if legacy_contact_offset_m is not None
                else None
            ),
        },
    )

    set_run_stage("settle_simulation")
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    robot = Articulation(ROBOT_PRIM_PATH)
    base_link = XformPrim(BASE_LINK_PRIM_PATH)
    base_reference = XformPrim(BASE_REFERENCE_PRIM_PATH)
    gripper = XformPrim(FIXED_FINGER_PRIM_PATH)
    contact_point = FixedToolPoint(gripper, tool_model, meters_per_unit)
    sphere = XformPrim(SPHERE_PRIM_PATH)
    contact_tracker = SphereFingerContactTracker(sphere_prim)

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
    record_run_section("scene", {"completed_settle_steps": completed_steps})

    set_run_stage("align_and_seed")
    sphere_positions, _ = sphere.get_world_poses()
    sphere_center = as_numpy(sphere_positions)[0]
    shoulder_pan_angle = align_base_with_sphere(
        robot,
        base_reference,
        base_link,
        sphere_center,
    )
    command_ik_seed_presets(robot)
    set_run_stage("validate_tool_model")
    validate_tool_model_alignment(
        stage=stage,
        robot=robot,
        base_link=base_link,
        gripper=gripper,
        tool_model=tool_model,
        meters_per_unit=meters_per_unit,
    )

    # Re-read all live poses because alignment and the pre-IK configuration
    # commands advance physics. The authored contact point replaces the prior
    # mesh-derived fingertip estimate everywhere below.
    contact_point_world = contact_point.world_position_stage_units()
    contact_point_local_stage_units = meters_to_stage(
        tool_model.position_in_parent_m,
        meters_per_unit,
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
    sphere_surface_grasp_point = choose_grasp_point_nearest_contact(
        sphere_grasp_points,
        contact_point_world,
    )
    sphere_grasp_point = offset_outward_from_sphere(
        sphere_center,
        sphere_surface_grasp_point,
        ARGS.sphere_surface_clearance_mm / 1000.0 / meters_per_unit,
    )

    # Build the requested top-down pose and vertical contact-point path from
    # the live gripper geometry.
    local_tool_approach = tool_model.approach_axis_in_parent
    local_jaw_closing = tool_model.closing_axis_in_parent
    world_jaw_closing = sphere_center - sphere_surface_grasp_point
    nominal_top_down_rotation = make_top_down_target_rotation(
        local_approach=local_tool_approach,
        local_closing=local_jaw_closing,
        world_closing=world_jaw_closing,
    )
    yaw_candidates = top_down_yaw_candidates(
        nominal_top_down_rotation,
        ARGS.top_down_yaw_offsets_deg,
    )
    top_down_waypoints = vertical_approach_waypoints(
        final_target_world_stage_units=sphere_grasp_point,
        meters_per_unit=meters_per_unit,
        hover_height_mm=ARGS.top_down_hover_height_mm,
        descent_step_mm=ARGS.top_down_descent_step_mm,
    )
    grasp_candidates = generate_sphere_grasp_candidates(
        sphere_center,
        sphere_radius,
        sphere_surface_grasp_point,
        meters_per_unit=meters_per_unit,
        surface_clearance_m=ARGS.sphere_surface_clearance_mm / 1000.0,
        surface_offsets_deg=GRASP_SURFACE_OFFSETS_DEG,
        approach_tilts_deg=GRASP_APPROACH_TILTS_DEG,
        hover_height_mm=ARGS.top_down_hover_height_mm,
        descent_step_mm=ARGS.top_down_descent_step_mm,
    )
    record_run_section(
        "target_geometry",
        {
            "sphere_center_world_m": sphere_center * meters_per_unit,
            "sphere_radius_m": sphere_radius * meters_per_unit,
            "base_heading_world": heading,
            "urdf_tool_point_world_m": (
                contact_point_world * meters_per_unit
            ),
            "sphere_grasp_points_world_m": [
                point * meters_per_unit for point in sphere_grasp_points
            ],
            "selected_surface_point_world_m": (
                sphere_surface_grasp_point * meters_per_unit
            ),
            "approach_target_world_m": sphere_grasp_point * meters_per_unit,
            "sphere_surface_clearance_mm": (
                ARGS.sphere_surface_clearance_mm
            ),
            "hover_target_world_m": top_down_waypoints[0] * meters_per_unit,
            "waypoint_count": len(top_down_waypoints),
            "local_tool_approach": local_tool_approach,
            "local_jaw_closing": local_jaw_closing,
            "nominal_top_down_rotation": nominal_top_down_rotation,
        },
    )

    print()
    print(
        "Aligned base heading (world horizontal unit vector): "
        f"x={heading[0]:.6f}, y={heading[1]:.6f}, z={heading[2]:.6f}"
    )
    print_point("URDF fixed tool point (world)", contact_point_world * meters_per_unit)
    print_point(
        "Sphere perpendicular grasp point A (world)",
        sphere_grasp_points[0] * meters_per_unit,
    )
    print_point(
        "Sphere perpendicular grasp point B (world)",
        sphere_grasp_points[1] * meters_per_unit,
    )
    print_point(
        "Selected sphere surface point (world)",
        sphere_surface_grasp_point * meters_per_unit,
    )
    print_point(
        f"IK approach point ({ARGS.sphere_surface_clearance_mm:.1f} mm clearance, world)",
        sphere_grasp_point * meters_per_unit,
    )
    print()
    print("Prepared top-down IK geometry:")
    print(
        "  Local approach axis: "
        f"{normalized_vector(local_tool_approach, 'Local tool approach axis').tolist()}"
    )
    print(
        "  Local jaw-closing axis: "
        f"{unit_perpendicular_to(local_jaw_closing, local_tool_approach, 'Local jaw-closing axis').tolist()}"
    )
    print(
        "  Nominal target rotation (world from gripper_link): "
        f"{nominal_top_down_rotation.tolist()}"
    )
    print("  Constrained approach axis: tool +Z aligned to each sampled axis")
    print("  Tool yaw: unconstrained")
    print(
        "  Vertical waypoints: "
        f"{len(top_down_waypoints)} "
        f"(hover={ARGS.top_down_hover_height_mm:.1f} mm, "
        f"max spacing={ARGS.top_down_descent_step_mm:.1f} mm)"
    )
    print_point(
        "  Hover contact target (world)",
        top_down_waypoints[0] * meters_per_unit,
    )

    # Calculate, gate, and rank all sampled grasp paths before issuing motion.
    set_run_stage("plan_approach")
    top_down_candidate_result = calculate_top_down_candidates(
        stage=stage,
        robot=robot,
        base_link=base_link,
        fingertip_offset_stage_units=contact_point_local_stage_units,
        grasp_candidates_world=grasp_candidates,
        meters_per_unit=meters_per_unit,
    )
    candidates = top_down_candidate_result["candidates"]
    assert isinstance(candidates, list)
    fallback = top_down_candidate_result.get("position_only_fallback")
    record_run_section(
        "approach_planning",
        {
            "candidate_count": len(candidates),
            "valid_candidate_count": sum(
                isinstance(candidate, dict)
                and candidate.get("status") == "valid"
                for candidate in candidates
            ),
            "rejected_candidate_count": sum(
                isinstance(candidate, dict)
                and candidate.get("status") == "rejected"
                for candidate in candidates
            ),
            "skipped_candidate_count": sum(
                isinstance(candidate, dict)
                and candidate.get("status") == "skipped"
                for candidate in candidates
            ),
            "selected_candidate_id": top_down_candidate_result.get(
                "selected_candidate_id"
            ),
            "planning_metrics": top_down_candidate_result.get(
                "planning_metrics"
            ),
            "fallback_status": (
                fallback.get("status") if isinstance(fallback, dict) else None
            ),
            "fallback_max_position_error_mm": (
                float(fallback["max_position_error_m"]) * 1000.0
                if isinstance(fallback, dict) and "max_position_error_m" in fallback
                else None
            ),
            "candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "status",
                        "rejection_stage",
                        "skip_reason",
                        "endpoint_rank",
                        "constraint_mode",
                        "candidate_id",
                        "target_kind",
                        "surface_offset_deg",
                        "approach_tilt_deg",
                        "surface_clearance_m",
                        "failed_waypoint_index",
                        "reason",
                        "endpoint_diagnostics",
                        "failed_waypoint_diagnostics",
                        "position_tolerance_m",
                        "axis_tolerance_rad",
                        "position_seed_iterations",
                        "endpoint_iterations",
                        "endpoint_joint_travel_deg",
                        "endpoint_position_error_m",
                        "endpoint_axis_error_rad",
                        "endpoint_closing_axis_error_rad",
                        "waypoint_iterations",
                        "max_position_error_m",
                        "terminal_position_error_m",
                        "max_axis_error_rad",
                        "terminal_axis_error_rad",
                        "total_joint_travel_deg",
                    )
                    if key in candidate
                }
                for candidate in candidates
                if isinstance(candidate, dict)
            ],
        },
    )
    top_down_selection = select_top_down_path(top_down_candidate_result)
    selected_path = top_down_selection["path"]
    assert isinstance(selected_path, dict)
    selected_metrics = {
        key: selected_path[key]
        for key in (
            "status",
            "constraint_mode",
            "candidate_id",
            "target_kind",
            "surface_offset_deg",
            "approach_tilt_deg",
            "surface_clearance_m",
            "position_tolerance_m",
            "axis_tolerance_rad",
            "max_position_error_m",
            "terminal_position_error_m",
            "max_axis_error_rad",
            "terminal_axis_error_rad",
            "total_joint_travel_deg",
        )
        if key in selected_path
    }
    record_run_section(
        "approach_planning",
        {
            "selection_mode": top_down_selection["mode"],
            "selection_reason": top_down_selection["selection_reason"],
            "selected_path": selected_metrics,
        },
    )
    selected_waypoint_targets = selected_path.get(
        "waypoint_target_positions_world_stage_units"
    )
    top_down_waypoints = [
        np.asarray(waypoint, dtype=np.float64)
        for waypoint in selected_waypoint_targets
    ]
    sphere_grasp_point = top_down_waypoints[-1].copy()
    selected_candidate_spec = next(
        candidate
        for candidate in grasp_candidates
        if candidate["candidate_id"] == selected_path["candidate_id"]
    )
    record_run_section(
        "target_geometry",
        {
            "selected_candidate_id": selected_path["candidate_id"],
            "selected_approach_target_world_m": (
                sphere_grasp_point * meters_per_unit
            ),
            "selected_hover_target_world_m": (
                top_down_waypoints[0] * meters_per_unit
            ),
            "selected_approach_axis_world": selected_candidate_spec[
                "target_axis_world"
            ],
        },
    )
    print_top_down_candidate_selection(
        top_down_candidate_result,
        top_down_selection,
    )

    draw_ik_debug_overlay(
        sphere_center_world=sphere_center,
        sphere_grasp_points_world=sphere_grasp_points,
        selected_sphere_grasp_point_world=sphere_grasp_point,
        contact_point_world=contact_point_world,
        base_heading_world=heading,
        top_down_waypoints_world=top_down_waypoints,
        target_approach_axis_world=np.asarray(
            selected_candidate_spec["target_axis_world"], dtype=np.float64
        ),
        local_tool_approach=local_tool_approach,
        local_jaw_closing=local_jaw_closing,
        yaw_candidates_world=yaw_candidates,
        candidate_result=top_down_candidate_result,
        selection=top_down_selection,
        meters_per_unit=meters_per_unit,
    )
    for _ in range(DEBUG_POINT_WAIT_STEPS):
        SIMULATION_APP.update()
    set_run_stage("execute_approach")
    arm_hold_target_rad = apply_selected_approach_path(
        robot=robot,
        contact_point=contact_point,
        contact_tracker=contact_tracker,
        waypoint_targets_world_stage_units=top_down_waypoints,
        meters_per_unit=meters_per_unit,
        selection=top_down_selection,
    )
    sphere_after_approach_positions, _ = sphere.get_world_poses()
    record_run_section(
        "approach_execution",
        {
            "sphere_world_m_after_approach": (
                as_numpy(sphere_after_approach_positions)[0] * meters_per_unit
            ),
            "sphere_displacement_after_approach_mm": (
                np.linalg.norm(
                    as_numpy(sphere_after_approach_positions)[0] - sphere_center
                )
                * meters_per_unit
                * 1000.0
            ),
        },
    )
    set_run_stage("close_gripper")
    gripper_close_target_rad = close_gripper_until_stable_grasp(
        robot=robot,
        arm_hold_target_rad=arm_hold_target_rad,
        contact_tracker=contact_tracker,
        sphere=sphere,
        meters_per_unit=meters_per_unit,
    )

    # Start the lift from the live post-grasp fingertip position. This makes
    # the requested displacement purely vertical even if the arm stopped
    # slightly short of its original approach target.
    lift_start_world = contact_point.world_position_stage_units()
    lift_target_world = lift_start_world.copy()
    lift_target_world[2] += (
        ARGS.lift_height_mm / 1000.0 / meters_per_unit
    )
    sphere_positions, _ = sphere.get_world_poses()
    sphere_lift_start_world = as_numpy(sphere_positions)[0].astype(np.float64)
    record_run_section(
        "lift",
        {
            "start_fingertip_world_m": lift_start_world * meters_per_unit,
            "target_fingertip_world_m": lift_target_world * meters_per_unit,
            "sphere_start_world_m": sphere_lift_start_world * meters_per_unit,
        },
    )
    print()
    print_point(
        "Lift start point (world)",
        lift_start_world * meters_per_unit,
    )
    print_point(
        "Lift target point (world)",
        lift_target_world * meters_per_unit,
    )

    set_run_stage("plan_lift")
    lift_ik_result = calculate_position_only_ik(
        stage=stage,
        robot=robot,
        base_link=base_link,
        fingertip_offset_stage_units=contact_point_local_stage_units,
        target_world_stage_units=lift_target_world,
        meters_per_unit=meters_per_unit,
    )
    record_run_section(
        "lift",
        {
            "planned_position_error_mm": (
                float(lift_ik_result["position_error_m"]) * 1000.0
            ),
            "planned_joints_deg": lift_ik_result["solved_joints_deg"],
        },
    )
    set_run_stage("execute_lift")
    lift_grasped_sphere(
        robot=robot,
        contact_point=contact_point,
        sphere=sphere,
        sphere_start_world_stage_units=sphere_lift_start_world,
        target_world_stage_units=lift_target_world,
        meters_per_unit=meters_per_unit,
        ik_result=lift_ik_result,
        gripper_target_rad=gripper_close_target_rad,
        contact_tracker=contact_tracker,
    )
    contact_tracker.print_status()
    RUN_SUMMARY.update(
        {
            "status": "completed",
            "success": True,
            "current_stage": "completed",
            "failure_stage": None,
            "failure_reason": None,
        }
    )


if __name__ == "__main__":
    run_error: BaseException | None = None
    try:
        main()
    except BaseException as error:
        run_error = error
        RUN_SUMMARY.update(
            {
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                "success": False,
                "failure_stage": CURRENT_RUN_STAGE,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        RUN_SUMMARY["duration_seconds"] = time.monotonic() - RUN_STARTED_MONOTONIC
        try:
            write_run_summary()
        except Exception as summary_error:
            print(
                f"WARNING: Could not write structured run result: {summary_error}",
                file=sys.stderr,
            )
            if run_error is None:
                raise
        finally:
            SIMULATION_APP.close()
