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
import traceback
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
CONTACT_POINT_PRIM_PATH = f"{FIXED_FINGER_PRIM_PATH}/contactPoint"
SPHERE_PRIM_PATH = "/Sphere"
SHOULDER_PAN_DOF_NAME = "shoulder_pan"
WRIST_PRESET_DOF_NAME = "wrist_roll"
GRIPPER_PRESET_DOF_NAME = "gripper"
BASE_ALIGNMENT_STEPS = 300
PRESET_COMMAND_STEPS = 300
WRIST_PRESET_POSITION_RAD = -1.55
GRIPPER_PRESET_POSITION_RAD = 1.5
GRIPPER_DOF_INDEX = 5
GRIPPER_CLOSE_PROGRESS_INTERVAL_STEPS = 30
GRASP_STABLE_CONTACT_FRAMES = 10
GRIPPER_REMAINING_CLOSE_GAP_THRESHOLD_RAD = 0.05
GRIPPER_LOWER_LIMIT_TOLERANCE_RAD = 0.01
DEFAULT_LIFT_HEIGHT_MM = 100.0
DEFAULT_LIFT_STOP_DISTANCE_MM = 7.0
DEFAULT_PREGRASP_CLEARANCE_MM = 30.0
LIFT_CONTACT_LOSS_TOLERANCE_FRAMES = 5
IK_MAX_ITERATIONS = 100
IK_POSITION_TOLERANCE_M = 0.0005
IK_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
IK_POSITION_WEIGHT = 1.0
IK_ORIENTATION_WEIGHT = 1.0
DEFAULT_APPROACH_STOP_DISTANCE_MM = 7.0
IK_COMMAND_STEPS = 300
POSE_IK_WAYPOINT_COMMAND_STEPS = 60
POSE_IK_WAYPOINT_SPACING_M = 0.005
IK_SETTLE_MAX_STEPS = 300
IK_SETTLED_FRAMES = 10
IK_SETTLED_ARM_VELOCITY_THRESHOLD_RAD_S = 0.01
DEBUG_POINT_WAIT_STEPS = 45
# Keep the contact point just outside the object until the grasp motion begins.
SPHERE_SURFACE_CLEARANCE_M = 0.0
LEROBOT_FINGERTIP_FRAME_RPY = "0 3.141592653589793 0"
# Diagnostic-only comparison: test whether the pre-grasp position is
# reachable before asking the 5-DOF arm to satisfy a full 6D pose.
DIAGNOSTIC_POSITION_ONLY_PREGRASP = True


def lerobot_worker_main() -> int:
    """Run LeRobot/Placo requests without importing Isaac Sim."""
    import numpy as worker_np

    from lerobot.model.kinematics import RobotKinematics

    def create_fingertip_urdf(
        temp_dir: Path, fingertip_offset_m: worker_np.ndarray
    ) -> Path:
        """Create a temporary URDF with Isaac's fixed fingertip frame."""
        # RobotKinematics constrains a URDF frame, while the desired point is
        # the fixed-finger mesh endpoint measured by Isaac. Add that exact
        # point as a temporary fixed frame without modifying third_party. Its
        # orientation matches gripper_frame_link, whose +Z tool axis points
        # through the gripper fingers.
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
            fingertip_joint, "child", {"link": LEROBOT_FINGERTIP_FRAME}
        )
        ET.SubElement(
            fingertip_joint,
            "origin",
            {
                "xyz": " ".join(
                    f"{value:.17g}" for value in fingertip_offset_m
                ),
                "rpy": LEROBOT_FINGERTIP_FRAME_RPY,
            },
        )
        fingertip_urdf = temp_dir / "so101_fingertip.urdf"
        urdf_tree.write(fingertip_urdf, encoding="utf-8", xml_declaration=True)
        return fingertip_urdf

    def validate_rigid_transform(
        transform: worker_np.ndarray, label: str
    ) -> None:
        """Reject malformed homogeneous transforms before calling Placo."""
        if transform.shape != (4, 4):
            raise ValueError(f"{label} must have shape (4, 4)")
        if not worker_np.all(worker_np.isfinite(transform)):
            raise ValueError(f"{label} contains NaN or infinity")
        if not worker_np.allclose(
            transform[3], worker_np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9
        ):
            raise ValueError(f"{label} must have homogeneous final row [0, 0, 0, 1]")

        rotation = transform[:3, :3]
        if not worker_np.allclose(
            rotation.T @ rotation, worker_np.eye(3), atol=1e-6
        ):
            raise ValueError(f"{label} rotation is not orthonormal")
        determinant = float(worker_np.linalg.det(rotation))
        if not worker_np.isclose(determinant, 1.0, atol=1e-6):
            raise ValueError(
                f"{label} rotation must have determinant +1, got {determinant}"
            )

    def orientation_error_rad(
        solved_rotation: worker_np.ndarray,
        target_rotation: worker_np.ndarray,
    ) -> float:
        """Return the shortest angular distance between two rotations."""
        relative_rotation = target_rotation.T @ solved_rotation
        cosine = float((worker_np.trace(relative_rotation) - 1.0) * 0.5)
        return float(worker_np.arccos(worker_np.clip(cosine, -1.0, 1.0)))

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
    elif operation == "position_ik":
        fingertip_offset_m = worker_np.asarray(
            request["fingertip_offset_m"], dtype=worker_np.float64
        )
        seed_joints_deg = worker_np.asarray(
            request["seed_joints_deg"], dtype=worker_np.float64
        )
        target_position_m = worker_np.asarray(
            request["target_position_m"], dtype=worker_np.float64
        )
        if fingertip_offset_m.shape != (3,):
            raise ValueError("fingertip_offset_m must contain three values")
        if seed_joints_deg.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError(
                f"seed_joints_deg must contain {len(ARM_JOINT_NAMES)} values"
            )
        if target_position_m.shape != (3,):
            raise ValueError("target_position_m must contain three values")
        if not all(
            worker_np.all(worker_np.isfinite(value))
            for value in (
                fingertip_offset_m,
                seed_joints_deg,
                target_position_m,
            )
        ):
            raise ValueError("Position IK request contains NaN or infinity")

        with tempfile.TemporaryDirectory(prefix="so101_position_ik_") as temp_dir:
            fingertip_urdf = create_fingertip_urdf(
                Path(temp_dir), fingertip_offset_m
            )
            kinematics = RobotKinematics(
                urdf_path=str(fingertip_urdf),
                target_frame_name=LEROBOT_FINGERTIP_FRAME,
                joint_names=ARM_JOINT_NAMES,
            )

            joints_deg = seed_joints_deg.copy()
            target_pose = worker_np.asarray(
                kinematics.forward_kinematics(joints_deg),
                dtype=worker_np.float64,
            )
            target_pose[:3, 3] = target_position_m
            solved_pose = target_pose.copy()
            position_error_m = float("inf")
            iterations = 0
            max_iterations = int(request["max_iterations"])
            tolerance_m = float(request["position_tolerance_m"])

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
                    kinematics.forward_kinematics(joints_deg),
                    dtype=worker_np.float64,
                )
                position_error_m = float(
                    worker_np.linalg.norm(
                        solved_pose[:3, 3] - target_position_m
                    )
                )
                if position_error_m <= tolerance_m:
                    break

        result = {
            "status": "converged"
            if position_error_m <= tolerance_m
            else "not_converged",
            "joint_names": ARM_JOINT_NAMES,
            "seed_joints_deg": seed_joints_deg.tolist(),
            "solved_joints_deg": joints_deg.tolist(),
            "target_position_m": target_position_m.tolist(),
            "solved_position_m": solved_pose[:3, 3].tolist(),
            "position_error_m": position_error_m,
            "iterations": iterations,
            "position_weight": 1.0,
            "orientation_weight": 0.0,
        }
    elif operation == "pose_ik_path":
        fingertip_offset_m = worker_np.asarray(
            request["fingertip_offset_m"], dtype=worker_np.float64
        )
        seed_joints_deg = worker_np.asarray(
            request["seed_joints_deg"], dtype=worker_np.float64
        )
        target_poses_m = worker_np.asarray(
            request["target_poses_m"], dtype=worker_np.float64
        )
        if fingertip_offset_m.shape != (3,):
            raise ValueError("fingertip_offset_m must contain three values")
        if seed_joints_deg.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError(
                f"seed_joints_deg must contain {len(ARM_JOINT_NAMES)} values"
            )
        if target_poses_m.ndim != 3 or target_poses_m.shape[1:] != (4, 4):
            raise ValueError(
                "target_poses_m must contain one or more transforms of shape (4, 4)"
            )
        if target_poses_m.shape[0] == 0:
            raise ValueError("target_poses_m must contain at least one target pose")
        if not all(
            worker_np.all(worker_np.isfinite(value))
            for value in (fingertip_offset_m, seed_joints_deg, target_poses_m)
        ):
            raise ValueError("Pose IK request contains NaN or infinity")

        max_iterations = int(request["max_iterations"])
        position_tolerance_m = float(request["position_tolerance_m"])
        orientation_tolerance_rad = float(request["orientation_tolerance_rad"])
        position_weight = float(request["position_weight"])
        orientation_weight = float(request["orientation_weight"])
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                position_tolerance_m,
                orientation_tolerance_rad,
                position_weight,
                orientation_weight,
            )
        ):
            raise ValueError(
                "Pose IK tolerances and weights must be finite positive values"
            )
        for index, target_pose_m in enumerate(target_poses_m):
            validate_rigid_transform(target_pose_m, f"target_poses_m[{index}]")

        with tempfile.TemporaryDirectory(prefix="so101_pose_ik_") as temp_dir:
            fingertip_urdf = create_fingertip_urdf(
                Path(temp_dir), fingertip_offset_m
            )
            kinematics = RobotKinematics(
                urdf_path=str(fingertip_urdf),
                target_frame_name=LEROBOT_FINGERTIP_FRAME,
                joint_names=ARM_JOINT_NAMES,
            )

            joints_deg = seed_joints_deg.copy()
            waypoint_results: list[dict[str, object]] = []
            status = "converged"
            for waypoint_index, target_pose_m in enumerate(target_poses_m):
                solved_pose = worker_np.asarray(
                    kinematics.forward_kinematics(joints_deg),
                    dtype=worker_np.float64,
                )
                initial_position_error_m = float(
                    worker_np.linalg.norm(
                        solved_pose[:3, 3] - target_pose_m[:3, 3]
                    )
                )
                initial_angular_error_rad = orientation_error_rad(
                    solved_pose[:3, :3], target_pose_m[:3, :3]
                )
                position_error_m = float("inf")
                angular_error_rad = float("inf")
                iterations = 0
                for iterations in range(1, max_iterations + 1):
                    joints_deg = worker_np.asarray(
                        kinematics.inverse_kinematics(
                            current_joint_pos=joints_deg,
                            desired_ee_pose=target_pose_m,
                            position_weight=position_weight,
                            orientation_weight=orientation_weight,
                        ),
                        dtype=worker_np.float64,
                    )
                    if not worker_np.all(worker_np.isfinite(joints_deg)):
                        raise RuntimeError("Pose IK returned NaN or infinity")
                    solved_pose = worker_np.asarray(
                        kinematics.forward_kinematics(joints_deg),
                        dtype=worker_np.float64,
                    )
                    position_error_m = float(
                        worker_np.linalg.norm(
                            solved_pose[:3, 3] - target_pose_m[:3, 3]
                        )
                    )
                    angular_error_rad = orientation_error_rad(
                        solved_pose[:3, :3], target_pose_m[:3, :3]
                    )
                    if (
                        position_error_m <= position_tolerance_m
                        and angular_error_rad <= orientation_tolerance_rad
                    ):
                        break

                waypoint_status = (
                    "converged"
                    if (
                        position_error_m <= position_tolerance_m
                        and angular_error_rad <= orientation_tolerance_rad
                    )
                    else "not_converged"
                )
                waypoint_results.append(
                    {
                        "index": waypoint_index,
                        "status": waypoint_status,
                        "target_pose_m": target_pose_m.tolist(),
                        "solved_pose_m": solved_pose.tolist(),
                        "solved_joints_deg": joints_deg.tolist(),
                        "initial_position_error_m": initial_position_error_m,
                        "initial_orientation_error_rad": initial_angular_error_rad,
                        "position_error_m": position_error_m,
                        "orientation_error_rad": angular_error_rad,
                        "iterations": iterations,
                    }
                )
                if waypoint_status != "converged":
                    status = "not_converged"
                    break

        result = {
            "status": status,
            "joint_names": ARM_JOINT_NAMES,
            "seed_joints_deg": seed_joints_deg.tolist(),
            "completed_waypoints": len(waypoint_results),
            "target_waypoints": len(target_poses_m),
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "position_weight": position_weight,
            "orientation_weight": orientation_weight,
            "waypoints": waypoint_results,
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
        "--lerobot-python",
        type=Path,
        default=DEFAULT_LEROBOT_PYTHON,
        help=(
            "Python interpreter containing LeRobot and Placo "
            f"(default: {DEFAULT_LEROBOT_PYTHON})"
        ),
    )
    parser.add_argument(
        "--approach-stop-distance-mm",
        type=float,
        default=DEFAULT_APPROACH_STOP_DISTANCE_MM,
        help=(
            "Stop arm movement when the authored gripper contact point is "
            "within this distance of the IK approach point "
            f"(default: {DEFAULT_APPROACH_STOP_DISTANCE_MM:.1f} mm)"
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
        "--pregrasp-clearance-mm",
        type=float,
        default=DEFAULT_PREGRASP_CLEARANCE_MM,
        help=(
            "Vertical clearance between the sphere top and the open-gripper "
            "pre-grasp contact point "
            f"(default: {DEFAULT_PREGRASP_CLEARANCE_MM:.1f} mm)"
        ),
    )
    args, _ = parser.parse_known_args()
    if args.settle_steps < 0:
        parser.error("--settle-steps cannot be negative")
    if (
        not math.isfinite(args.approach_stop_distance_mm)
        or args.approach_stop_distance_mm <= 0.0
    ):
        parser.error("--approach-stop-distance-mm must be a positive number")
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
        not math.isfinite(args.pregrasp_clearance_mm)
        or args.pregrasp_clearance_mm <= 0.0
    ):
        parser.error("--pregrasp-clearance-mm must be a positive number")
    return args


ARGS = parse_args()
SIMULATION_APP = SimulationApp({"headless": ARGS.headless})


import numpy as np  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.util.debug_draw import _debug_draw  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from omni.physx.bindings._physx import ContactEventType  # noqa: E402
from pxr import PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics  # noqa: E402


def run_lerobot_worker(request: dict[str, object]) -> dict[str, object]:
    """Run one request in the external LeRobot Python environment."""
    operation = str(request.get("operation", "<missing>"))
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

    if operation != "setup":
        builtins.print(
            f"[IK DIAGNOSTIC] Launching external worker: operation={operation}",
            flush=True,
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
    if operation != "setup":
        builtins.print(
            "[IK DIAGNOSTIC] Worker process returned: "
            f"returncode={completed.returncode}, "
            f"stdout_chars={len(completed.stdout)}, "
            f"stderr_chars={len(completed.stderr)}",
            flush=True,
        )
        if completed.stderr.strip():
            builtins.print(
                "[IK DIAGNOSTIC] Worker stderr follows:\n"
                f"{completed.stderr.rstrip()}",
                file=sys.stderr,
                flush=True,
            )

    if completed.returncode != 0 or not marker_lines:
        raise RuntimeError(
            f"LeRobot worker failed during operation={operation!r}: "
            f"returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )

    payload = marker_lines[-1][len(LEROBOT_WORKER_MARKER) :]
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"LeRobot worker returned malformed JSON during operation={operation!r}: "
            f"{error}\npayload={payload}\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        ) from error

    if operation != "setup":
        builtins.print(
            f"[IK DIAGNOSTIC] Worker result status={result.get('status')!r}",
            flush=True,
        )
    return result


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


def rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert an Isaac-order quaternion into a 3x3 rotation matrix."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_rpy(rpy: np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
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


def make_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def world_transform_meters(
    prim: XformPrim, meters_per_unit: float
) -> np.ndarray:
    positions, orientations = prim.get_world_poses()
    return make_transform(
        as_numpy(positions)[0].astype(np.float64) * meters_per_unit,
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


def resolve_arm_dof_indices(robot: Articulation) -> list[int]:
    index_by_name = {str(name): index for index, name in enumerate(robot.dof_names)}
    missing = [name for name in ARM_JOINT_NAMES if name not in index_by_name]
    if missing:
        raise RuntimeError(
            f"Missing required arm DOFs {missing}; actual names={robot.dof_names}"
        )
    indices = [index_by_name[name] for name in ARM_JOINT_NAMES]
    if len(set(indices)) != len(ARM_JOINT_NAMES):
        raise RuntimeError(f"Arm DOF mapping is not unique: {indices}")
    return indices


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

    # The imported USD base and the LeRobot URDF base use different coordinate
    # frames.  Derive their fixed bridge from the independently authored
    # shoulder-pan joint frames rather than fitting it from the end effector.
    usd_base_to_shoulder_zero = usd_joint_zero_transform(stage, "shoulder_pan")
    lerobot_base_to_shoulder_zero = urdf_joint_origin("shoulder_pan")
    usd_base_from_lerobot_base = (
        usd_base_to_shoulder_zero
        @ np.linalg.inv(lerobot_base_to_shoulder_zero)
    )
    world_from_usd_base = world_transform_meters(base_link, meters_per_unit)
    world_from_lerobot_base = (
        world_from_usd_base @ usd_base_from_lerobot_base
    )

    target_world_m = (
        np.asarray(target_world_stage_units, dtype=np.float64) * meters_per_unit
    )
    target_world_homogeneous = np.append(target_world_m, 1.0)
    target_lerobot_m = (
        np.linalg.inv(world_from_lerobot_base) @ target_world_homogeneous
    )[:3]
    fingertip_offset_m = (
        np.asarray(fingertip_offset_stage_units, dtype=np.float64)
        * meters_per_unit
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
    violations = [
        name
        for name, value, index in zip(
            ARM_JOINT_NAMES, solved_joints_rad, arm_indices
        )
        if value < lower_limits[index] or value > upper_limits[index]
    ]
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


def run_position_only_pregrasp_diagnostic(
    stage: Usd.Stage,
    robot: Articulation,
    base_link: XformPrim,
    fingertip_offset_stage_units: np.ndarray,
    target_world_stage_units: np.ndarray,
    meters_per_unit: float,
) -> None:
    """Test pre-grasp positional reachability without commanding the robot."""
    builtins.print(
        "\n[IK DIAGNOSTIC] Testing the same pre-grasp target with "
        "orientation disabled. No joints will be commanded.",
        flush=True,
    )
    try:
        result = calculate_position_only_ik(
            stage=stage,
            robot=robot,
            base_link=base_link,
            fingertip_offset_stage_units=fingertip_offset_stage_units,
            target_world_stage_units=target_world_stage_units,
            meters_per_unit=meters_per_unit,
        )
    except BaseException:
        builtins.print(
            "[IK DIAGNOSTIC] Position-only pre-grasp test FAILED. "
            "The issue is not limited to the full orientation constraint.",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return

    builtins.print(
        "[IK DIAGNOSTIC] Position-only pre-grasp test PASSED: "
        f"error={float(result['position_error_m']) * 1000.0:.3f} mm, "
        f"iterations={result['iterations']}. "
        "If full pose IK now fails, orientation is the limiting constraint.",
        flush=True,
    )


def draw_pre_ik_points(
    contact_point_world: np.ndarray,
    sphere_grasp_point: np.ndarray,
    pregrasp_point: np.ndarray,
) -> None:
    """Draw the authored contact point plus top-down approach targets."""
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    draw.clear_lines()
    draw.draw_points(
        [
            contact_point_world.tolist(),
            pregrasp_point.tolist(),
            sphere_grasp_point.tolist(),
        ],
        [
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 0.65, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ],
        [18.0, 18.0, 18.0],
    )
    draw.draw_lines(
        [pregrasp_point.tolist()],
        [sphere_grasp_point.tolist()],
        [(1.0, 0.65, 0.0, 1.0)],
        [3.0],
    )

    print()
    print("Debug draw legend:")
    print("  CYAN current authored contact point")
    print("  ORANGE top-down pre-grasp point and descent path")
    print("  RED  selected sphere grasp point")


class SphereFingerContactTracker:
    """Track sphere contact with each gripper body from PhysX callbacks."""

    def __init__(self, sphere_prim: Usd.Prim) -> None:
        self._sphere_path = SPHERE_PRIM_PATH
        self._finger_paths = {
            FIXED_FINGER_PRIM_PATH: "fixed",
            f"{ROBOT_PRIM_PATH}/jaw": "moving",
        }
        self._active_fingers: set[str] = set()

        # This applies only to the opened stage in this process.  The world
        # asset is not saved or otherwise modified.
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

    def print_status(self) -> None:
        print(
            "Sphere finger-contact status: "
            f"fixed={self.fixed_finger_in_contact}, "
            f"moving={self.moving_finger_in_contact}"
        )


def apply_position_only_ik(
    robot: Articulation,
    contact_point: XformPrim,
    target_world_stage_units: np.ndarray,
    meters_per_unit: float,
    ik_result: dict[str, object],
) -> np.ndarray:
    """Move toward IK and stop as soon as the fixed contact point is close."""
    arm_indices = resolve_arm_dof_indices(robot)
    solved_joints_deg = np.asarray(
        ik_result["solved_joints_deg"], dtype=np.float64
    )
    solved_joints_rad = np.deg2rad(solved_joints_deg)
    current_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    start_joints_rad = current_all[arm_indices].copy()

    print()
    print("Applying position-only IK result:")
    print(f"  Start joints (rad): {start_joints_rad.tolist()}")
    print(f"  Target joints (rad): {solved_joints_rad.tolist()}")

    target_world_m = (
        np.asarray(target_world_stage_units, dtype=np.float64) * meters_per_unit
    )
    stop_distance_m = ARGS.approach_stop_distance_mm / 1000.0
    approach_reached = False
    applied_steps = 0

    print(
        "  Contact-point stop distance: "
        f"{stop_distance_m * 1000.0:.3f} mm"
    )

    # Send a smooth joint-space trajectory through the articulation's ordinary
    # position drives. Stop immediately once the authored fixed-finger contact
    # point is close to the selected clearance point, rather than always
    # sending the complete trajectory and an arbitrary hold period.
    for step in range(1, IK_COMMAND_STEPS + 1):
        fraction = step / IK_COMMAND_STEPS
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        command = start_joints_rad + blend * (
            solved_joints_rad - start_joints_rad
        )
        robot.set_dof_position_targets(command, dof_indices=arm_indices)
        SIMULATION_APP.update()

        contact_positions, _ = contact_point.get_world_poses()
        actual_fingertip_stage_units = as_numpy(contact_positions)[0]
        fingertip_error_m = float(
            np.linalg.norm(
                actual_fingertip_stage_units * meters_per_unit - target_world_m
            )
        )
        applied_steps = step
        if fingertip_error_m <= stop_distance_m:
            approach_reached = True

            # The last interpolated command can be slightly ahead of the
            # measured pose. Replace it with the live arm pose so the position
            # drives do not continue advancing after this loop terminates.
            actual_all = as_numpy(
                robot.get_dof_positions()
            )[0].astype(np.float64)
            stopped_joints_rad = actual_all[arm_indices].copy()
            robot.set_dof_position_targets(
                stopped_joints_rad,
                dof_indices=arm_indices,
            )
            builtins.print(
                "[CONTACT POINT ACTIVATED] "
                f"step={step}, "
                f"distance={fingertip_error_m * 1000.0:.3f} mm, "
                f"threshold={stop_distance_m * 1000.0:.3f} mm",
                flush=True,
            )
            break

    actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    actual_joints_rad = actual_all[arm_indices]
    joint_errors_rad = actual_joints_rad - solved_joints_rad
    contact_positions, _ = contact_point.get_world_poses()
    actual_fingertip_stage_units = as_numpy(contact_positions)[0]
    actual_fingertip_m = actual_fingertip_stage_units * meters_per_unit
    fingertip_error_m = float(
        np.linalg.norm(actual_fingertip_m - target_world_m)
    )

    print(f"  Applied approach steps: {applied_steps}")
    print(
        "  Approach stop threshold: "
        f"{stop_distance_m * 1000.0:.3f} mm"
    )
    print(f"  Actual joints (rad): {actual_joints_rad.tolist()}")
    print(f"  Joint tracking errors (rad): {joint_errors_rad.tolist()}")
    print_point("Applied IK fingertip (world)", actual_fingertip_m)
    print_point("Applied IK target (world)", target_world_m)
    print(f"  Measured fingertip position error: {fingertip_error_m * 1000.0:.3f} mm")

    if not approach_reached:
        print(
            "  WARNING: maximum approach steps were reached before the "
            "contact-point threshold. Holding the measured arm pose."
        )
    return actual_joints_rad.copy()


def rotation_error_rad(
    actual_rotation: np.ndarray, target_rotation: np.ndarray
) -> float:
    """Return the shortest angular distance between two world rotations."""
    relative_rotation = target_rotation.T @ actual_rotation
    cosine = float((np.trace(relative_rotation) - 1.0) * 0.5)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def fixed_fingertip_world_pose(
    gripper: XformPrim,
    contact_point: XformPrim,
    meters_per_unit: float,
) -> np.ndarray:
    """Return the live pose of the temporary fixed-fingertip IK frame."""
    gripper_positions, gripper_orientations = gripper.get_world_poses()
    contact_positions, _ = contact_point.get_world_poses()
    return make_transform(
        as_numpy(contact_positions)[0].astype(np.float64) * meters_per_unit,
        rotation_matrix_wxyz(as_numpy(gripper_orientations)[0])
        @ rotation_matrix_rpy(np.array([0.0, math.pi, 0.0])),
    )


def world_from_lerobot_base_transform(
    stage: Usd.Stage,
    base_link: XformPrim,
    meters_per_unit: float,
) -> np.ndarray:
    """Bridge the imported USD base frame and LeRobot's URDF base frame."""
    usd_base_to_shoulder_zero = usd_joint_zero_transform(stage, "shoulder_pan")
    lerobot_base_to_shoulder_zero = urdf_joint_origin("shoulder_pan")
    usd_base_from_lerobot_base = (
        usd_base_to_shoulder_zero
        @ np.linalg.inv(lerobot_base_to_shoulder_zero)
    )
    return (
        world_transform_meters(base_link, meters_per_unit)
        @ usd_base_from_lerobot_base
    )


def calculate_pose_ik_path(
    stage: Usd.Stage,
    robot: Articulation,
    base_link: XformPrim,
    fingertip_offset_stage_units: np.ndarray,
    target_world_poses_stage_units: list[np.ndarray],
    meters_per_unit: float,
    label: str,
) -> dict[str, object]:
    """Calculate a warm-started fixed-fingertip world-pose IK path."""
    if not target_world_poses_stage_units:
        raise ValueError("Pose IK path must contain at least one target pose")

    target_world_poses_stage = np.asarray(
        target_world_poses_stage_units, dtype=np.float64
    )
    if target_world_poses_stage.ndim != 3 or target_world_poses_stage.shape[1:] != (
        4,
        4,
    ):
        raise ValueError(
            "Pose IK path must contain homogeneous transforms of shape (4, 4)"
        )

    arm_indices = resolve_arm_dof_indices(robot)
    all_joint_positions = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    seed_joints_rad = all_joint_positions[arm_indices]
    seed_joints_deg = np.rad2deg(seed_joints_rad)
    world_from_lerobot_base = world_from_lerobot_base_transform(
        stage, base_link, meters_per_unit
    )
    lerobot_from_world = np.linalg.inv(world_from_lerobot_base)
    target_lerobot_poses_m = target_world_poses_stage.copy()
    target_lerobot_poses_m[:, :3, 3] *= meters_per_unit
    target_lerobot_poses_m = np.asarray(
        [
            lerobot_from_world @ target_world_pose_m
            for target_world_pose_m in target_lerobot_poses_m
        ],
        dtype=np.float64,
    )
    fingertip_offset_m = (
        np.asarray(fingertip_offset_stage_units, dtype=np.float64)
        * meters_per_unit
    )

    print()
    print(f"{label} pose IK inputs:")
    print(f"  Waypoints: {len(target_world_poses_stage)}")
    print(f"  Seed joints (deg): {seed_joints_deg.tolist()}")
    print(
        "  Fixed fingertip offset in gripper frame (m): "
        f"{fingertip_offset_m.tolist()}"
    )
    print_point(
        "  First target fingertip position (world)",
        target_world_poses_stage[0, :3, 3] * meters_per_unit,
    )
    print_point(
        "  Final target fingertip position (world)",
        target_world_poses_stage[-1, :3, 3] * meters_per_unit,
    )
    builtins.print(
        "[IK DIAGNOSTIC] First target pose in LeRobot base frame (meters):\n"
        f"{np.array2string(target_lerobot_poses_m[0], precision=8, suppress_small=False)}",
        flush=True,
    )
    builtins.print(
        "[IK DIAGNOSTIC] First target rotation checks: "
        f"det={np.linalg.det(target_lerobot_poses_m[0, :3, :3]):.9f}, "
        f"orthonormal_error="
        f"{np.linalg.norm(target_lerobot_poses_m[0, :3, :3].T @ target_lerobot_poses_m[0, :3, :3] - np.eye(3)):.3e}",
        flush=True,
    )

    result = run_lerobot_worker(
        {
            "operation": "pose_ik_path",
            "seed_joints_deg": seed_joints_deg.tolist(),
            "fingertip_offset_m": fingertip_offset_m.tolist(),
            "target_poses_m": target_lerobot_poses_m.tolist(),
            "max_iterations": IK_MAX_ITERATIONS,
            "position_tolerance_m": IK_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": IK_ORIENTATION_TOLERANCE_RAD,
            "position_weight": IK_POSITION_WEIGHT,
            "orientation_weight": IK_ORIENTATION_WEIGHT,
        }
    )
    waypoints = result.get("waypoints")
    builtins.print(
        f"[IK DIAGNOSTIC] {label} raw status={result.get('status')!r}, "
        f"completed_waypoints={result.get('completed_waypoints')}, "
        f"requested_waypoints={result.get('target_waypoints')}",
        flush=True,
    )
    if isinstance(waypoints, list):
        for waypoint in waypoints:
            builtins.print(
                "[IK DIAGNOSTIC] "
                f"waypoint={waypoint.get('index')}, "
                f"status={waypoint.get('status')}, "
                f"initial_position_error="
                f"{float(waypoint.get('initial_position_error_m', float('nan'))) * 1000.0:.3f} mm, "
                f"initial_orientation_error="
                f"{math.degrees(float(waypoint.get('initial_orientation_error_rad', float('nan')))):.3f} deg, "
                f"final_position_error="
                f"{float(waypoint.get('position_error_m', float('nan'))) * 1000.0:.3f} mm, "
                f"final_orientation_error="
                f"{math.degrees(float(waypoint.get('orientation_error_rad', float('nan')))):.3f} deg, "
                f"iterations={waypoint.get('iterations')}",
                flush=True,
            )
            builtins.print(
                "[IK DIAGNOSTIC] "
                f"waypoint={waypoint.get('index')} solved_joints_deg="
                f"{waypoint.get('solved_joints_deg')}",
                flush=True,
            )

    if not isinstance(waypoints, list) or len(waypoints) != len(
        target_world_poses_stage
    ):
        raise RuntimeError(
            "LeRobot pose IK returned an incomplete waypoint result: "
            f"expected={len(target_world_poses_stage)}, got={waypoints}"
        )
    if result.get("status") != "converged":
        failed_waypoint = waypoints[-1]
        raise RuntimeError(
            f"{label} pose IK did not converge at waypoint "
            f"{failed_waypoint.get('index')}: position_error="
            f"{float(failed_waypoint.get('position_error_m', float('nan'))) * 1000.0:.3f} mm, "
            f"orientation_error="
            f"{math.degrees(float(failed_waypoint.get('orientation_error_rad', float('nan')))):.3f} deg"
        )

    lower_limits, upper_limits = robot.get_dof_limits()
    lower_limits = as_numpy(lower_limits)[0]
    upper_limits = as_numpy(upper_limits)[0]
    for waypoint in waypoints:
        solved_joints_deg = np.asarray(
            waypoint["solved_joints_deg"], dtype=np.float64
        )
        solved_joints_rad = np.deg2rad(solved_joints_deg)
        if solved_joints_rad.shape != (len(ARM_JOINT_NAMES),):
            raise RuntimeError(
                "LeRobot returned an invalid pose IK joint vector: "
                f"{solved_joints_deg.tolist()}"
            )
        if not np.all(np.isfinite(solved_joints_rad)):
            raise RuntimeError("LeRobot returned NaN or infinite pose IK joints")
        violations = [
            name
            for name, value, index in zip(
                ARM_JOINT_NAMES, solved_joints_rad, arm_indices
            )
            if value < lower_limits[index] or value > upper_limits[index]
        ]
        if violations:
            raise RuntimeError(
                "Pose IK solution violates joint limits at waypoint "
                f"{waypoint['index']}: {violations}"
            )

    final_waypoint = waypoints[-1]
    print(f"{label} pose IK result:")
    print(f"  Status: {result['status']}")
    print(f"  Final joints (deg): {final_waypoint['solved_joints_deg']}")
    print(
        "  Final position error: "
        f"{float(final_waypoint['position_error_m']) * 1000.0:.3f} mm"
    )
    print(
        "  Final orientation error: "
        f"{math.degrees(float(final_waypoint['orientation_error_rad'])):.3f} deg"
    )
    return result


def apply_pose_ik_path(
    robot: Articulation,
    gripper: XformPrim,
    contact_point: XformPrim,
    target_world_poses_stage_units: list[np.ndarray],
    meters_per_unit: float,
    ik_result: dict[str, object],
    stop_distance_m: float,
    label: str,
) -> np.ndarray:
    """Command a pose-IK path and hold at its final measured arm pose."""
    waypoints = ik_result["waypoints"]
    if not isinstance(waypoints, list) or len(waypoints) != len(
        target_world_poses_stage_units
    ):
        raise RuntimeError("Pose IK result does not match the requested path")

    arm_indices = resolve_arm_dof_indices(robot)
    target_world_poses_m = [
        np.asarray(target_pose, dtype=np.float64).copy()
        for target_pose in target_world_poses_stage_units
    ]
    for target_pose_m in target_world_poses_m:
        target_pose_m[:3, 3] *= meters_per_unit

    print()
    print(f"Applying {label} pose IK path:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Position stop distance: {stop_distance_m * 1000.0:.3f} mm")
    print(
        "  Orientation stop tolerance: "
        f"{math.degrees(IK_ORIENTATION_TOLERANCE_RAD):.3f} deg"
    )

    for waypoint_index, (waypoint, target_world_pose_m) in enumerate(
        zip(waypoints, target_world_poses_m)
    ):
        solved_joints_rad = np.deg2rad(
            np.asarray(waypoint["solved_joints_deg"], dtype=np.float64)
        )
        current_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
        start_joints_rad = current_all[arm_indices].copy()
        waypoint_reached = False

        for step in range(1, POSE_IK_WAYPOINT_COMMAND_STEPS + 1):
            fraction = step / POSE_IK_WAYPOINT_COMMAND_STEPS
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            arm_command = start_joints_rad + blend * (
                solved_joints_rad - start_joints_rad
            )
            robot.set_dof_position_targets(arm_command, dof_indices=arm_indices)
            SIMULATION_APP.update()

            actual_pose_m = fixed_fingertip_world_pose(
                gripper, contact_point, meters_per_unit
            )
            position_error_m = float(
                np.linalg.norm(
                    actual_pose_m[:3, 3] - target_world_pose_m[:3, 3]
                )
            )
            orientation_error = rotation_error_rad(
                actual_pose_m[:3, :3], target_world_pose_m[:3, :3]
            )
            if (
                waypoint_index == len(waypoints) - 1
                and position_error_m <= stop_distance_m
                and orientation_error <= IK_ORIENTATION_TOLERANCE_RAD
            ):
                waypoint_reached = True
                break

        actual_pose_m = fixed_fingertip_world_pose(
            gripper, contact_point, meters_per_unit
        )
        position_error_m = float(
            np.linalg.norm(
                actual_pose_m[:3, 3] - target_world_pose_m[:3, 3]
            )
        )
        orientation_error = rotation_error_rad(
            actual_pose_m[:3, :3], target_world_pose_m[:3, :3]
        )
        print(
            f"  Waypoint {waypoint_index + 1}/{len(waypoints)}: "
            f"position_error={position_error_m * 1000.0:.3f} mm, "
            f"orientation_error={math.degrees(orientation_error):.3f} deg"
        )
        if waypoint_index != len(waypoints) - 1:
            continue
        if not waypoint_reached:
            actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
            robot.set_dof_position_targets(
                actual_all[arm_indices], dof_indices=arm_indices
            )
            raise RuntimeError(
                f"{label} did not reach its final pose before the waypoint "
                f"command limit: position_error={position_error_m * 1000.0:.3f} mm, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )

    actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    arm_hold_target_rad = actual_all[arm_indices].copy()
    robot.set_dof_position_targets(arm_hold_target_rad, dof_indices=arm_indices)
    builtins.print(f"[{label.upper()} COMPLETED]", flush=True)
    return arm_hold_target_rad


def close_gripper_until_stable_grasp(
    robot: Articulation,
    arm_hold_target_rad: np.ndarray,
    contact_tracker: SphereFingerContactTracker,
) -> float:
    """Close until stable and keep the fully closed target active."""
    arm_indices = resolve_arm_dof_indices(robot)
    arm_target_rad = np.asarray(arm_hold_target_rad, dtype=np.float64)
    lower_limits, _ = robot.get_dof_limits()
    closed_target_rad = float(
        as_numpy(lower_limits)[0, GRIPPER_DOF_INDEX]
    )

    print()
    print("Closing gripper until stable two-finger grasp:")
    print(f"  Input index: {GRIPPER_DOF_INDEX}")
    print(f"  Fully closed target (rad): {closed_target_rad:.6f}")
    print(
        "  Stable grasp requirements: "
        f"{GRASP_STABLE_CONTACT_FRAMES} consecutive physics frames, "
        f"remaining close gap >= {GRIPPER_REMAINING_CLOSE_GAP_THRESHOLD_RAD:.3f} rad"
    )

    qualifying_frames = 0
    step = 0
    while SIMULATION_APP.is_running():
        step += 1
        robot.set_dof_position_targets(arm_target_rad, dof_indices=arm_indices)
        robot.set_dof_position_targets(
            [closed_target_rad], dof_indices=[GRIPPER_DOF_INDEX]
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
        target_still_farther_closed = (
            target_gap_rad >= GRIPPER_REMAINING_CLOSE_GAP_THRESHOLD_RAD
        )
        qualifies = (
            fixed_contact
            and moving_contact
            and target_still_farther_closed
        )

        if qualifies:
            qualifying_frames += 1
            if qualifying_frames == 1:
                print("  Stable-grasp counter started")
            if qualifying_frames >= GRASP_STABLE_CONTACT_FRAMES:
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
                    f"keeping closed gripper target={closed_target_rad:.6f} rad "
                    "to maintain squeeze force during lift.",
                    flush=True,
                )
                return closed_target_rad
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

    raise RuntimeError("Simulation stopped before a stable grasp was detected")


def lift_grasped_sphere(
    robot: Articulation,
    gripper: XformPrim,
    contact_point: XformPrim,
    target_world_poses_stage_units: list[np.ndarray],
    meters_per_unit: float,
    ik_result: dict[str, object],
    gripper_target_rad: float,
    contact_tracker: SphereFingerContactTracker,
) -> np.ndarray:
    """Lift vertically through pose IK while holding the finalized grasp."""
    waypoints = ik_result["waypoints"]
    if not isinstance(waypoints, list) or len(waypoints) != len(
        target_world_poses_stage_units
    ):
        raise RuntimeError("Lift pose IK result does not match the requested path")

    arm_indices = resolve_arm_dof_indices(robot)
    target_world_poses_m = [
        np.asarray(target_pose, dtype=np.float64).copy()
        for target_pose in target_world_poses_stage_units
    ]
    for target_pose_m in target_world_poses_m:
        target_pose_m[:3, 3] *= meters_per_unit

    stop_distance_m = ARGS.lift_stop_distance_mm / 1000.0
    contact_loss_frames = 0
    total_steps = 0

    print()
    print("Lifting grasped sphere with orientation-holding pose IK:")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Constant gripper target (rad): {gripper_target_rad:.6f}")
    print(f"  Lift height: {ARGS.lift_height_mm:.3f} mm")
    print(f"  Lift stop distance: {ARGS.lift_stop_distance_mm:.3f} mm")
    print(
        "  Orientation stop tolerance: "
        f"{math.degrees(IK_ORIENTATION_TOLERANCE_RAD):.3f} deg"
    )

    for waypoint_index, (waypoint, target_world_pose_m) in enumerate(
        zip(waypoints, target_world_poses_m)
    ):
        solved_joints_rad = np.deg2rad(
            np.asarray(waypoint["solved_joints_deg"], dtype=np.float64)
        )
        current_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
        start_joints_rad = current_all[arm_indices].copy()
        waypoint_reached = False

        for step in range(1, POSE_IK_WAYPOINT_COMMAND_STEPS + 1):
            total_steps += 1
            fraction = step / POSE_IK_WAYPOINT_COMMAND_STEPS
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            arm_command = start_joints_rad + blend * (
                solved_joints_rad - start_joints_rad
            )
            robot.set_dof_position_targets(arm_command, dof_indices=arm_indices)
            robot.set_dof_position_targets(
                [gripper_target_rad], dof_indices=[GRIPPER_DOF_INDEX]
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
                        actual_all[arm_indices], dof_indices=arm_indices
                    )
                    robot.set_dof_position_targets(
                        [gripper_target_rad], dof_indices=[GRIPPER_DOF_INDEX]
                    )
                    raise RuntimeError(
                        "Stable grasp was lost during orientation-holding lift: "
                        f"step={total_steps}, fixed_contact={fixed_contact}, "
                        f"moving_contact={moving_contact}"
                    )

            actual_pose_m = fixed_fingertip_world_pose(
                gripper, contact_point, meters_per_unit
            )
            position_error_m = float(
                np.linalg.norm(
                    actual_pose_m[:3, 3] - target_world_pose_m[:3, 3]
                )
            )
            orientation_error = rotation_error_rad(
                actual_pose_m[:3, :3], target_world_pose_m[:3, :3]
            )
            if (
                waypoint_index == len(waypoints) - 1
                and position_error_m <= stop_distance_m
                and orientation_error <= IK_ORIENTATION_TOLERANCE_RAD
            ):
                waypoint_reached = True
                break

        actual_pose_m = fixed_fingertip_world_pose(
            gripper, contact_point, meters_per_unit
        )
        position_error_m = float(
            np.linalg.norm(
                actual_pose_m[:3, 3] - target_world_pose_m[:3, 3]
            )
        )
        orientation_error = rotation_error_rad(
            actual_pose_m[:3, :3], target_world_pose_m[:3, :3]
        )
        print(
            f"  Waypoint {waypoint_index + 1}/{len(waypoints)}: "
            f"position_error={position_error_m * 1000.0:.3f} mm, "
            f"orientation_error={math.degrees(orientation_error):.3f} deg"
        )
        if waypoint_index != len(waypoints) - 1:
            continue
        if not waypoint_reached:
            actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
            robot.set_dof_position_targets(
                actual_all[arm_indices], dof_indices=arm_indices
            )
            robot.set_dof_position_targets(
                [gripper_target_rad], dof_indices=[GRIPPER_DOF_INDEX]
            )
            raise RuntimeError(
                "Lift did not reach its final pose before the waypoint command "
                f"limit: position_error={position_error_m * 1000.0:.3f} mm, "
                f"orientation_error={math.degrees(orientation_error):.3f} deg"
            )

    actual_all = as_numpy(robot.get_dof_positions())[0].astype(np.float64)
    lifted_arm_hold_rad = actual_all[arm_indices].copy()
    robot.set_dof_position_targets(
        lifted_arm_hold_rad, dof_indices=arm_indices
    )
    robot.set_dof_position_targets(
        [gripper_target_rad], dof_indices=[GRIPPER_DOF_INDEX]
    )
    builtins.print(
        "[LIFT COMPLETED] "
        f"step={total_steps}, "
        f"distance={position_error_m * 1000.0:.3f} mm, "
        f"orientation_error={math.degrees(orientation_error):.3f} deg, "
        f"threshold={stop_distance_m * 1000.0:.3f} mm",
        flush=True,
    )
    return lifted_arm_hold_rad


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


def sphere_world_radius(sphere_prim: Usd.Prim) -> float:
    """Return the radius of a uniformly scaled sphere in stage units."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    size = np.asarray(
        cache.ComputeWorldBound(sphere_prim).ComputeAlignedRange().GetSize(),
        dtype=np.float64,
    )
    radii = 0.5 * size
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise RuntimeError(f"Invalid sphere world bounds: {size.tolist()}")
    if not np.allclose(radii, radii[0], rtol=1e-3, atol=1e-6):
        raise RuntimeError(
            "The target prim is not spherical: "
            f"radii={radii.tolist()}"
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


def top_down_grasp_rotation(
    sphere_center: np.ndarray,
    sphere_surface_grasp_point: np.ndarray,
) -> np.ndarray:
    """Build the fixed-fingertip rotation for an equatorial top-down grasp."""
    tool_x = np.asarray(
        sphere_surface_grasp_point, dtype=np.float64
    ) - np.asarray(sphere_center, dtype=np.float64)
    tool_x[2] = 0.0
    tool_x_length = float(np.linalg.norm(tool_x))
    if tool_x_length <= 1e-12:
        raise RuntimeError("The selected sphere contact side is not horizontal")
    tool_x /= tool_x_length

    tool_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    rotation = np.column_stack((tool_x, tool_y, tool_z))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise RuntimeError("Top-down grasp rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
        raise RuntimeError("Top-down grasp rotation is not right-handed")
    return rotation


def vertical_pose_waypoints(
    start_position_stage_units: np.ndarray,
    end_position_stage_units: np.ndarray,
    rotation: np.ndarray,
    meters_per_unit: float,
) -> list[np.ndarray]:
    """Return evenly spaced vertical world poses, excluding the start pose."""
    start = np.asarray(start_position_stage_units, dtype=np.float64)
    end = np.asarray(end_position_stage_units, dtype=np.float64)
    horizontal_delta = float(np.linalg.norm((end - start)[:2]))
    if horizontal_delta > 1e-9:
        raise RuntimeError(
            "Top-down waypoint path must be vertical, but its horizontal "
            f"displacement is {horizontal_delta} stage units"
        )
    distance_m = abs(float(end[2] - start[2])) * meters_per_unit
    waypoint_count = max(
        1,
        math.ceil(distance_m / POSE_IK_WAYPOINT_SPACING_M - 1e-12),
    )
    return [
        make_transform(start + (end - start) * (index / waypoint_count), rotation)
        for index in range(1, waypoint_count + 1)
    ]


def print_point(label: str, point_meters: np.ndarray) -> None:
    print(
        f"{label}: "
        f"x={point_meters[0]:.6f} m, "
        f"y={point_meters[1]:.6f} m, "
        f"z={point_meters[2]:.6f} m"
    )


def main() -> None:
    initialize_lerobot()

    world_path = ARGS.world.expanduser().resolve()
    if not world_path.is_file():
        raise FileNotFoundError(f"World USD does not exist: {world_path}")

    opened, stage = stage_utils.open_stage(str(world_path))
    if not opened or stage is None:
        raise RuntimeError(f"Isaac Sim could not open world: {world_path}")

    require_prim(stage, FIXED_FINGER_PRIM_PATH)
    require_prim(stage, CONTACT_POINT_PRIM_PATH)
    require_prim(stage, BASE_LINK_PRIM_PATH)
    require_prim(stage, BASE_REFERENCE_PRIM_PATH)
    sphere_prim = require_prim(stage, SPHERE_PRIM_PATH)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit: {meters_per_unit}")

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    robot = Articulation(ROBOT_PRIM_PATH)
    base_link = XformPrim(BASE_LINK_PRIM_PATH)
    base_reference = XformPrim(BASE_REFERENCE_PRIM_PATH)
    gripper = XformPrim(FIXED_FINGER_PRIM_PATH)
    contact_point = XformPrim(CONTACT_POINT_PRIM_PATH)
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

    sphere_positions, _ = sphere.get_world_poses()
    sphere_center = as_numpy(sphere_positions)[0]
    shoulder_pan_angle = align_base_with_sphere(
        robot,
        base_reference,
        base_link,
        sphere_center,
    )
    command_ik_seed_presets(robot)

    # Re-read all live poses because alignment and the pre-IK configuration
    # commands advance physics. The authored contact point replaces the prior
    # mesh-derived fingertip estimate everywhere below.
    gripper_positions, gripper_orientations = gripper.get_world_poses()
    gripper_position = as_numpy(gripper_positions)[0]
    gripper_orientation = as_numpy(gripper_orientations)[0]
    contact_positions, _ = contact_point.get_world_poses()
    contact_point_world = as_numpy(contact_positions)[0]
    contact_point_local_stage_units = inverse_rotate_vector_wxyz(
        gripper_orientation,
        contact_point_world - gripper_position,
    )

    sphere_positions, _ = sphere.get_world_poses()
    sphere_center = as_numpy(sphere_positions)[0]
    sphere_radius = sphere_world_radius(sphere_prim)
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
        SPHERE_SURFACE_CLEARANCE_M / meters_per_unit,
    )
    top_down_rotation = top_down_grasp_rotation(
        sphere_center, sphere_surface_grasp_point
    )
    pregrasp_point = sphere_grasp_point.copy()
    pregrasp_point[2] = (
        sphere_center[2]
        + sphere_radius
        + ARGS.pregrasp_clearance_mm / 1000.0 / meters_per_unit
    )
    pregrasp_pose = make_transform(pregrasp_point, top_down_rotation)
    descent_poses = vertical_pose_waypoints(
        pregrasp_point,
        sphere_grasp_point,
        top_down_rotation,
        meters_per_unit,
    )

    print()
    print(
        "Aligned base heading (world horizontal unit vector): "
        f"x={heading[0]:.6f}, y={heading[1]:.6f}, z={heading[2]:.6f}"
    )
    print_point("Authored contact point (world)", contact_point_world * meters_per_unit)
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
        f"IK approach point ({SPHERE_SURFACE_CLEARANCE_M * 1000.0:.1f} mm clearance, world)",
        sphere_grasp_point * meters_per_unit,
    )
    print_point(
        f"Top-down pre-grasp point ({ARGS.pregrasp_clearance_mm:.1f} mm above sphere top, world)",
        pregrasp_point * meters_per_unit,
    )
    print(
        "Top-down fixed-fingertip axes in world: "
        f"x={top_down_rotation[:, 0].tolist()}, "
        f"y={top_down_rotation[:, 1].tolist()}, "
        f"z={top_down_rotation[:, 2].tolist()}"
    )
    print(f"Top-down vertical-descent waypoints: {len(descent_poses)}")

    draw_pre_ik_points(contact_point_world, sphere_grasp_point, pregrasp_point)
    for _ in range(DEBUG_POINT_WAIT_STEPS):
        SIMULATION_APP.update()

    if DIAGNOSTIC_POSITION_ONLY_PREGRASP:
        run_position_only_pregrasp_diagnostic(
            stage=stage,
            robot=robot,
            base_link=base_link,
            fingertip_offset_stage_units=contact_point_local_stage_units,
            target_world_stage_units=pregrasp_point,
            meters_per_unit=meters_per_unit,
        )

    pregrasp_ik_result = calculate_pose_ik_path(
        stage=stage,
        robot=robot,
        base_link=base_link,
        fingertip_offset_stage_units=contact_point_local_stage_units,
        target_world_poses_stage_units=[pregrasp_pose],
        meters_per_unit=meters_per_unit,
        label="Top-down pre-grasp",
    )
    apply_pose_ik_path(
        robot=robot,
        gripper=gripper,
        contact_point=contact_point,
        target_world_poses_stage_units=[pregrasp_pose],
        meters_per_unit=meters_per_unit,
        ik_result=pregrasp_ik_result,
        stop_distance_m=ARGS.approach_stop_distance_mm / 1000.0,
        label="Top-down pre-grasp",
    )
    descent_ik_result = calculate_pose_ik_path(
        stage=stage,
        robot=robot,
        base_link=base_link,
        fingertip_offset_stage_units=contact_point_local_stage_units,
        target_world_poses_stage_units=descent_poses,
        meters_per_unit=meters_per_unit,
        label="Top-down descent",
    )
    arm_hold_target_rad = apply_pose_ik_path(
        robot=robot,
        gripper=gripper,
        contact_point=contact_point,
        target_world_poses_stage_units=descent_poses,
        meters_per_unit=meters_per_unit,
        ik_result=descent_ik_result,
        stop_distance_m=ARGS.approach_stop_distance_mm / 1000.0,
        label="Top-down descent",
    )
    gripper_close_target_rad = close_gripper_until_stable_grasp(
        robot=robot,
        arm_hold_target_rad=arm_hold_target_rad,
        contact_tracker=contact_tracker,
    )

    # Start the lift from the live post-grasp fingertip position. This makes
    # the requested displacement purely vertical even if the arm stopped
    # slightly short of its original approach target.
    contact_positions, _ = contact_point.get_world_poses()
    lift_start_world = as_numpy(contact_positions)[0].astype(np.float64)
    lift_target_world = lift_start_world.copy()
    lift_target_world[2] += (
        ARGS.lift_height_mm / 1000.0 / meters_per_unit
    )
    lift_poses = vertical_pose_waypoints(
        lift_start_world,
        lift_target_world,
        top_down_rotation,
        meters_per_unit,
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
    print(f"Orientation-holding lift waypoints: {len(lift_poses)}")

    lift_ik_result = calculate_pose_ik_path(
        stage=stage,
        robot=robot,
        base_link=base_link,
        fingertip_offset_stage_units=contact_point_local_stage_units,
        target_world_poses_stage_units=lift_poses,
        meters_per_unit=meters_per_unit,
        label="Orientation-holding lift",
    )
    lift_grasped_sphere(
        robot=robot,
        gripper=gripper,
        contact_point=contact_point,
        target_world_poses_stage_units=lift_poses,
        meters_per_unit=meters_per_unit,
        ik_result=lift_ik_result,
        gripper_target_rad=gripper_close_target_rad,
        contact_tracker=contact_tracker,
    )
    contact_tracker.print_status()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        builtins.print(
            "\n[FATAL DIAGNOSTIC] Unhandled exception before Isaac shutdown: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        builtins.print(
            "[IK DIAGNOSTIC] Closing Isaac SimulationApp.",
            flush=True,
        )
    SIMULATION_APP.close()