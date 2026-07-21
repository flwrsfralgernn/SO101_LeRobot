from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics


# This script is located at:
# /home/rog/Downloads/so101_lerobot/scripts/ik/validate_lerobot_ik.py
#
# parents[0] -> scripts/ik
# parents[1] -> scripts
# parents[2] -> so101_lerobot
PROJECT_ROOT = Path(__file__).resolve().parents[2]

URDF_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.urdf"
)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

MAX_IK_STEPS = 100
POSITION_TOLERANCE_M = 0.0005


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"URDF path: {URDF_PATH}")

    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    print("URDF found successfully.")

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES,
    )

    # LeRobot RobotKinematics expects joint angles in degrees.
    seed_joints_deg = np.array(
        [0.0, -30.0, 60.0, 30.0, 0.0],
        dtype=np.float64,
    )

    start_pose = kinematics.forward_kinematics(seed_joints_deg)

    # Create a small, reachable test target:
    # move the gripper upward by 1 cm.
    target_pose = start_pose.copy()
    target_pose[:3, 3] += np.array(
        [0.0, 0.0, 0.01],
        dtype=np.float64,
    )

    joints_deg = seed_joints_deg.copy()
    solved_pose = start_pose.copy()
    position_error_m = float("inf")

    for step in range(1, MAX_IK_STEPS + 1):
        joints_deg = np.asarray(
            kinematics.inverse_kinematics(
                current_joint_pos=joints_deg,
                desired_ee_pose=target_pose,
                position_weight=1.0,
                orientation_weight=0.0,
            ),
            dtype=np.float64,
        )

        solved_pose = kinematics.forward_kinematics(joints_deg)

        position_error_m = float(
            np.linalg.norm(
                solved_pose[:3, 3] - target_pose[:3, 3]
            )
        )

        if position_error_m <= POSITION_TOLERANCE_M:
            break

    print()
    print("Seed joints in degrees:")
    print(seed_joints_deg)

    print()
    print("Starting end-effector position in meters:")
    print(start_pose[:3, 3])

    print()
    print("Requested end-effector position in meters:")
    print(target_pose[:3, 3])

    print()
    print("Solved joints in degrees:")
    print(joints_deg)

    print()
    print("Solved joints in radians for future Isaac Sim use:")
    print(np.deg2rad(joints_deg))

    print()
    print("Solved end-effector position in meters:")
    print(solved_pose[:3, 3])

    print()
    print(f"IK iterations: {step}")
    print(f"Position error: {position_error_m * 1000.0:.3f} mm")

    if not np.all(np.isfinite(joints_deg)):
        raise RuntimeError("IK returned NaN or infinite joint values.")

    if position_error_m > POSITION_TOLERANCE_M:
        raise RuntimeError(
            "IK did not converge within the requested tolerance. "
            f"Final error: {position_error_m * 1000.0:.3f} mm"
        )

    print()
    print("IK validation passed.")


if __name__ == "__main__":
    main()