"""Focused regression tests for runtime-independent IK utilities."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from scripts.ik.kinematics_utils import (
    axis_alignment_error_rad,
    finite_array,
    forward_kinematics_checked,
    frame_point_meters_in_world_stage,
    inverse_transform_point,
    joint_limit_violations,
    make_transform,
    meters_to_stage,
    pose_residual,
    resolve_named_indices,
    rotation_matrix_rpy,
    rotation_matrix_wxyz,
    stage_to_meters,
    transform_point,
    world_rotation_in_frame,
    world_stage_point_in_frame_meters,
)
from scripts.ik.tool_model import fixed_tool_model_from_urdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SO101_URDF_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.urdf"
)


class ArrayAndUnitTests(unittest.TestCase):
    def test_finite_array_enforces_shape_and_finiteness(self) -> None:
        np.testing.assert_array_equal(
            finite_array([1, 2, 3], shape=(3,), label="point"),
            np.array([1.0, 2.0, 3.0]),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            finite_array([1, 2], shape=(3,), label="point")
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            finite_array([1, np.nan, 3], shape=(3,), label="point")

    def test_stage_meter_round_trip_for_centimeter_stage(self) -> None:
        stage_point = np.array([12.5, -3.0, 42.0])
        point_m = stage_to_meters(stage_point, 0.01)
        np.testing.assert_allclose(point_m, [0.125, -0.03, 0.42])
        np.testing.assert_allclose(meters_to_stage(point_m, 0.01), stage_point)

    def test_unit_conversion_rejects_invalid_scale(self) -> None:
        for invalid_scale in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(scale=invalid_scale):
                with self.assertRaises(ValueError):
                    stage_to_meters([1.0], invalid_scale)


class FrameTransformTests(unittest.TestCase):
    def test_wxyz_quaternion_uses_isaac_order(self) -> None:
        half_angle = math.pi / 4.0
        rotation = rotation_matrix_wxyz(
            [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
        )
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero"):
            rotation_matrix_wxyz([0.0, 0.0, 0.0, 0.0])

    def test_urdf_rpy_is_fixed_axis_xyz(self) -> None:
        rotation = rotation_matrix_rpy([0.0, 0.0, math.pi / 2.0])
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )

    def test_transform_and_inverse_round_trip(self) -> None:
        transform = make_transform(
            [0.3, -0.2, 0.5],
            rotation_matrix_rpy([0.2, -0.4, 0.7]),
        )
        point = np.array([0.1, 0.25, -0.3])
        np.testing.assert_allclose(
            inverse_transform_point(transform, transform_point(transform, point)),
            point,
            atol=1e-12,
        )

    def test_world_and_local_point_conversion_has_explicit_units(self) -> None:
        world_from_frame = make_transform(
            [1.0, 2.0, 3.0],
            rotation_matrix_rpy([0.0, 0.0, math.pi / 2.0]),
        )
        local_point_m = np.array([0.25, 0.0, -0.5])
        world_point_stage = frame_point_meters_in_world_stage(
            local_point_m,
            world_from_frame,
            0.01,
        )
        np.testing.assert_allclose(world_point_stage, [100.0, 225.0, 250.0])
        np.testing.assert_allclose(
            world_stage_point_in_frame_meters(
                world_point_stage,
                world_from_frame,
                0.01,
            ),
            local_point_m,
            atol=1e-12,
        )

    def test_world_rotation_is_expressed_in_local_frame(self) -> None:
        world_from_frame = make_transform(
            [4.0, 5.0, 6.0],
            rotation_matrix_rpy([0.0, 0.0, math.pi / 2.0]),
        )
        world_from_tool = rotation_matrix_rpy([0.0, 0.0, math.pi])
        expected_frame_from_tool = rotation_matrix_rpy(
            [0.0, 0.0, math.pi / 2.0]
        )
        np.testing.assert_allclose(
            world_rotation_in_frame(world_from_tool, world_from_frame),
            expected_frame_from_tool,
            atol=1e-12,
        )

    def test_axis_alignment_error_leaves_roll_about_axis_free(self) -> None:
        self.assertAlmostEqual(
            axis_alignment_error_rad([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]),
            0.0,
        )
        self.assertAlmostEqual(
            axis_alignment_error_rad([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            math.pi / 2.0,
        )
        self.assertAlmostEqual(
            axis_alignment_error_rad([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
            math.pi,
        )


class JointAndForwardKinematicsTests(unittest.TestCase):
    JOINT_NAMES = ["shoulder", "elbow", "wrist"]

    def test_joint_names_are_reordered_deterministically(self) -> None:
        self.assertEqual(
            resolve_named_indices(
                ["gripper", "wrist", "shoulder", "elbow"],
                self.JOINT_NAMES,
            ),
            [2, 3, 1],
        )

    def test_joint_name_validation_rejects_missing_or_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required joints"):
            resolve_named_indices(["shoulder", "elbow"], self.JOINT_NAMES)
        with self.assertRaisesRegex(ValueError, "not unique"):
            resolve_named_indices(
                ["shoulder", "elbow", "elbow", "wrist"],
                self.JOINT_NAMES,
            )

    def test_joint_limits_are_inclusive_and_support_tolerance(self) -> None:
        lower = [-1.0, -2.0, -3.0]
        upper = [1.0, 2.0, 3.0]
        self.assertEqual(
            joint_limit_violations(
                [-1.0, 2.0, 3.0005],
                lower,
                upper,
                self.JOINT_NAMES,
                tolerance=0.001,
            ),
            [],
        )
        self.assertEqual(
            joint_limit_violations(
                [-1.1, 0.0, 3.1],
                lower,
                upper,
                self.JOINT_NAMES,
            ),
            ["shoulder", "wrist"],
        )

    def test_invalid_joint_limit_interval_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower joint limits exceed"):
            joint_limit_violations(
                [0.0, 0.0, 0.0],
                [1.0, -2.0, -3.0],
                [0.0, 2.0, 3.0],
                self.JOINT_NAMES,
            )

    def test_checked_fk_validates_input_and_output(self) -> None:
        def fake_fk(joints: np.ndarray) -> np.ndarray:
            return make_transform(
                [joints[0], joints[1], joints[2]],
                np.eye(3),
            )

        pose = forward_kinematics_checked(
            fake_fk,
            [0.1, 0.2, 0.3],
            joint_count=3,
        )
        np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "FK joint vector"):
            forward_kinematics_checked(fake_fk, [0.1, 0.2], joint_count=3)

    def test_pose_residual_reports_position_and_orientation_separately(self) -> None:
        actual = make_transform([0.1, 0.0, 0.0], np.eye(3))
        desired = make_transform(
            [0.0, 0.0, 0.0],
            rotation_matrix_rpy([0.0, 0.0, math.pi / 2.0]),
        )
        position_error_m, orientation_error_rad = pose_residual(actual, desired)
        self.assertAlmostEqual(position_error_m, 0.1)
        self.assertAlmostEqual(orientation_error_rad, math.pi / 2.0)


class FixedToolModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = fixed_tool_model_from_urdf(
            SO101_URDF_PATH,
            joint_name="gripper_frame_joint",
            expected_parent_link="gripper_link",
            expected_tool_link="gripper_frame_link",
        )

    def test_official_so101_tool_origin_is_loaded_from_urdf(self) -> None:
        np.testing.assert_allclose(
            self.model.position_in_parent_m,
            [-0.0079, -0.000218121, -0.0981274],
            rtol=0.0,
            atol=1e-12,
        )

    def test_tool_axes_are_orthogonal_and_follow_urdf_rotation(self) -> None:
        approach = self.model.approach_axis_in_parent
        closing = self.model.closing_axis_in_parent
        self.assertAlmostEqual(float(np.linalg.norm(approach)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(closing)), 1.0)
        self.assertAlmostEqual(float(np.dot(approach, closing)), 0.0)
        np.testing.assert_allclose(approach, [0.0, 0.0, -1.0], atol=3e-6)
        np.testing.assert_allclose(closing, [1.0, 0.0, 0.0], atol=3e-6)


if __name__ == "__main__":
    unittest.main()
