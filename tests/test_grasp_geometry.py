import unittest

import numpy as np

from scripts.ik.grasp_geometry import (
    generate_sphere_grasp_candidates,
    linear_approach_waypoints,
    vertical_approach_waypoints,
)


class GraspGeometryTests(unittest.TestCase):
    def test_vertical_waypoints_preserve_xy_and_finish_at_target(self) -> None:
        target = np.array([0.1, -0.2, 0.3])
        waypoints = vertical_approach_waypoints(
            target,
            meters_per_unit=1.0,
            hover_height_mm=70.0,
            descent_step_mm=10.0,
        )
        # Preserve the current float-derived sampling used by the live scene.
        self.assertEqual(len(waypoints), 9)
        np.testing.assert_allclose(waypoints[0], [0.1, -0.2, 0.37])
        np.testing.assert_allclose(waypoints[-1], target)
        self.assertTrue(
            all(left[2] > right[2] for left, right in zip(waypoints, waypoints[1:]))
        )

    def test_linear_waypoints_follow_the_approach_axis(self) -> None:
        target = np.array([0.03, 0.0, 0.05])
        waypoints = linear_approach_waypoints(
            target,
            approach_axis_world=[-1.0, 0.0, 0.0],
            meters_per_unit=1.0,
            hover_height_mm=20.0,
            descent_step_mm=10.0,
        )
        np.testing.assert_allclose(waypoints[0], [0.05, 0.0, 0.05])
        np.testing.assert_allclose(waypoints[-1], target)
        np.testing.assert_allclose(
            np.diff(np.asarray(waypoints), axis=0),
            [[-0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
        )

    def test_candidate_sampling_uses_elevation_and_axis_aligned_paths(self) -> None:
        candidates = generate_sphere_grasp_candidates(
            sphere_center=[0.0, 0.0, 0.05],
            sphere_radius=0.025,
            nominal_surface_point=[0.025, 0.0, 0.05],
            meters_per_unit=1.0,
            surface_clearance_m=0.005,
            surface_offsets_deg=[0.0, 90.0],
            hover_height_mm=20.0,
            descent_step_mm=10.0,
            approach_elevations_deg=[90.0, 75.0, 0.0],
        )
        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            candidates[0]["candidate_id"], "surface_+0_elevation_90"
        )
        self.assertEqual(
            candidates[-1]["candidate_id"], "surface_+90_elevation_0"
        )
        for candidate in candidates:
            self.assertAlmostEqual(
                float(np.linalg.norm(candidate["target_axis_world"])),
                1.0,
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(candidate["target_closing_axis_world"])),
                1.0,
            )
            self.assertEqual(len(candidate["waypoints_world_stage_units"]), 3)
            waypoints = np.asarray(candidate["waypoints_world_stage_units"])
            self.assertAlmostEqual(
                float(np.linalg.norm(waypoints[-1, :2])),
                0.030,
            )
            self.assertAlmostEqual(float(waypoints[-1, 2]), 0.05)
            segments = np.diff(waypoints, axis=0)
            np.testing.assert_allclose(
                segments / np.linalg.norm(segments, axis=1)[:, None],
                np.repeat(
                    np.asarray(candidate["target_axis_world"])[None, :],
                    len(segments),
                    axis=0,
                ),
                atol=1e-12,
            )

        top_down = candidates[0]
        np.testing.assert_allclose(
            top_down["target_axis_world"], [0.0, 0.0, -1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            top_down["waypoints_world_stage_units"][0], [0.03, 0.0, 0.07]
        )
        horizontal = candidates[2]
        np.testing.assert_allclose(
            horizontal["target_axis_world"], [-1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            horizontal["waypoints_world_stage_units"][0], [0.05, 0.0, 0.05]
        )

    def test_candidate_sampling_accepts_legacy_tilts_and_rejects_bad_inputs(self) -> None:
        legacy = generate_sphere_grasp_candidates(
            sphere_center=[0.0, 0.0, 0.05],
            sphere_radius=0.025,
            nominal_surface_point=[0.025, 0.0, 0.05],
            meters_per_unit=1.0,
            surface_clearance_m=0.005,
            surface_offsets_deg=[0.0],
            hover_height_mm=20.0,
            descent_step_mm=10.0,
            approach_tilts_deg=[0.0],
        )
        self.assertEqual(legacy[0]["approach_elevation_deg"], 90.0)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            generate_sphere_grasp_candidates(
                sphere_center=[0.0, 0.0, 0.05],
                sphere_radius=0.025,
                nominal_surface_point=[0.025, 0.0, 0.05],
                meters_per_unit=1.0,
                surface_clearance_m=0.005,
                surface_offsets_deg=[0.0],
                hover_height_mm=20.0,
                descent_step_mm=10.0,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 90\]"):
            generate_sphere_grasp_candidates(
                sphere_center=[0.0, 0.0, 0.05],
                sphere_radius=0.025,
                nominal_surface_point=[0.025, 0.0, 0.05],
                meters_per_unit=1.0,
                surface_clearance_m=0.005,
                surface_offsets_deg=[0.0],
                hover_height_mm=20.0,
                descent_step_mm=10.0,
                approach_elevations_deg=[91.0],
            )
        with self.assertRaisesRegex(ValueError, "nonzero length"):
            linear_approach_waypoints(
                [0.0, 0.0, 0.0],
                approach_axis_world=[0.0, 0.0, 0.0],
                meters_per_unit=1.0,
                hover_height_mm=20.0,
                descent_step_mm=10.0,
            )
if __name__ == "__main__":
    unittest.main()
