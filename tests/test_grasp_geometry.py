import unittest

import numpy as np

from scripts.ik.grasp_geometry import (
    generate_sphere_grasp_candidates,
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

    def test_candidate_sampling_is_parameterized_and_axes_are_unit_length(self) -> None:
        candidates = generate_sphere_grasp_candidates(
            sphere_center=[0.0, 0.0, 0.05],
            sphere_radius=0.025,
            nominal_surface_point=[0.025, 0.0, 0.05],
            meters_per_unit=1.0,
            surface_clearance_m=0.005,
            surface_offsets_deg=[0.0, 90.0],
            approach_tilts_deg=[0.0, 15.0],
            hover_height_mm=20.0,
            descent_step_mm=10.0,
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0]["candidate_id"], "surface_+0_tilt_0")
        self.assertEqual(candidates[-1]["candidate_id"], "surface_+90_tilt_15")
        for candidate in candidates:
            self.assertAlmostEqual(
                float(np.linalg.norm(candidate["target_axis_world"])),
                1.0,
            )
            self.assertEqual(len(candidate["waypoints_world_stage_units"]), 3)
            surface_point = np.asarray(
                candidate["surface_point_world_stage_units"]
            )
            approach_point = np.asarray(
                candidate["approach_point_world_stage_units"]
            )
            waypoints = np.asarray(candidate["waypoints_world_stage_units"])
            self.assertAlmostEqual(
                float(np.linalg.norm(surface_point[:2])),
                0.025,
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(approach_point[:2])),
                0.030,
            )
            np.testing.assert_allclose(waypoints[-1], approach_point)
            np.testing.assert_allclose(
                np.linalg.norm(waypoints[:, :2], axis=1),
                0.030,
            )
if __name__ == "__main__":
    unittest.main()
