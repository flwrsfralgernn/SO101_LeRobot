import unittest

from scripts.ik.candidate_planning import (
    evaluate_first_successful_elevation,
    evaluate_first_valid_path,
    ranked_endpoint_candidates,
    validate_approach_elevation_schedule,
)


def endpoint(candidate_id: str, **overrides: float) -> dict[str, object]:
    candidate: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_index": 0,
        "status": "endpoint_valid",
        "endpoint_closing_axis_error_rad": 0.1,
        "approach_tilt_deg": 15.0,
        "surface_offset_deg": 30.0,
        "endpoint_joint_travel_deg": 20.0,
        "endpoint_position_error_m": 0.001,
        "endpoint_axis_error_rad": 0.05,
    }
    candidate.update(overrides)
    return candidate


class EndpointPlanningTests(unittest.TestCase):
    def test_endpoint_ranking_is_deterministic_and_uses_declared_priority(self) -> None:
        candidates = [
            endpoint("travel", endpoint_joint_travel_deg=5.0, candidate_index=2),
            endpoint("jaw", endpoint_closing_axis_error_rad=0.01, candidate_index=1),
            endpoint("tilt", approach_tilt_deg=0.0, candidate_index=0),
        ]
        self.assertEqual(
            [item["candidate_id"] for item in ranked_endpoint_candidates(candidates)],
            ["jaw", "tilt", "travel"],
        )

    def test_all_endpoint_rejections_evaluate_no_paths(self) -> None:
        candidates = [
            {"candidate_id": "a", "status": "rejected"},
            {"candidate_id": "b", "status": "rejected"},
        ]
        selected, evaluated, skipped = evaluate_first_valid_path(
            candidates,
            lambda candidate: self.fail("path evaluator should not run"),
        )
        self.assertIsNone(selected)
        self.assertEqual(evaluated, 0)
        self.assertEqual(skipped, 0)

    def test_first_path_failure_then_success_skips_remaining_candidates(self) -> None:
        candidates = [
            endpoint("first", candidate_index=0),
            endpoint("second", candidate_index=1),
            endpoint("third", candidate_index=2),
        ]
        calls: list[str] = []

        def evaluate(candidate: dict[str, object]) -> dict[str, object]:
            candidate_id = str(candidate["candidate_id"])
            calls.append(candidate_id)
            result = dict(candidate)
            if candidate_id == "first":
                result.update(
                    {
                        "status": "rejected",
                        "rejection_stage": "path",
                        "reason": "path failed",
                    }
                )
            else:
                result["status"] = "valid"
            return result

        selected, evaluated, skipped = evaluate_first_valid_path(
            candidates, evaluate
        )
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(selected["candidate_id"], "second")
        self.assertEqual(evaluated, 2)
        self.assertEqual(skipped, 1)
        self.assertEqual(candidates[2]["status"], "skipped")
        self.assertEqual(
            candidates[2]["selected_candidate_id"], "second"
        )


class ElevationFallbackTests(unittest.TestCase):
    def test_fallback_attempts_elevations_in_order_and_stops_on_success(self) -> None:
        calls: list[float] = []
        batches = {
            90.0: {
                "selected_candidate_id": None,
                "candidates": [{"candidate_id": "top", "status": "rejected"}],
            },
            75.0: {
                "selected_candidate_id": "angled",
                "candidates": [{"candidate_id": "angled", "status": "valid"}],
            },
        }

        def evaluate_batch(elevation_deg: float) -> dict[str, object]:
            calls.append(elevation_deg)
            return batches[elevation_deg]

        result = evaluate_first_successful_elevation(
            [90.0, 75.0, 60.0],
            evaluate_batch,
        )

        self.assertEqual(calls, [90.0, 75.0])
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["attempted_elevations_deg"], [90.0, 75.0])
        self.assertEqual(result["selected_approach_elevation_deg"], 75.0)
        self.assertEqual(result["selected_candidate_id"], "angled")
        self.assertIs(result["selected_batch"], batches[75.0])
        self.assertIs(result["attempts"][0]["batch"], batches[90.0])

    def test_fallback_retains_all_attempts_when_elevations_are_exhausted(self) -> None:
        calls: list[float] = []

        def evaluate_batch(elevation_deg: float) -> dict[str, object]:
            calls.append(elevation_deg)
            return {
                "selected_candidate_id": None,
                "candidates": [
                    {
                        "candidate_id": f"candidate_{elevation_deg:.0f}",
                        "status": "rejected",
                        "reason": "joint limit",
                    }
                ],
                "planning_metrics": {"full_paths_evaluated": 0},
            }

        result = evaluate_first_successful_elevation(
            [90.0, 75.0, 0.0],
            evaluate_batch,
        )

        self.assertEqual(calls, [90.0, 75.0, 0.0])
        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["attempted_elevations_deg"], [90.0, 75.0, 0.0])
        self.assertIsNone(result["selected_approach_elevation_deg"])
        self.assertIsNone(result["selected_candidate_id"])
        self.assertIsNone(result["selected_batch"])
        self.assertEqual(
            result["attempts"][2]["batch"]["candidates"][0]["reason"],
            "joint limit",
        )

    def test_elevation_schedule_requires_descending_finite_values_in_range(self) -> None:
        self.assertEqual(
            validate_approach_elevation_schedule([90.0, 75.0, 0.0]),
            [90.0, 75.0, 0.0],
        )
        for invalid_schedule, message in (
            ([], "must not be empty"),
            ([90.0, 90.0], "strictly descending"),
            ([75.0, 90.0], "strictly descending"),
            ([91.0], r"\[0, 90\]"),
        ):
            with self.subTest(schedule=invalid_schedule):
                with self.assertRaisesRegex(ValueError, message):
                    validate_approach_elevation_schedule(invalid_schedule)

    def test_fallback_rejects_malformed_batch_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing selected_candidate_id"):
            evaluate_first_successful_elevation([90.0], lambda _: {})
        with self.assertRaisesRegex(TypeError, "must be a string or None"):
            evaluate_first_successful_elevation(
                [90.0],
                lambda _: {"selected_candidate_id": 42},
            )


if __name__ == "__main__":
    unittest.main()
