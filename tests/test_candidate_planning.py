import unittest

from scripts.ik.candidate_planning import (
    evaluate_first_valid_path,
    ranked_endpoint_candidates,
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


if __name__ == "__main__":
    unittest.main()
