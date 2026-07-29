"""Pure endpoint-first candidate ordering and status helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from typing import Any


Candidate = dict[str, Any]
CandidateBatch = dict[str, Any]


def validate_approach_elevation_schedule(
    elevations_deg: Iterable[float],
) -> list[float]:
    """Return a non-empty, strictly descending elevation schedule.

    Elevation is measured above horizontal: 90 degrees is top-down and
    0 degrees is horizontal. A strict order makes the fallback decision
    deterministic and prevents silently retrying the same approach angle.
    """
    schedule = [float(elevation) for elevation in elevations_deg]
    if not schedule:
        raise ValueError("approach elevation schedule must not be empty")
    if any(
        not math.isfinite(elevation) or not 0.0 <= elevation <= 90.0
        for elevation in schedule
    ):
        raise ValueError(
            "approach elevations must be finite values in [0, 90]"
        )
    if any(
        current <= following
        for current, following in zip(schedule, schedule[1:])
    ):
        raise ValueError(
            "approach elevation schedule must be strictly descending"
        )
    return schedule


def evaluate_first_successful_elevation(
    elevations_deg: Iterable[float],
    evaluate_batch: Callable[[float], CandidateBatch],
) -> dict[str, Any]:
    """Evaluate batches in elevation order until one selects a candidate.

    ``evaluate_batch`` must return a candidate-batch result containing
    ``selected_candidate_id`` with either a string ID or ``None``. Raw batch
    results are retained for all attempts so callers can report candidate
    diagnostics after a successful fallback or total exhaustion.
    """
    schedule = validate_approach_elevation_schedule(elevations_deg)
    attempts: list[dict[str, Any]] = []
    for elevation_deg in schedule:
        batch = evaluate_batch(elevation_deg)
        if not isinstance(batch, dict):
            raise TypeError("approach elevation batch result must be a mapping")
        if "selected_candidate_id" not in batch:
            raise ValueError(
                "approach elevation batch result is missing selected_candidate_id"
            )
        selected_candidate_id = batch["selected_candidate_id"]
        if selected_candidate_id is not None and not isinstance(
            selected_candidate_id,
            str,
        ):
            raise TypeError(
                "selected_candidate_id must be a string or None"
            )
        attempt = {
            "approach_elevation_deg": elevation_deg,
            "approach_tilt_deg": 90.0 - elevation_deg,
            "batch": batch,
        }
        attempts.append(attempt)
        if selected_candidate_id is not None:
            return {
                "status": "selected",
                "configured_elevations_deg": schedule,
                "attempted_elevations_deg": [
                    attempt["approach_elevation_deg"] for attempt in attempts
                ],
                "attempts": attempts,
                "selected_approach_elevation_deg": elevation_deg,
                "selected_candidate_id": selected_candidate_id,
                "selected_batch": batch,
            }

    return {
        "status": "exhausted",
        "configured_elevations_deg": schedule,
        "attempted_elevations_deg": [
            attempt["approach_elevation_deg"] for attempt in attempts
        ],
        "attempts": attempts,
        "selected_approach_elevation_deg": None,
        "selected_candidate_id": None,
        "selected_batch": None,
    }


def endpoint_score(candidate: Candidate) -> tuple[float, ...]:
    """Return the deterministic priority used by fast first-valid planning."""
    return (
        float(candidate["endpoint_closing_axis_error_rad"]),
        float(candidate["approach_tilt_deg"]),
        abs(float(candidate["surface_offset_deg"])),
        float(candidate["endpoint_joint_travel_deg"]),
        float(candidate["endpoint_position_error_m"]),
        float(candidate["endpoint_axis_error_rad"]),
        float(candidate["candidate_index"]),
    )


def ranked_endpoint_candidates(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    """Return endpoint-feasible candidates in deterministic priority order."""
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.get("status") == "endpoint_valid"
        ),
        key=endpoint_score,
    )


def evaluate_first_valid_path(
    candidates: list[Candidate],
    evaluate_path: Callable[[Candidate], Candidate],
) -> tuple[Candidate | None, int, int]:
    """Evaluate ranked endpoints until one full path succeeds.

    Returns ``(selected, evaluated_count, skipped_count)`` and mutates the
    candidate records so diagnostics cover rejected and skipped branches.
    """
    ranked = ranked_endpoint_candidates(candidates)
    selected: Candidate | None = None
    evaluated_count = 0
    for rank, candidate in enumerate(ranked, start=1):
        candidate["endpoint_rank"] = rank
        if selected is not None:
            candidate.update(
                {
                    "status": "skipped",
                    "skip_reason": "first_valid_path_selected",
                    "selected_candidate_id": selected.get("candidate_id"),
                }
            )
            continue
        evaluated_count += 1
        result = evaluate_path(candidate)
        candidate.clear()
        candidate.update(result)
        candidate["endpoint_rank"] = rank
        if candidate.get("status") == "valid":
            selected = candidate

    skipped_count = sum(
        candidate.get("status") == "skipped" for candidate in candidates
    )
    return selected, evaluated_count, skipped_count
