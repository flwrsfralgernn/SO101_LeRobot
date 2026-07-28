"""Pure endpoint-first candidate ordering and status helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


Candidate = dict[str, Any]


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
