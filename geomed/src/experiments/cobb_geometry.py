"""Detector-independent Auto-EV selection from predicted vertebral corners."""
from __future__ import annotations
import numpy as np

LANDMARK_COUNT = 68
VERTEBRA_COUNT = 17
POINTS_PER_VERTEBRA = 4
DEGENERATE_DIRECTION_EPSILON_PX = 1e-6

def rank_primary_cobb_candidates(
    points: np.ndarray,
    image_size: tuple[int, int],
    max_candidates: int | None = None,
) -> list[dict]:
    """Return the existing selector's candidates in deterministic score order.

    Candidate generation and scoring are identical to ``select_primary_cobb``:
    degenerate vertebral directions are excluded and the score is the analytic
    angle between predicted vertebral directions.  A stable descending sort
    preserves the old ``np.argmax`` tie break used by the winner.
    """
    del image_size  # Kept in the public interface for selector compatibility.
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (LANDMARK_COUNT, 2) or not np.isfinite(values).all():
        raise ValueError("AASCE prediction must contain 68 finite 2D points")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive or None")

    vertebrae = values.reshape(VERTEBRA_COUNT, POINTS_PER_VERTEBRA, 2)
    directions = 0.5 * (
        (vertebrae[:, 1] - vertebrae[:, 0])
        + (vertebrae[:, 3] - vertebrae[:, 2])
    )
    norms = np.linalg.norm(directions, axis=1)
    valid_indices = np.flatnonzero(norms > DEGENERATE_DIRECTION_EPSILON_PX)
    invalid_mask = norms <= DEGENERATE_DIRECTION_EPSILON_PX
    fallback = len(valid_indices) < 2
    if fallback:
        return [{
            "rank": 1,
            "upper_index": 0,
            "lower_index": 1,
            "score": 0.0,
            "analytic_cobb": 0.0,
            "candidate_points": np.vstack([vertebrae[0], vertebrae[1]]).astype(np.float32),
            "fallback": True,
            "degenerate_vertebra_mask": invalid_mask.copy(),
        }]

    valid_directions = directions[valid_indices]
    valid_norms = norms[valid_indices]
    cosine = np.clip(
        valid_directions @ valid_directions.T
        / (valid_norms[:, None] * valid_norms[None, :]),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(cosine))
    upper, lower = np.triu_indices(len(valid_indices), 1)
    scores = angles[upper, lower]
    order = np.argsort(-scores, kind="stable")
    if max_candidates is not None:
        order = order[:max_candidates]

    candidates = []
    for rank, selected in enumerate(order, start=1):
        upper_index = int(valid_indices[upper[selected]])
        lower_index = int(valid_indices[lower[selected]])
        angle = float(scores[selected])
        candidates.append({
            "rank": rank,
            "upper_index": upper_index,
            "lower_index": lower_index,
            "score": angle,
            "analytic_cobb": angle,
            "candidate_points": np.vstack([
                vertebrae[upper_index], vertebrae[lower_index]
            ]).astype(np.float32),
            "fallback": False,
            "degenerate_vertebra_mask": invalid_mask.copy(),
        })
    return candidates

def select_primary_cobb(
    points: np.ndarray,
    image_size: tuple[int, int],
    return_diagnostics: bool = False,
):
    """Select the primary EV pair without dropping degenerate predictions.

    Zero-length predicted vertebral directions are excluded from candidate
    pairs.  If fewer than two valid directions remain, the sample is retained
    with a deterministic 0-degree fallback and pair (0, 1).  Released GT has
    no degenerate directions, so this agrees exactly with ``aasce_primary_cobb``
    on valid inputs.
    """
    candidates = rank_primary_cobb_candidates(points, image_size, max_candidates=1)
    winner = candidates[0]
    fallback = bool(winner["fallback"])
    invalid_mask = np.asarray(winner["degenerate_vertebra_mask"], dtype=bool)
    output = (
        float(winner["analytic_cobb"]),
        int(winner["upper_index"]),
        int(winner["lower_index"]),
        np.asarray(winner["candidate_points"], dtype=np.float32),
    )
    if not return_diagnostics:
        return output
    diagnostics = {
        "valid": not fallback,
        "fallback": fallback,
        "degenerate_vertebra_count": int(np.count_nonzero(invalid_mask)),
        "degenerate_vertebra_mask": invalid_mask,
    }
    return (*output, diagnostics)
