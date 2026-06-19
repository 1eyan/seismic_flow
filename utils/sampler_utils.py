"""Diverse top-k selection utilities (self-contained, NumPy only).

Extracted from reg_tool/patch_sampler.py and reg_tool/anchor_selector.py.
No internal dependencies beyond numpy.
"""

from typing import List, Optional, Sequence
import numpy as np


def parse_metric_weights(metric_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    if metric_weights is None:
        metric_weights = [1.0, 1.0, 0.5, 0.5]
    w = np.asarray(metric_weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != 4:
        raise ValueError(f"metric_weights must have length 4, got {w.shape[0]}")
    if np.any(w < 0):
        raise ValueError("metric_weights must be non-negative")
    return w


def weighted_sqdist_to_one(
    center_coord,
    all_coords,
    metric_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    center = np.asarray(center_coord, dtype=np.float32).reshape(-1)
    if center.shape[0] != 4:
        raise ValueError(f"center_coord must have shape [4], got {center.shape}")
    coords = np.asarray(all_coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"all_coords must have shape [N, 4], got {coords.shape}")
    w = parse_metric_weights(metric_weights)
    diff = coords - center[None, :]
    d2 = np.sum((diff * diff) * w[None, :], axis=1)
    return d2.astype(np.float64)


def diverse_topk(
    center_coord,
    candidate_idx,
    all_coords,
    k: int,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
) -> np.ndarray:
    coords_np = np.asarray(all_coords, dtype=np.float32)
    if coords_np.ndim != 2 or coords_np.shape[1] != 4:
        raise ValueError(f"all_coords must have shape [N, 4], got {coords_np.shape}")
    cand = np.asarray(candidate_idx, dtype=np.int64).reshape(-1)
    if cand.size == 0 or int(k) <= 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(cand < 0) or np.any(cand >= coords_np.shape[0]):
        raise ValueError("candidate_idx contains out-of-range index")

    center = np.asarray(center_coord, dtype=np.float32).reshape(-1)
    if center.shape[0] != 4:
        raise ValueError(f"center_coord must have shape [4], got {center.shape}")

    k_eff = min(int(k), int(cand.size))
    w = parse_metric_weights(metric_weights)
    cand_coords = coords_np[cand]

    center_d2 = np.sum(((cand_coords - center.reshape(1, 4)) ** 2) * w.reshape(1, 4), axis=1)
    first = int(np.argmin(center_d2))
    selected_locals: List[int] = [first]
    selected_mask = np.zeros(cand.size, dtype=bool)
    selected_mask[first] = True

    min_d2_to_selected = np.sum(
        ((cand_coords - cand_coords[first : first + 1]) ** 2) * w.reshape(1, 4),
        axis=1,
    )

    for _ in range(1, k_eff):
        scores = -center_d2 + float(beta) * min_d2_to_selected
        scores[selected_mask] = -np.inf
        next_local = int(np.argmax(scores))
        if not np.isfinite(scores[next_local]):
            break
        selected_locals.append(next_local)
        selected_mask[next_local] = True

        d2_new = np.sum(
            ((cand_coords - cand_coords[next_local : next_local + 1]) ** 2) * w.reshape(1, 4),
            axis=1,
        )
        min_d2_to_selected = np.minimum(min_d2_to_selected, d2_new)

    return cand[np.asarray(selected_locals, dtype=np.int64)]
