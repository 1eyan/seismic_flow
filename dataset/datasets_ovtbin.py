"""OVT SSL patch dataset: raw irregular context → regular grid target.

Training: raw irregular context → observed  grid target (trace_mask=1)
Inference: raw irregular context → missing   grid target (trace_mask=0)

All KNN queries run in normalized weighted 4D space.
Patch slots are unified-sorted by (cmp_line, cmp) for spatial coherence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_h5_group(h5_path: str) -> Dict[str, Any]:
    import h5py
    h5: Dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        for key in f:
            node = f[key]
            if hasattr(node, "keys") and "data" in node:
                for k in node.keys():
                    h5[k] = node[k][:]
                break
    return h5


def _ensure_midpoint(h5_data: Dict[str, Any]) -> None:
    if "midpoint_x" not in h5_data:
        h5_data["midpoint_x"] = (h5_data["sx"] + h5_data["rx"]) / 2.0
    if "midpoint_y" not in h5_data:
        h5_data["midpoint_y"] = (h5_data["sy"] + h5_data["ry"]) / 2.0


def _crop_or_pad_time(traces: np.ndarray, time_ps: int) -> np.ndarray:
    """Left-align: crop excess from left, pad zeros to left."""
    n_t, n_s = traces.shape
    if n_s > time_ps:
        return traces[:, n_s - time_ps:]
    if n_s < time_ps:
        pad = np.zeros((n_t, time_ps - n_s), dtype=traces.dtype)
        return np.concatenate([pad, traces], axis=1)
    return traces


# ---------------------------------------------------------------------------
# DatasetH5_ovtbin
# ---------------------------------------------------------------------------

class DatasetH5_ovtbin:
    """OVT self-supervised patch dataset.

    Parameters
    ----------
    h5_raw : str
        Path to raw irregular H5 (context pool).
    h5_grid : str
        Path to test_aligned.h5 (target grid with trace_mask).
    h5_regular : str
        Path to label_ovt.h5 (coordinate statistics only).
    train : bool
        True → targets from trace_mask==1; False → targets from trace_mask==0.
    time_ps : int
        Time samples per trace.
    trace_ps : int
        Total trace slots per patch.
    ovt_target_slots : int
        Number of target slots per patch (rest are context).
    kdtree_offset_weight : float
        Multiplier for offset dimensions in 4D KNN space.
    profile :
        SEG-Y profile (for key_columns, not primary in this dataset).
    """

    def __init__(
        self,
        h5_raw: str,
        h5_grid: str,
        h5_regular: str,
        train: bool = True,
        time_ps: int = 1256,
        trace_ps: int = 128,
        ovt_target_slots: int = 32,
        kdtree_offset_weight: float = 2.0,
        profile=None,
        seed: int = 0,
        full_coverage: bool = False,
    ):
        self.train = train
        self.rng = np.random.RandomState(seed)
        self.time_ps = int(time_ps)
        self.trace_ps = int(trace_ps)
        self.target_slots = int(ovt_target_slots)
        self.context_slots = self.trace_ps - self.target_slots
        self.kdtree_offset_weight = float(kdtree_offset_weight)
        self.profile = profile
        self.dt_ms = 4
        self.t0_ms = 0
        self.patch_amp_percentile = 99.5

        # ------------------------------------------------------------------
        # 1. Load H5 data
        # ------------------------------------------------------------------
        print(f"DatasetH5_ovtbin (train={train}) loading...")

        self.h5_raw = _load_h5_group(h5_raw)
        _ensure_midpoint(self.h5_raw)
        n_raw = self.h5_raw["data"].shape[0]
        print(f"  raw context H5: {n_raw} traces")

        self.h5_grid = _load_h5_group(h5_grid)
        _ensure_midpoint(self.h5_grid)
        n_grid = self.h5_grid["data"].shape[0]
        trace_mask = self.h5_grid["trace_mask"].ravel().astype(np.float32)
        self._grid_observed = np.flatnonzero(trace_mask >= 0.5)
        self._grid_missing = np.flatnonzero(trace_mask < 0.5)
        print(f"  grid H5: {n_grid} cells, "
              f"observed={len(self._grid_observed)}, missing={len(self._grid_missing)}")

        self.h5_regular = _load_h5_group(h5_regular)
        _ensure_midpoint(self.h5_regular)
        print(f"  regular H5: {self.h5_regular['data'].shape[0]} traces (stats only)")

        # ------------------------------------------------------------------
        # 2. Cell key: raw → cell_key mapping (for leak prevention)
        # ------------------------------------------------------------------
        self._raw_cell_key = self._build_cell_keys(self.h5_raw)
        self._raw_cell_to_indices: Dict[Tuple, np.ndarray] = {}
        for i, ck in enumerate(self._raw_cell_key):
            ck_tuple = tuple(int(v) for v in ck)
            self._raw_cell_to_indices.setdefault(ck_tuple, []).append(i)
        for ck in self._raw_cell_to_indices:
            self._raw_cell_to_indices[ck] = np.array(
                self._raw_cell_to_indices[ck], dtype=np.int64
            )
        self._grid_cell_key = self._build_cell_keys(self.h5_grid)

        # ------------------------------------------------------------------
        # 3. Coordinate statistics (from regular H5)
        # ------------------------------------------------------------------
        self.coord_stats = self._compute_coord_stats()
        print(f"  coord stats: rx=[{self.coord_stats['rx_min']:.0f}, {self.coord_stats['rx_max']:.0f}], "
              f"ry=[{self.coord_stats['ry_min']:.0f}, {self.coord_stats['ry_max']:.0f}], "
              f"sx=[{self.coord_stats['sx_min']:.0f}, {self.coord_stats['sx_max']:.0f}], "
              f"sy=[{self.coord_stats['sy_min']:.0f}, {self.coord_stats['sy_max']:.0f}]")

        # ------------------------------------------------------------------
        # 4. OVT groups
        # ------------------------------------------------------------------
        self.raw_ovt_groups = self._build_ovt_groups(self.h5_raw)
        grid_ovt_groups = self._build_ovt_groups(self.h5_grid)
        self.grid_observed_by_ovt = self._filter_ovt_groups(
            grid_ovt_groups, self._grid_observed
        )
        self.grid_missing_by_ovt = self._filter_ovt_groups(
            grid_ovt_groups, self._grid_missing
        )

        # OVT weight for training sampling
        self.ovt_ids = sorted(self.grid_observed_by_ovt.keys())
        counts = np.array([len(self.grid_observed_by_ovt[o]) for o in self.ovt_ids],
                          dtype=np.float64)
        self.ovt_weights = np.sqrt(counts)
        self.ovt_weights /= self.ovt_weights.sum()

        # ------------------------------------------------------------------
        # 5. Build normalized weighted KDTrees on raw pool
        # ------------------------------------------------------------------
        self._raw_norm_coords = self._normalize_raw_coords()
        # weighted 4D vectors for KNN
        w = self.kdtree_offset_weight
        self._raw_weighted = np.column_stack([
            self._raw_norm_coords["mx_n"],
            self._raw_norm_coords["my_n"],
            w * self._raw_norm_coords["ox_n"],
            w * self._raw_norm_coords["oy_n"],
        ]).astype(np.float64)

        # Per-OVT KDTrees
        self._per_ovt_kdtrees: Dict[int, object] = {}
        for ovt_id, indices in self.raw_ovt_groups.items():
            try:
                from scipy.spatial import KDTree
                self._per_ovt_kdtrees[ovt_id] = KDTree(self._raw_weighted[indices])
            except ImportError:
                pass  # fallback to brute-force later

        # Global 4D KDTree
        self._global_4d_kdtree = None
        try:
            from scipy.spatial import KDTree
            self._global_4d_kdtree = KDTree(self._raw_weighted)
            print(f"  global 4D KDTree: {n_raw} points")
        except ImportError:
            print("  scipy not available; using numpy top-k fallback")

        self.full_coverage = full_coverage and not train

        # ------------------------------------------------------------------
        # 6. Target pool
        # ------------------------------------------------------------------
        if self.train:
            self._target_pool = self._grid_observed.copy()
        else:
            self._target_pool = self._grid_missing.copy()

        # ------------------------------------------------------------------
        # 7. Full-coverage plan (deterministic, all grid cells, no random sampling)
        # ------------------------------------------------------------------
        self._coverage_plan = []  # list of (ovt_id, target_grid_indices)
        if self.full_coverage:
            # Merge observed + missing per OVT — covers EVERY grid cell exactly once
            self._grid_all_by_ovt = {}
            all_ovt_ids = set(self.grid_observed_by_ovt.keys()) | set(self.grid_missing_by_ovt.keys())
            n_all_cells = 0
            for ovt_id in sorted(all_ovt_ids):
                obs = self.grid_observed_by_ovt.get(ovt_id, np.array([], dtype=np.int64))
                mis = self.grid_missing_by_ovt.get(ovt_id, np.array([], dtype=np.int64))
                merged = np.concatenate([obs, mis])
                self._grid_all_by_ovt[ovt_id] = merged
                n_all_cells += len(merged)

            # Build coverage plan: chunk each OVT's cells into batches of target_slots.
            # Last chunk per OVT may be smaller — no cross-OVT padding (方案A).
            for ovt_id in sorted(self._grid_all_by_ovt.keys()):
                pool = self._grid_all_by_ovt[ovt_id]
                for start in range(0, len(pool), self.target_slots):
                    chunk = pool[start:start + self.target_slots]
                    self._coverage_plan.append((int(ovt_id), chunk.copy()))

            self._epoch_len = len(self._coverage_plan)
            n_label = self.h5_regular["data"].shape[0]
            print(f"  [FULL_COVERAGE] grid cells: total={n_all_cells} "
                  f"observed={len(self._grid_observed)} missing={len(self._grid_missing)}")
            print(f"  [FULL_COVERAGE] coverage plan: batches={self._epoch_len} "
                  f"targets={n_all_cells}")
            print(f"  [FULL_COVERAGE] label H5 traces: {n_label}")
            if n_all_cells != n_label:
                raise ValueError(
                    f"[FULL_COVERAGE] grid cell count ({n_all_cells}) != "
                    f"label H5 trace count ({n_label}). Grid and label H5 mismatch!"
                )
            print(f"  [FULL_COVERAGE] grid cell count matches label H5 trace count: OK")
        else:
            # Each __getitem__ randomly samples target_slots from the pool.
            # Epoch length = pool_size / target_slots → each target seen ~1× per epoch.
            self._epoch_len = max(1, len(self._target_pool) // self.target_slots)

        print(f"  target pool: {len(self._target_pool)}, "
              f"epoch_len: {self._epoch_len} "
              f"({'full_coverage' if self.full_coverage else 'random sampling'})")

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_key_values(
        self, h5: Dict[str, Any], indices: np.ndarray
    ) -> np.ndarray:
        """Build SEG-Y geometry key matrix (N, n_keys) for SEGY write-back."""
        cols = []
        for key in self.profile.key_columns:
            fallback = self.profile.h5_fallback.get(key) if hasattr(self.profile, 'h5_fallback') else None
            col = h5[key].ravel()[indices].astype(np.int64)
            cols.append(col)
        return np.stack(cols, axis=1)

    @staticmethod
    def _build_cell_keys(h5: Dict[str, Any]) -> np.ndarray:
        """Return (N, 4) int64 array of (cmp_line, cmp, offset_x_bin, offset_y_bin)."""
        keys = np.column_stack([
            h5["cmp_line"].ravel().astype(np.int64),
            h5["cmp"].ravel().astype(np.int64),
            h5["offset_x_bin"].ravel().astype(np.int64),
            h5["offset_y_bin"].ravel().astype(np.int64),
        ])
        return keys

    def _compute_coord_stats(self) -> Dict[str, float]:
        """Compute min/max from regular H5 for midpoint and offset.

        midpoint_x/y → rx/ry slots.  offset_x/y (raw!) → sx/sy slots.
        Uses the regular H5's offset_x / offset_y (not offset_x_center) so that
        raw traces' real offsets share the same normalisation range.
        """
        rh5 = self.h5_regular
        # midpoint from regular H5
        mx = rh5["midpoint_x"].ravel()
        my = rh5["midpoint_y"].ravel()
        # offset — use real offset_x / offset_y if present, else fallback to offset_x_center
        if "offset_x" in rh5:
            ox = rh5["offset_x"].ravel()  # not offset_x_center! raw offset
        else:
            ox = rh5["offset_x_center"].ravel()
        if "offset_y" in rh5:
            oy = rh5["offset_y"].ravel()
        else:
            oy = rh5["offset_y_center"].ravel()

        # Percentile clip (same as DatasetH5_interp)
        mx_c = np.clip(mx, np.percentile(mx, 0.01), np.percentile(mx, 99.99))
        my_c = np.clip(my, np.percentile(my, 0.01), np.percentile(my, 99.99))
        ox_c = np.clip(ox, np.percentile(ox, 0.01), np.percentile(ox, 99.99))
        oy_c = np.clip(oy, np.percentile(oy, 0.01), np.percentile(oy, 99.99))

        # Add a small margin (expand by 0.1%) so raw traces slightly outside
        # the regular range don't all land at exactly 0 or 1.
        margin = 0.001
        def _expand(lo, hi):
            span = hi - lo
            return lo - span * margin, hi + span * margin

        rx_min, rx_max = _expand(mx_c.min(), mx_c.max())
        ry_min, ry_max = _expand(my_c.min(), my_c.max())
        sx_min, sx_max = _expand(ox_c.min(), ox_c.max())
        sy_min, sy_max = _expand(oy_c.min(), oy_c.max())

        return {
            "rx_min": float(rx_min), "rx_max": float(rx_max),
            "ry_min": float(ry_min), "ry_max": float(ry_max),
            "sx_min": float(sx_min), "sx_max": float(sx_max),
            "sy_min": float(sy_min), "sy_max": float(sy_max),
        }

    def _normalize_raw_coords(self) -> Dict[str, np.ndarray]:
        """Normalize all raw traces to [0,1] with clip.

        Uses real offset_x / offset_y (not offset_x_center) for raw traces.
        """
        s = self.coord_stats
        mx = self.h5_raw["midpoint_x"].ravel().astype(np.float64)
        my = self.h5_raw["midpoint_y"].ravel().astype(np.float64)
        ox = self.h5_raw["offset_x"].ravel().astype(np.float64)
        oy = self.h5_raw["offset_y"].ravel().astype(np.float64)

        mx_n = np.clip((mx - s["rx_min"]) / (s["rx_max"] - s["rx_min"] + 1e-12), 0.0, 1.0)
        my_n = np.clip((my - s["ry_min"]) / (s["ry_max"] - s["ry_min"] + 1e-12), 0.0, 1.0)
        ox_n = np.clip((ox - s["sx_min"]) / (s["sx_max"] - s["sx_min"] + 1e-12), 0.0, 1.0)
        oy_n = np.clip((oy - s["sy_min"]) / (s["sy_max"] - s["sy_min"] + 1e-12), 0.0, 1.0)

        return {"mx_n": mx_n, "my_n": my_n, "ox_n": ox_n, "oy_n": oy_n}

    def _normalize_grid_coords(self, indices: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Normalize grid cell coords to [0,1].

        Grid uses offset_x_center / offset_y_center.
        """
        s = self.coord_stats
        mx = self.h5_grid["midpoint_x"].ravel()[indices].astype(np.float64)
        my = self.h5_grid["midpoint_y"].ravel()[indices].astype(np.float64)
        ox = self.h5_grid["offset_x_center"].ravel()[indices].astype(np.float64)
        oy = self.h5_grid["offset_y_center"].ravel()[indices].astype(np.float64)

        mx_n = np.clip((mx - s["rx_min"]) / (s["rx_max"] - s["rx_min"] + 1e-12), 0.0, 1.0)
        my_n = np.clip((my - s["ry_min"]) / (s["ry_max"] - s["ry_min"] + 1e-12), 0.0, 1.0)
        ox_n = np.clip((ox - s["sx_min"]) / (s["sx_max"] - s["sx_min"] + 1e-12), 0.0, 1.0)
        oy_n = np.clip((oy - s["sy_min"]) / (s["sy_max"] - s["sy_min"] + 1e-12), 0.0, 1.0)

        return mx_n, my_n, ox_n, oy_n

    @staticmethod
    def _build_ovt_groups(h5: Dict[str, Any]) -> Dict[int, np.ndarray]:
        ovt = h5["ovt"].ravel().astype(np.int32)
        groups: Dict[int, List[int]] = {}
        for i, oid in enumerate(ovt):
            groups.setdefault(int(oid), []).append(i)
        return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}

    @staticmethod
    def _filter_ovt_groups(
        groups: Dict[int, np.ndarray], valid_set: np.ndarray
    ) -> Dict[int, np.ndarray]:
        valid = set(valid_set.tolist() if valid_set.dtype != np.int64
                     else [int(x) for x in valid_set])
        result = {}
        for oid, indices in groups.items():
            filtered = np.array([i for i in indices if int(i) in valid], dtype=np.int64)
            if len(filtered) > 0:
                result[oid] = filtered
        return result

    # ==================================================================
    # Adaptive KNN — target-weighted multi-query
    # ==================================================================

    def _query_context(
        self,
        target_grid_indices: np.ndarray,
        forbidden_cell_keys: set,
        target_ovt_id: int,
    ) -> np.ndarray:
        """Adaptive KNN: multi-query each target, deduplicate, return top context_slots.

        Uses target-weighted multi-query in normalized weighted 4D space.
        Priority: same-OVT raw traces first, then global 4D.
        Applies leak prevention by excluding raw traces whose cell key is forbidden.

        Returns raw trace indices (from h5_raw).
        """
        w = self.kdtree_offset_weight
        t_mx_n, t_my_n, t_ox_n, t_oy_n = self._normalize_grid_coords(target_grid_indices)
        t_weighted = np.column_stack([
            t_mx_n, t_my_n, w * t_ox_n, w * t_oy_n
        ]).astype(np.float64)

        k_per_target = 8
        max_iter = 10
        raw_pool_size = self.h5_raw["data"].shape[0]

        # Build forbidden set of raw trace indices
        forbidden_raw: set = set()
        if forbidden_cell_keys:
            for ck in forbidden_cell_keys:
                idxs = self._raw_cell_to_indices.get(ck)
                if idxs is not None:
                    forbidden_raw.update(int(i) for i in idxs)

        seen: set = set()
        candidates: List[Tuple[int, float]] = []  # (raw_idx, min_distance)

        # ================================================================
        # Phase 1: Same-OVT exclusive — fill from per-OVT KDTree first
        # ================================================================
        kdt = self._per_ovt_kdtrees.get(target_ovt_id)
        if kdt is not None:
            ovt_indices = self.raw_ovt_groups[target_ovt_id]
            ovt_pool_size = len(ovt_indices)
            k_ovt = 8
            for _iter in range(max_iter):
                for tq in t_weighted:
                    k_q = min(k_ovt, ovt_pool_size)
                    dists, nn = kdt.query(tq, k=k_q)
                    nn = np.atleast_1d(nn)
                    dists = np.atleast_1d(dists)
                    for d, local_idx in zip(dists, nn):
                        raw_idx = int(ovt_indices[local_idx])
                        if raw_idx not in seen and raw_idx not in forbidden_raw:
                            seen.add(raw_idx)
                            candidates.append((raw_idx, float(d)))
                if len(candidates) >= self.context_slots:
                    break
                k_ovt *= 2
                if k_ovt > ovt_pool_size:
                    break

        # ================================================================
        # Phase 2: Global fallback — fill remaining slots from 4D KDTree
        # ================================================================
        n_needed = self.context_slots - len(candidates)
        if n_needed > 0 and self._global_4d_kdtree is not None:
            k_global = 8
            for _iter in range(max_iter):
                for tq in t_weighted:
                    k_q = min(k_global, raw_pool_size)
                    dists, nn = self._global_4d_kdtree.query(tq, k=k_q)
                    nn = np.atleast_1d(nn)
                    dists = np.atleast_1d(dists)
                    for d, raw_idx in zip(dists, nn):
                        raw_idx = int(raw_idx)
                        if raw_idx not in seen and raw_idx not in forbidden_raw:
                            seen.add(raw_idx)
                            candidates.append((raw_idx, float(d)))
                if len(candidates) >= self.context_slots:
                    break
                k_global *= 2
                if k_global > raw_pool_size:
                    break

        # Sort by distance, take top context_slots
        candidates.sort(key=lambda x: x[1])
        return np.array([c[0] for c in candidates[:self.context_slots]], dtype=np.int64)

    # ==================================================================
    # Patch construction
    # ==================================================================

    def _unified_sort_order(self, cmp_lines: np.ndarray, cmps: np.ndarray) -> np.ndarray:
        """Return sort indices: primary=cmp_line(asc), secondary=cmp(asc).

        Within a single OVT, spatial adjacency is the meaningful ordering.
        """
        return np.lexsort((cmps, cmp_lines))

    def _build_patch(
        self,
        target_indices: np.ndarray,
        context_indices: np.ndarray,
        target_data: np.ndarray,
        context_data: np.ndarray,
        ovt_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Assemble patch dict from target and context arrays.

        target_indices / context_indices: grid cell / raw trace indices.
        target_data / context_data: (n_t, time_ps) data arrays.
        """
        n_target = target_data.shape[0]
        n_context = context_data.shape[0]

        # Total slots
        total = n_target + n_context
        data_patch = np.zeros((self.trace_ps, self.time_ps), dtype=np.float32)
        masked_patch = np.zeros((self.trace_ps, self.time_ps), dtype=np.float32)
        trace_mask = np.zeros(self.trace_ps, dtype=np.float32)
        target_slot_mask = np.zeros(self.trace_ps, dtype=np.float32)
        context_slot_mask = np.zeros(self.trace_ps, dtype=np.float32)
        valid_trace_mask = np.zeros(self.trace_ps, dtype=np.float32)
        rx_patch = np.zeros(self.trace_ps, dtype=np.float32)
        ry_patch = np.zeros(self.trace_ps, dtype=np.float32)
        sx_patch = np.zeros(self.trace_ps, dtype=np.float32)
        sy_patch = np.zeros(self.trace_ps, dtype=np.float32)
        cell_key_values = np.zeros((self.trace_ps, 4), dtype=np.int64)
        key_values = np.zeros((self.trace_ps, 4), dtype=np.int64)  # SEG-Y geometry keys
        trace_indices_all = np.zeros(self.trace_ps, dtype=np.int64)

        # ---- Spatial sort: cmp_line (primary), cmp (secondary) ----
        t_cl = self.h5_grid["cmp_line"].ravel()[target_indices].astype(np.int64)
        t_cmp = self.h5_grid["cmp"].ravel()[target_indices].astype(np.int64)
        c_cl = self.h5_raw["cmp_line"].ravel()[context_indices].astype(np.int64)
        c_cmp = self.h5_raw["cmp"].ravel()[context_indices].astype(np.int64)

        all_cl = np.concatenate([t_cl, c_cl])
        all_cmp = np.concatenate([t_cmp, c_cmp])

        # Unified spatial sort
        sort_idx = self._unified_sort_order(all_cl, all_cmp)

        # Place target and context data in unsorted arrays
        all_data = np.concatenate([target_data, context_data], axis=0)

        # Target coords (grid centers)
        t_mx_n, t_my_n, t_ox_n, t_oy_n = self._normalize_grid_coords(target_indices)
        # Context coords (raw positions)
        c_mx = self.h5_raw["midpoint_x"].ravel()[context_indices].astype(np.float64)
        c_my = self.h5_raw["midpoint_y"].ravel()[context_indices].astype(np.float64)
        c_ox = self.h5_raw["offset_x"].ravel()[context_indices].astype(np.float64)
        c_oy = self.h5_raw["offset_y"].ravel()[context_indices].astype(np.float64)
        c_mx_n, c_my_n, c_ox_n, c_oy_n = self._normalize_bare(c_mx, c_my, c_ox, c_oy)

        all_rx = np.concatenate([t_mx_n, c_mx_n])
        all_ry = np.concatenate([t_my_n, c_my_n])
        all_sx = np.concatenate([t_ox_n, c_ox_n])
        all_sy = np.concatenate([t_oy_n, c_oy_n])

        # Target cell keys
        t_ck = self._grid_cell_key[target_indices]
        c_ck = np.zeros((n_context, 4), dtype=np.int64)  # context don't have grid cell key; use 0
        all_ck = np.concatenate([t_ck, c_ck], axis=0)

        # Trace indices: grid indices for targets, raw indices for context
        all_tr_idx = np.concatenate([
            target_indices.astype(np.int64),
            context_indices.astype(np.int64),
        ])

        # SEG-Y geometry keys (for SEGY write-back)
        if self.profile is not None:
            t_kv = self._build_key_values(self.h5_grid, target_indices)
            c_kv = self._build_key_values(self.h5_raw, context_indices)
            all_kv = np.concatenate([t_kv, c_kv], axis=0)
        else:
            all_kv = np.zeros((total, 4), dtype=np.int64)

        # Masks
        t_mask = np.zeros(n_target, dtype=np.float32)   # targets: masked
        c_mask = np.ones(n_context, dtype=np.float32)   # context: observed
        all_trace_mask = np.concatenate([t_mask, c_mask])
        all_target_mask_arr = np.concatenate([
            np.ones(n_target, dtype=np.float32),
            np.zeros(n_context, dtype=np.float32),
        ])
        all_context_mask_arr = 1.0 - all_target_mask_arr
        all_valid = np.ones(total, dtype=np.float32)

        # Apply sort
        for new_pos, old_pos in enumerate(sort_idx):
            data_patch[new_pos] = all_data[old_pos]
            trace_mask[new_pos] = all_trace_mask[old_pos]
            target_slot_mask[new_pos] = all_target_mask_arr[old_pos]
            context_slot_mask[new_pos] = all_context_mask_arr[old_pos]
            valid_trace_mask[new_pos] = all_valid[old_pos]
            rx_patch[new_pos] = all_rx[old_pos]
            ry_patch[new_pos] = all_ry[old_pos]
            sx_patch[new_pos] = all_sx[old_pos]
            sy_patch[new_pos] = all_sy[old_pos]
            cell_key_values[new_pos] = all_ck[old_pos]
            key_values[new_pos] = all_kv[old_pos]
            trace_indices_all[new_pos] = all_tr_idx[old_pos]

        # masked_patch: target=0, context=data
        masked_patch[:] = data_patch[:]
        for i in range(self.trace_ps):
            if target_slot_mask[i] >= 0.5:
                masked_patch[i] = 0.0

        # Padding: if total < trace_ps, remaining slots are zero (already zero-initialized)
        # valid_trace_mask already 0 for those

        # Amp scale from all real data (target + context)
        real_data = data_patch[valid_trace_mask >= 0.5]
        amp_thres = np.percentile(np.abs(real_data), self.patch_amp_percentile) if real_data.size > 0 else 1.0
        amp_thres = max(amp_thres, 1e-6)
        data_patch /= amp_thres
        masked_patch /= amp_thres

        return {
            "data": data_patch,
            "masked_patch": masked_patch,
            "trace_mask": trace_mask,
            "target_slot_mask": target_slot_mask,
            "context_slot_mask": context_slot_mask,
            "valid_trace_mask": valid_trace_mask,
            "rx_patch": rx_patch,
            "ry_patch": ry_patch,
            "sx_patch": sx_patch,
            "sy_patch": sy_patch,
            "time_axis_2d": self._time_axis_2d(self.trace_ps),
            "amp_scale": np.float32(amp_thres),
            "amp_clip": np.float32(amp_thres),
            "amp_clip_percentile": np.float32(self.patch_amp_percentile),
            "ovt_id": np.int64(ovt_id) if ovt_id is not None else np.int64(0),
            "trace_indices": trace_indices_all,
            "cell_key_values": cell_key_values,
            "key_values": key_values,
        }

    def _normalize_bare(
        self, mx, my, ox, oy
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = self.coord_stats
        mx = np.asarray(mx, dtype=np.float64)
        my = np.asarray(my, dtype=np.float64)
        ox = np.asarray(ox, dtype=np.float64)
        oy = np.asarray(oy, dtype=np.float64)
        mx_n = np.clip((mx - s["rx_min"]) / (s["rx_max"] - s["rx_min"] + 1e-12), 0.0, 1.0)
        my_n = np.clip((my - s["ry_min"]) / (s["ry_max"] - s["ry_min"] + 1e-12), 0.0, 1.0)
        ox_n = np.clip((ox - s["sx_min"]) / (s["sx_max"] - s["sx_min"] + 1e-12), 0.0, 1.0)
        oy_n = np.clip((oy - s["sy_min"]) / (s["sy_max"] - s["sy_min"] + 1e-12), 0.0, 1.0)
        return mx_n.astype(np.float32), my_n.astype(np.float32), ox_n.astype(np.float32), oy_n.astype(np.float32)

    def _time_axis_2d(self, n_trace: int) -> np.ndarray:
        t = self.t0_ms + np.arange(self.time_ps) * self.dt_ms
        return np.tile(t.astype(np.float32), (n_trace, 1))

    # ==================================================================
    # __len__ / __getitem__
    # ==================================================================

    def __len__(self) -> int:
        return self._epoch_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.train:
            return self._get_train_item(idx)
        elif self.full_coverage:
            return self._get_test_item_full(idx)
        else:
            return self._get_test_item(idx)

    # ==================================================================
    # Training item
    # ==================================================================

    def _get_train_item(self, idx: int) -> Dict[str, Any]:
        # 1. Select OVT by sqrt(count) weight, then pick targets from that OVT
        ovt_id = int(self.rng.choice(self.ovt_ids, p=self.ovt_weights))
        ovt_pool = self.grid_observed_by_ovt[ovt_id]
        if len(ovt_pool) >= self.target_slots:
            target_grid_indices = self.rng.choice(
                ovt_pool, size=self.target_slots, replace=False
            )
        else:
            # Edge case: OVT has fewer than target_slots — take all, fill from pool
            n_fill = self.target_slots - len(ovt_pool)
            fill = self.rng.choice(
                self._target_pool, size=n_fill, replace=False
            )
            target_grid_indices = np.concatenate([ovt_pool, fill])

        # 2. Forbidden cell keys — the target cells themselves
        forbidden = set()
        for tgi in target_grid_indices:
            ck = tuple(int(v) for v in self._grid_cell_key[tgi])
            forbidden.add(ck)

        # 4. Adaptive KNN for context
        context_raw_indices = self._query_context(
            target_grid_indices, forbidden, ovt_id
        )

        # 5. Load data
        target_data = self.h5_grid["data"][target_grid_indices].astype(np.float32)
        target_data = _crop_or_pad_time(target_data, self.time_ps)

        context_data = self.h5_raw["data"][context_raw_indices].astype(np.float32)
        context_data = _crop_or_pad_time(context_data, self.time_ps)

        # 6. Build patch
        return self._build_patch(
            target_grid_indices, context_raw_indices, target_data, context_data,
            ovt_id=ovt_id,
        )

    # ==================================================================
    # Test / inference item
    # ==================================================================

    def _get_test_item(self, idx: int) -> Dict[str, Any]:
        # 1. Select OVT by sqrt(count) weight, then pick targets from that OVT
        ovt_id = int(self.rng.choice(self.ovt_ids, p=self.ovt_weights))
        ovt_pool = self.grid_missing_by_ovt[ovt_id]
        if len(ovt_pool) >= self.target_slots:
            target_grid_indices = self.rng.choice(
                ovt_pool, size=self.target_slots, replace=False
            )
        else:
            n_fill = self.target_slots - len(ovt_pool)
            fill = self.rng.choice(
                self._target_pool, size=n_fill, replace=False
            )
            target_grid_indices = np.concatenate([ovt_pool, fill])

        # 2. No forbidden keys for inference (missing cells have no raw traces)
        context_raw_indices = self._query_context(
            target_grid_indices, set(), ovt_id
        )

        # 4. Load data: targets from regular H5 (label ground truth for comparison),
        #    context from raw irregular H5 (observed traces).
        target_data = self.h5_regular["data"][target_grid_indices].astype(np.float32)
        target_data = _crop_or_pad_time(target_data, self.time_ps)

        context_data = self.h5_raw["data"][context_raw_indices].astype(np.float32)
        context_data = _crop_or_pad_time(context_data, self.time_ps)

        # 5. Build patch
        return self._build_patch(
            target_grid_indices, context_raw_indices, target_data, context_data,
            ovt_id=ovt_id,
        )

    def _get_test_item_full(self, idx: int) -> Dict[str, Any]:
        """Deterministic full-coverage inference: each grid cell exactly once.

        Reads from pre-computed _coverage_plan.  No random sampling.  The last
        chunk per OVT may have fewer than target_slots cells — valid_trace_mask
        handles variable batch sizes natively (方案A: 不跨OVT填充).
        """
        ovt_id, target_grid_indices = self._coverage_plan[idx]

        # No forbidden keys for inference
        context_raw_indices = self._query_context(
            target_grid_indices, set(), ovt_id
        )

        # target_data from regular H5 (label), context from raw H5
        target_data = self.h5_regular["data"][target_grid_indices].astype(np.float32)
        target_data = _crop_or_pad_time(target_data, self.time_ps)

        context_data = self.h5_raw["data"][context_raw_indices].astype(np.float32)
        context_data = _crop_or_pad_time(context_data, self.time_ps)

        return self._build_patch(
            target_grid_indices, context_raw_indices, target_data, context_data,
            ovt_id=ovt_id,
        )
