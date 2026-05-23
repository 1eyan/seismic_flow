#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPM V3 inference and SEGY missing-trace fill.

This script intentionally keeps one data path only:
  H5 loaded with DatasetH5_interp/train_fpmV3_ddp-compatible rules.

Trace matching key is selected by --segy_profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm import tqdm
except Exception:
    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable)


ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "flow")):
    if p not in sys.path:
        sys.path.insert(0, p)

from FPM import FlowMatchingModel
from segy_schema import SegyProfile, get_segy_profile, profile_names


DEFAULT_PROFILE = get_segy_profile()

def setup_ddp() -> Tuple[int, int, int]:
    """Initialize DDP environment. Returns (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world_size


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def none_or_str(value):
    if value is None or str(value).lower() in {"none", "null"}:
        return None
    return str(value)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gen_infer_fill_segy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "gen_infer_fill_segy.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def i32be(buf: bytes, pos_1b: int) -> int:
    return struct.unpack(">i", buf[pos_1b - 1 : pos_1b + 3])[0]


def bytes_per_sample(fmt: int) -> int:
    if fmt in (1, 2, 5):
        return 4
    if fmt == 3:
        return 2
    return 1


def scale_coord(v: int, scalar_raw: int) -> int:
    if scalar_raw == 0:
        return int(round(v))
    if scalar_raw > 0:
        return int(round(float(v) * float(scalar_raw)))
    return int(round(float(v) / float(-scalar_raw)))


def read_segy_headers(path: str, mode: str, profile: SegyProfile) -> List[dict]:
    byte_pos = profile.byte_pos
    key_columns = profile.key_columns
    headers = []
    with open(path, "rb") as f:
        f.seek(3200)
        bin_header = f.read(400)
        ns_bin = struct.unpack(">H", bin_header[20:22])[0]
        bps = bytes_per_sample(struct.unpack(">H", bin_header[24:26])[0])
        f.seek(3600)
        trace_idx = 0
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            sx = i32be(hdr, byte_pos["shot_x"])
            sy = i32be(hdr, byte_pos["shot_y"])
            rx = i32be(hdr, byte_pos["rec_x"])
            ry = i32be(hdr, byte_pos["rec_y"])
            if mode == "fixed":
                key = tuple(i32be(hdr, byte_pos[k]) for k in key_columns)
            else:
                scalar_raw = struct.unpack(">h", hdr[119:121])[0]
                computed = {
                    "shot_line": scale_coord(sy, scalar_raw),
                    "shot_no": scale_coord(sx, scalar_raw),
                    "shot_stake": scale_coord(sx, scalar_raw),
                    "recv_line": scale_coord(ry, scalar_raw),
                    "recv_no": scale_coord(rx, scalar_raw),
                    "recv_stake": scale_coord(rx, scalar_raw),
                }
                key = tuple(
                    computed[k] if k in computed else i32be(hdr, byte_pos[k])
                    for k in key_columns
                )
            ns_trace = struct.unpack(">H", hdr[114:116])[0] or ns_bin
            headers.append(
                {
                    "trace_idx": trace_idx,
                    "key": tuple(int(x) for x in key),
                    "coords": (sx, sy, rx, ry),
                    "ns": int(ns_trace),
                }
            )
            f.seek(int(ns_trace) * bps, os.SEEK_CUR)
            trace_idx += 1
    return headers


def read_segy_data(path: str) -> np.ndarray:
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        return f.trace.raw[:].astype(np.float32)


def write_segy_data(template_path: str, output_path: str, data: np.ndarray) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)
    with segyio.open(output_path, "r+", strict=False, ignore_geometry=True) as f:
        if f.tracecount != data.shape[0]:
            raise ValueError(f"SEGY tracecount={f.tracecount}, data traces={data.shape[0]}")
        for i in range(f.tracecount):
            f.trace[i] = data[i].astype(np.float32)


def build_lookup(headers: Iterable[dict]) -> Dict[Tuple[int, ...], List[int]]:
    lookup: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for h in headers:
        lookup[h["key"]].append(int(h["trace_idx"]))
    return dict(lookup)


def fit_trace(trace: np.ndarray, ns: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    if trace.size > ns:
        return trace[:ns]
    if trace.size < ns:
        # crop_or_pad_time keeps the *last* time_ps samples (deep part).
        # Pre-pad so the prediction lands at the correct (deep) time position.
        return np.pad(trace, (ns - trace.size, 0)).astype(np.float32)
    return trace


def add_prediction(pred_sum, pred_count, key, trace) -> None:
    if key not in pred_sum:
        pred_sum[key] = np.zeros_like(trace, dtype=np.float64)
    pred_sum[key] += trace.astype(np.float64)
    pred_count[key] += 1


def load_h5_first_group(h5_path: str) -> Dict[str, np.ndarray]:
    data = {}
    with h5py.File(h5_path, "r") as f:
        if "data" in f:
            for k, node in f.items():
                if isinstance(node, h5py.Dataset):
                    data[k] = node[:]
            return data
        for node in f.values():
            if hasattr(node, "keys") and "data" in node:
                return {k: node[k][:] for k in node.keys()}
    raise ValueError(f"No group containing 'data' found in H5: {h5_path}")


def h5_int_array(h5: Dict[str, np.ndarray], key: str, fallback: Optional[str] = None, default: int = 0) -> np.ndarray:
    if key in h5:
        return np.asarray(h5[key], dtype=np.int64)
    if fallback is not None and fallback in h5:
        return np.asarray(h5[fallback], dtype=np.int64)
    return np.full(len(h5["data"]), int(default), dtype=np.int64)


def h5_key_matrix(
    h5: Dict[str, np.ndarray],
    key_columns: Tuple[str, ...],
    h5_fallback: Mapping[str, str],
) -> np.ndarray:
    return np.stack(
        [h5_int_array(h5, k, fallback=h5_fallback.get(k)) for k in key_columns],
        axis=1,
    ).astype(np.int64)


def crop_or_pad_time(traces: np.ndarray, time_ps: int) -> np.ndarray:
    if traces.shape[1] > time_ps:
        return traces[:, traces.shape[1] - time_ps:]
    if traces.shape[1] < time_ps:
        return np.pad(traces, ((0, 0), (0, time_ps - traces.shape[1])), "constant")
    return traces


def compute_coord_stats(h5: Dict[str, np.ndarray]) -> Dict[str, float]:
    stats = {}
    for name in ("sx", "sy", "rx", "ry"):
        arr = np.asarray(h5[name], dtype=np.float64)
        arr = np.clip(arr, np.percentile(arr, 0.01), np.percentile(arr, 99.99))
        stats[f"{name}_min"] = float(arr.min())
        stats[f"{name}_max"] = float(arr.max())
    return stats


def normalize_coords(sx, sy, rx, ry, stats: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def norm(arr, name):
        lo = stats[f"{name}_min"]
        hi = stats[f"{name}_max"]
        arr = np.asarray(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo + 1e-12)

    return norm(sx, "sx"), norm(sy, "sy"), norm(rx, "rx"), norm(ry, "ry")


def typical_grid_step(arr: np.ndarray, eps: float = 1e-9) -> Optional[float]:
    u = np.sort(np.unique(arr))
    if u.size < 2:
        return None
    d = np.diff(u)
    d = d[d > eps]
    if d.size == 0:
        return None
    return float(np.median(d))


def compute_p_scale(h5: Dict[str, np.ndarray], stats: Dict[str, float]) -> Dict[str, float]:
    deltas = {}
    for name in ("sx", "sy", "rx", "ry"):
        arr = np.asarray(h5[name], dtype=np.float64)
        ds = typical_grid_step(arr)
        lo, hi = stats[f"{name}_min"], stats[f"{name}_max"]
        if ds is not None and (hi - lo) > 0:
            deltas[name] = float((hi - lo) / (2 * ds))
    return deltas


class H5FpmInferenceDataset(Dataset):
    """Inference-time H5 dataset with train_fpmV3_ddp/DatasetH5_interp rules computed locally."""

    def __init__(
        self,
        h5_regular: str,
        h5_mask: str,
        gather_mode: str,
        time_ps: Optional[int],
        missing_eps: float,
        profile: SegyProfile,
        use_p_scale: bool = False,
    ):
        self.profile = profile
        self.key_columns = tuple(profile.key_columns)
        self.h5_fallback = dict(profile.h5_fallback)
        self.gather_modes = dict(profile.gather_modes)
        self.h5_regular = load_h5_first_group(h5_regular)
        self.h5_mask = load_h5_first_group(h5_mask)
        n_traces = len(self.h5_mask["data"])
        if len(self.h5_regular["data"]) != n_traces:
            raise ValueError(
                f"h5_regular traces={len(self.h5_regular['data'])}, "
                f"h5_mask traces={n_traces}; they must contain the same trace keys/count"
            )
        self.gather_mode = gather_mode
        self.missing_eps = float(missing_eps)
        self.time_ps = int(time_ps if time_ps is not None else self.h5_mask["data"].shape[1])
        self.coord_stats = compute_coord_stats(self.h5_regular)
        self.p_scale = compute_p_scale(self.h5_regular, self.coord_stats)
        # 在归一化前应用 p_scale：将 stats 的 min/max 乘以 p_scale
        if use_p_scale and self.p_scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.p_scale.get(name)
                if s is not None:
                    self.coord_stats[f"{name}_min"] *= s
                    self.coord_stats[f"{name}_max"] *= s
            print(f"[H5FpmInferenceDataset] p_scale applied to coord_stats: {self.p_scale}")
        self.sx_all = np.asarray(self.h5_mask["sx"], dtype=np.float32)
        self.sy_all = np.asarray(self.h5_mask["sy"], dtype=np.float32)
        self.rx_all = np.asarray(self.h5_mask["rx"], dtype=np.float32)
        self.ry_all = np.asarray(self.h5_mask["ry"], dtype=np.float32)
        self.shot_line_all = h5_int_array(self.h5_mask, "shot_line")
        # Load per-trace identity fields dynamically from the active profile.
        # This populates self.shot_line_all, self.shot_no_all, self.recv_line_all,
        # self.recv_no_all (and any *_stake variants in profile.key_columns) so both
        # mask_keys (profile.key_columns-driven) and _build_groups (gather_mode-driven)
        # have the attributes they need.
        for _field, _fallback in profile.all_h5_key_fields.items():
            # Skip shot_line — already loaded above explicitly for clarity
            if _field == "shot_line":
                continue
            setattr(self, f"{_field}_all",
                    h5_int_array(self.h5_mask, _field, fallback=_fallback))
        self.mask_keys = np.stack(
            [getattr(self, f"{k}_all") for k in self.key_columns],
            axis=1,
        ).astype(np.int64)
        regular_keys = h5_key_matrix(self.h5_regular, self.key_columns, self.h5_fallback)
        self.regular_aligned_by_key = not np.array_equal(regular_keys, self.mask_keys)
        if self.regular_aligned_by_key:
            lookup = {}
            for i, key in enumerate(regular_keys):
                tkey = tuple(int(v) for v in key)
                if tkey in lookup:
                    raise ValueError(f"Duplicate key in h5_regular: {tkey}")
                lookup[tkey] = i
            aligned = []
            missing_keys = []
            for key in self.mask_keys:
                tkey = tuple(int(v) for v in key)
                idx = lookup.get(tkey)
                if idx is None:
                    missing_keys.append(tkey)
                else:
                    aligned.append(idx)
            if missing_keys:
                raise ValueError(
                    "h5_regular is missing keys from h5_mask, first keys: "
                    f"{missing_keys[:10]}"
                )
            self.regular_data = np.asarray(self.h5_regular["data"], dtype=np.float32)[np.asarray(aligned, dtype=np.int64)]
        else:
            self.regular_data = np.asarray(self.h5_regular["data"], dtype=np.float32)
        self.line_indices, self.group_keys, self.group_desc = self._build_groups()
        self.trace_counts = [int(len(self.line_indices[gk])) for gk in self.group_keys]

    def _build_groups(self):
        mode = self.gather_mode
        field_names = self.gather_modes.get(mode)
        if field_names is None:
            valid = ", ".join(sorted(self.gather_modes))
            raise ValueError(f"Unsupported gather_mode={mode!r} for profile={self.profile.name}; choose one of: {valid}")

        parts = [getattr(self, f"{fn}_all") for fn in field_names]
        desc = " × ".join(field_names)

        keys = np.column_stack(parts)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        group_keys = [tuple(int(v) for v in row) for row in uniq]
        line_indices = {group_keys[i]: np.flatnonzero(inv == i).astype(np.int64) for i in range(len(group_keys))}
        group_keys = sorted(group_keys)
        return {g: line_indices[g] for g in group_keys}, group_keys, desc

    def __len__(self) -> int:
        return len(self.group_keys)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        group_key = self.group_keys[idx]
        trace_idx = self.line_indices[group_key]

        target = crop_or_pad_time(np.asarray(self.regular_data[trace_idx], dtype=np.float32), self.time_ps)
        observed = crop_or_pad_time(np.asarray(self.h5_mask["data"][trace_idx], dtype=np.float32), self.time_ps)
        trace_mask = np.any(np.abs(observed) > self.missing_eps, axis=1)

        finite_obs = observed[np.isfinite(observed)]
        thres = float(np.percentile(np.abs(finite_obs), 99.5)) if finite_obs.size else 1e-6
        thres = max(thres, 1e-6)
        masked_patch = np.clip(observed, -thres, thres) / thres
        data_patch = np.clip(target, -thres, thres) / thres

        sx = self.sx_all[trace_idx]
        sy = self.sy_all[trace_idx]
        rx = self.rx_all[trace_idx]
        ry = self.ry_all[trace_idx]
        sx_n, sy_n, rx_n, ry_n = normalize_coords(sx, sy, rx, ry, self.coord_stats)

        gather_key = self.mask_keys[trace_idx].astype(np.int32)

        return {
            "data": data_patch.astype(np.float32),
            "masked_patch": masked_patch.astype(np.float32),
            "trace_mask": trace_mask.astype(np.float32),
            "sx_patch": sx_n.astype(np.float32),
            "sy_patch": sy_n.astype(np.float32),
            "rx_patch": rx_n.astype(np.float32),
            "ry_patch": ry_n.astype(np.float32),
            "gather_key": gather_key,
            "amp_scale": np.float32(thres),
            "line_group_idx": np.int64(idx),
            "trace_indices": trace_idx.astype(np.int64),
        }


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that groups dataset indices by trace count into equal-trace-count batches.

    Yields lists of indices where all items in a batch have identical trace counts.
    No padding is needed — the model sees only real traces with real coordinates.
    """

    def __init__(self, dataset: H5FpmInferenceDataset, batch_size: int,
                 shuffle: bool = False, drop_last: bool = False,
                 seed: int = 42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed) if shuffle else None
        trace_counts = getattr(dataset, "trace_counts", None)
        if trace_counts is None or len(trace_counts) != len(dataset):
            raise ValueError(
                "BucketBatchSampler requires dataset.trace_counts (list of trace counts per sample)."
            )
        self.trace_counts = trace_counts
        self._build_batches(list(range(len(self.trace_counts))))

    def _build_batches(self, indices: List[int]):
        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            buckets[int(self.trace_counts[idx])].append(idx)

        batches: List[List[int]] = []
        for count, bucket_indices in sorted(buckets.items()):
            if self.rng is not None:
                self.rng.shuffle(bucket_indices)
            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.rng is not None:
            self.rng.shuffle(batches)
        self._batches = batches

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class DistributedBucketBatchSampler(BucketBatchSampler):
    """DDP-aware BucketBatchSampler: builds all batches from the full dataset first
    (identical to single-GPU BucketBatchSampler), then distributes complete batches
    round-robin across ranks.

    Since the unit of distribution is a *batch* (not individual samples), the total
    batch count differs by at most 1 between any two ranks — regardless of how many
    trace-count buckets exist.  This guarantees that dist.barrier / all_gather at
    the end of inference will not time out.
    """

    def __init__(self, dataset: H5FpmInferenceDataset, batch_size: int,
                 num_replicas: int, rank: int,
                 shuffle: bool = False, drop_last: bool = False,
                 seed: int = 42):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed) if shuffle else None
        trace_counts = getattr(dataset, "trace_counts", None)
        if trace_counts is None or len(trace_counts) != len(dataset):
            raise ValueError(
                "DistributedBucketBatchSampler requires dataset.trace_counts "
                "(list of trace counts per sample)."
            )
        self.trace_counts = trace_counts
        total = len(self.trace_counts)

        # Step 1 — build ALL batches from the full dataset (same as single-GPU BucketBatchSampler)
        all_batches: List[List[int]] = []
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i in range(total):
            buckets[int(self.trace_counts[i])].append(i)

        for _count, bucket_indices in sorted(buckets.items()):
            if self.rng is not None:
                self.rng.shuffle(bucket_indices)
            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        if self.rng is not None:
            self.rng.shuffle(all_batches)

        # Step 2 — round-robin batches across ranks.
        # Total batch count differs by at most 1 (N % W at worst).
        self._batches = all_batches[self.rank::self.num_replicas]
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def collate_batched(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Simple stack collate — all samples in batch have identical trace counts (guaranteed by BucketBatchSampler)."""
    stacked = {}
    for key in ("masked_patch", "data", "trace_mask",
                "sx_patch", "sy_patch", "rx_patch", "ry_patch"):
        stacked[key] = np.stack([b[key] for b in batch], axis=0)
    stacked["gather_key"] = np.stack([b["gather_key"] for b in batch], axis=0)
    stacked["amp_scale"] = np.array([float(b["amp_scale"]) for b in batch], dtype=np.float32)
    stacked["line_group_idx"] = np.array([int(b["line_group_idx"]) for b in batch], dtype=np.int64)
    stacked["trace_indices"] = np.stack([b["trace_indices"] for b in batch], axis=0)
    return stacked


def visualize_batch(
    masked_patch: np.ndarray,
    pred: np.ndarray,
    data: np.ndarray,
    trace_mask: np.ndarray,
    sample_idx: int,
    vis_dir: Path,
) -> None:
    """Visualize masked input, prediction, ground truth, and residual side by side."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    missing = trace_mask < 0.5
    residual = pred - data
    vmax = float(max(masked_patch.max(), pred.max(), data.max()))
    res_abs_max = float(max(abs(residual.min()), abs(residual.max()), 1e-6))
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    for ax, img, title, vmn, vmx, cmap in zip(
        axes,
        [masked_patch, pred, data, residual],
        ["Masked Input", "FPM Prediction", "Ground Truth", "Residual (Pred - GT)"],
        [-vmax, -vmax, -vmax, -res_abs_max],
        [vmax, vmax, vmax, res_abs_max],
        ["seismic", "seismic", "seismic", "seismic"],
    ):
        ax.imshow(img.T, aspect="auto", cmap=cmap, vmin=vmn, vmax=vmx, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time Sample")
    n_traces = trace_mask.shape[0]
    n_missing = int(missing.sum())
    fig.suptitle(f"Sample {sample_idx} | {n_traces} traces | {n_missing} missing", fontsize=12)
    fig.savefig(vis_dir / f"batch_{sample_idx:04d}.png", dpi=120)
    plt.close(fig)


def make_backbone(args: argparse.Namespace) -> torch.nn.Module:
    if "gated" in args.model_type:
        from models.gated_seisdit_gen import create_gated_seisdit

        return create_gated_seisdit(
            image_channels=2,
            d_model=args.gated_d_model,
            d_ff=args.gated_d_ff,
            nhead=args.gated_nhead,
            num_encoder_layers=args.gated_encoder_layers,
            num_bottleneck_layers=args.gated_bottleneck_layers,
            num_decoder_layers=args.gated_decoder_layers,
            chunk_length=args.chunk_length_flow,
            chunk_overlap=args.chunk_overlap_flow,
            use_energy_stats=args.use_energy_mlp,
            use_missing_embed=args.use_missing_embedding,
            headwise_attn_output_gate=args.headwise_attn_output_gate,
            elementwise_attn_output_gate=args.elementwise_attn_output_gate,
        )
    if "tp" in args.model_type or "vit" in args.model_type:
        from models.seisdit_vit_bottleneck import SeisDiTRope

        return SeisDiTRope(
            image_channels=2,
            n_channels=32,
            f_dict=None,
            num_layers=8,
            d_model=512,
            pe_type="transformer",
        )

    from models.seisdit_trace_axis import SeisDiTRopeV2
    if 'trace_axis' in args.model_type:
        return SeisDiTRopeV2(
            image_channels=2,
            n_channels=32,
            f_dict=None,
            num_layers=8,
            d_model=512,
            pe_type="transformer",
            missing_focus_adapter=args.use_missing_embedding,
            geom_mode=args.geom_mode,
        )
    else:
        raise ValueError(f"Unsupported model_type={args.model_type!r}")


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool, logger: logging.Logger) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=strict)
    logger.info("checkpoint loaded: missing=%s unexpected=%s", result.missing_keys, result.unexpected_keys)


def flow_sample(
    fpm: FlowMatchingModel,
    x_norm: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if x_norm.ndim == 4:
        x_cond = torch.from_numpy(x_norm.astype(np.float32)).to(device)
    elif x_norm.ndim == 3:
        x_cond = torch.from_numpy(x_norm.astype(np.float32))[:, None, :, :].to(device)
    elif x_norm.ndim == 2:
        x_cond = torch.from_numpy(x_norm.astype(np.float32))[None, None, :, :].to(device)
    else:
        x_cond = torch.from_numpy(x_norm.astype(np.float32))[None].to(device)
    coords_t = torch.from_numpy(coords.astype(np.float32)).to(device)
    cond = (
        coords_t[:, :, 2].float(),
        coords_t[:, :, 3].float(),
        coords_t[:, :, 0].float(),
        coords_t[:, :, 1].float(),
    )
    with torch.inference_mode():
        pred = fpm.sample(condL=cond, x_cond=x_cond, time_axis=None)
    return pred.squeeze(1).detach().cpu().numpy().astype(np.float32)


def chunked_inference(
    fpm: FlowMatchingModel,
    x_norm: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
    max_traces: int,
    overlap: int,
) -> np.ndarray:
    """Split a large-trace group into overlapping spatial chunks along the trace axis.

    x_norm:  (B, 1, H, T)  — batch of masked patches (channel dim already added)
    coords:  (B, H, 4)     — normalised coordinates
    Returns: (B, H, T)      — stitched predictions, overlap regions crossfaded
    """
    B, _, H, T = x_norm.shape
    stride = max(1, max_traces - overlap)
    pred_accum = np.zeros((B, H, T), dtype=np.float64)
    pred_weight = np.zeros((B, H), dtype=np.float64)

    chunk_start = 0
    while chunk_start < H:
        chunk_end = min(chunk_start + max_traces, H)
        chunk_x = x_norm[:, :, chunk_start:chunk_end, :]       # (B, 1, chunk_H, T)
        chunk_coords = coords[:, chunk_start:chunk_end, :]     # (B, chunk_H, 4)

        chunk_pred = flow_sample(fpm, chunk_x, chunk_coords, device)  # (B, chunk_H, T)

        chunk_len = chunk_end - chunk_start
        chunk_weights = np.ones(chunk_len, dtype=np.float64)

        if chunk_start > 0:
            left_overlap = min(overlap, chunk_len)
            chunk_weights[:left_overlap] = np.linspace(0.0, 1.0, left_overlap, dtype=np.float64)

        if chunk_end < H:
            right_overlap = min(overlap, chunk_len)
            chunk_weights[-right_overlap:] = np.linspace(1.0, 0.0, right_overlap, dtype=np.float64)

        chunk_weights_3d = chunk_weights[None, :, None]  # (1, chunk_H, 1)
        pred_accum[:, chunk_start:chunk_end, :] += chunk_pred.astype(np.float64) * chunk_weights_3d
        pred_weight[:, chunk_start:chunk_end] += chunk_weights

        chunk_start += stride

    pred_weight = np.maximum(pred_weight, 1e-10)
    return (pred_accum / pred_weight[:, :, None]).astype(np.float32)


def build_dataset(args: argparse.Namespace, logger: logging.Logger, profile: SegyProfile) -> H5FpmInferenceDataset:
    logger.info("building local H5FpmInferenceDataset with DatasetH5_interp-compatible rules")
    dataset = H5FpmInferenceDataset(
        h5_regular=args.h5_regular,
        h5_mask=args.h5_mask,
        gather_mode=args.gather_mode,
        time_ps=args.time_ps,
        missing_eps=args.h5_missing_eps,
        profile=profile,
        use_p_scale=args.use_p_scale,
    )
    logger.info(
        "dataset ready: samples=%d gather_mode=%s group_desc=%s time_ps=%d h5_missing_eps=%.3g regular_aligned_by_key=%s",
        len(dataset),
        args.gather_mode,
        dataset.group_desc,
        dataset.time_ps,
        dataset.missing_eps,
        dataset.regular_aligned_by_key,
    )
    return dataset


def infer_h5_dataset(args, fpm, device, logger, ns: int, profile: SegyProfile, dataset=None,
                     sampler=None, rank: int = 0, world_size: int = 1,
                     log_interval: int = 1):
    if dataset is None:
        dataset = build_dataset(args, logger, profile)
    use_batch = getattr(args, 'inference_batch_size', 1) > 1

    if use_batch:
        if sampler is not None:
            batch_sampler = DistributedBucketBatchSampler(
                dataset, args.inference_batch_size,
                num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=False)
        else:
            batch_sampler = BucketBatchSampler(dataset, args.inference_batch_size,
                                               shuffle=False, drop_last=False)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_batched,
            multiprocessing_context="spawn",
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            multiprocessing_context="spawn",
        )

    pred_sum, pred_count = {}, defaultdict(int)
    total_missing = 0
    total_traces = 0
    vis_dir = Path(args.output_dir) / "vis"
    do_vis = args.visualize and rank == 0
    vis_max = args.vis_batches if args.vis_batches > 0 else float("inf")
    desc = f"FPM V3 H5 inference [rank {rank}/{world_size}]" if world_size > 1 else "FPM V3 H5 inference"
    iterator = tqdm(loader, total=len(loader), desc=desc, unit="batch", disable=not args.progress or rank != 0)
    n_loader = len(loader)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    global_sample_idx = 0
    for batch_idx, batch in enumerate(iterator):
        if use_batch:
            current_bs = int(to_numpy(batch["amp_scale"]).shape[0])
        else:
            current_bs = 1

        x_all = to_numpy(batch["masked_patch"]).astype(np.float32)
        coords_all = np.stack(
            [
                to_numpy(batch["sx_patch"]),
                to_numpy(batch["sy_patch"]),
                to_numpy(batch["rx_patch"]),
                to_numpy(batch["ry_patch"]),
            ],
            axis=2,
        ).astype(np.float32)
        trace_mask_all = to_numpy(batch["trace_mask"]).astype(np.float32)
        scale_all = to_numpy(batch["amp_scale"]).astype(np.float32)
        data_all = to_numpy(batch["data"]).astype(np.float32)
        gather_keys_all = to_numpy(batch["gather_key"])

        x_for_model = x_all[:, None, :, :].astype(np.float32)
        coords_for_model = coords_all.astype(np.float32)
        if not use_batch:
            x_for_model = x_for_model[None] if x_for_model.ndim == 3 else x_for_model
            coords_for_model = coords_for_model[None] if coords_for_model.ndim == 2 else coords_for_model

        n_traces = x_for_model.shape[2]
        max_chunk = getattr(args, "max_traces_per_chunk", 0)
        max_total = getattr(args, "max_total_traces_per_batch", 0)
        chunk_overlap = getattr(args, "chunk_overlap", 32)

        # Determine sub-batch size: limit total traces (B × H) per forward pass
        B = x_for_model.shape[0]
        need_chunk = max_chunk > 0 and n_traces > max_chunk
        total_traces = B * n_traces
        if max_total > 0 and total_traces > max_total:
            max_groups_per_sub = max(1, max_total // n_traces)
        else:
            max_groups_per_sub = B
        if max_groups_per_sub >= B:
            # Single forward pass (may still need chunking for large H)
            if need_chunk:
                pred_all = chunked_inference(fpm, x_for_model, coords_for_model, device,
                                             max_chunk, chunk_overlap)
            else:
                pred_all = flow_sample(fpm, x_for_model, coords_for_model, device)
        else:
            # Split into sub-batches, each sub-batch may also need chunking
            pred_parts = []
            for start in range(0, B, max_groups_per_sub):
                end = min(start + max_groups_per_sub, B)
                sub_x = x_for_model[start:end]
                sub_coords = coords_for_model[start:end]
                if need_chunk:
                    part = chunked_inference(fpm, sub_x, sub_coords, device,
                                             max_chunk, chunk_overlap)
                else:
                    part = flow_sample(fpm, sub_x, sub_coords, device)
                pred_parts.append(part)
            pred_all = np.concatenate(pred_parts, axis=0)
         
        for s in range(current_bs):
            trace_mask = trace_mask_all[s].astype(np.float32)
            scale = float(scale_all[s]) if scale_all.ndim > 0 else float(scale_all)
            pred = pred_all[s] * scale
            keys = gather_keys_all[s].astype(np.int64)
            missing = trace_mask < 0.5
            n_traces = int(trace_mask.size)
            missing_count = int(missing.sum())
            total_missing += missing_count
            total_traces += n_traces

            if do_vis and global_sample_idx < vis_max:
                masked_patch = x_all[s] * scale
                data_gt = data_all[s] * scale
                visualize_batch(masked_patch, pred, data_gt, trace_mask, global_sample_idx, vis_dir)

            for j, is_missing in enumerate(missing):
                if is_missing:
                    add_prediction(pred_sum, pred_count, tuple(int(v) for v in keys[j]), fit_trace(pred[j], ns))

            if args.progress and rank == 0 and s == current_bs - 1:
                iterator.set_postfix(batch=batch_idx, total_traces=total_traces, total_missing=total_missing)
            global_sample_idx += 1

        if batch_idx % log_interval == 0 or batch_idx == n_loader - 1:
            logger.info(
                "progress: %d/%d (%.1f%%) | total_traces=%d total_missing=%d",
                batch_idx + 1,
                n_loader,
                100.0 * (batch_idx + 1) / n_loader if n_loader > 0 else 100,
                total_traces,
                total_missing,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time
    logger.info(
        "inference finished: %.2fs | samples=%d traces=%d dataset_missing=%d prediction_keys=%d",
        seconds,
        len(dataset) if sampler is None else len(loader.dataset),
        total_traces,
        total_missing,
        len(pred_sum),
    )
    return pred_sum, pred_count, seconds, {
        "dataset_samples": int(len(dataset) if sampler is None else len(loader.dataset)),
        "dataset_traces": int(total_traces),
        "dataset_missing": int(total_missing),
        "prediction_keys": int(len(pred_sum)),
    }


def fill_and_verify(args, headers, missing_global, pred_sum, pred_count, logger,
                    profile: SegyProfile, label_data=None) -> dict:
    mask_data = read_segy_data(args.mask_path)
    lookup = build_lookup(headers)
    out = mask_data.copy()
    ns = mask_data.shape[1]
    written, unmatched = set(), []

    for key, total in pred_sum.items():
        indices = lookup.get(key)
        if not indices:
            unmatched.append(key)
            continue
        trace = fit_trace(total / max(pred_count[key], 1), ns)
        wrote = False
        for trace_idx in indices:
            if missing_global[trace_idx]:
                out[trace_idx] = trace
                written.add(trace_idx)
                wrote = True
        if not wrote:
            unmatched.append(key)

    write_start = time.perf_counter()
    write_segy_data(args.mask_path, args.output_segy, out)
    writeback_seconds = time.perf_counter() - write_start

    written_sorted = sorted(written)
    missing_indices = set(np.flatnonzero(missing_global).tolist())
    unfilled = sorted(missing_indices - written)
    after = read_segy_data(args.output_segy)
    still_missing = np.flatnonzero(missing_global & np.all(np.abs(after) <= args.missing_eps, axis=1)).tolist()
    observed_changed = np.flatnonzero(
        (~missing_global) & np.any(np.abs(after - mask_data) > args.missing_eps, axis=1)
    ).tolist()

    # Residual SEGY: pred - label, only for filled traces
    residual_stats = {}
    if label_data is not None:
        residual = np.zeros_like(out)
        for trace_idx in written_sorted:
            residual[trace_idx] = out[trace_idx] - label_data[trace_idx]
        write_segy_data(args.mask_path, args.output_residual_segy, residual)
        filled_residuals = residual[written_sorted]
        residual_stats = {
            "residual_max_abs": float(np.max(np.abs(filled_residuals))) if filled_residuals.size else 0.0,
            "residual_mean_abs": float(np.mean(np.abs(filled_residuals))) if filled_residuals.size else 0.0,
            "output_residual_segy": args.output_residual_segy,
        }
        logger.info("residual SEGY written: %s | max_abs=%.6g mean_abs=%.6g",
                     args.output_residual_segy, residual_stats["residual_max_abs"], residual_stats["residual_mean_abs"])

    summary = {
        "segy_profile": profile.name,
        "key_columns": list(profile.key_columns),
        "segy_traces": int(mask_data.shape[0]),
        "segy_samples": int(mask_data.shape[1]),
        "missing_total": int(len(missing_indices)),
        "written": int(len(written_sorted)),
        "unfilled": int(len(unfilled)),
        "still_missing_after_write": int(len(still_missing)),
        "observed_changed": int(len(observed_changed)),
        "prediction_keys": int(len(pred_sum)),
        "unmatched_prediction_keys": int(len(unmatched)),
        "writeback_seconds": round(writeback_seconds, 3),
        "output_segy": args.output_segy,
        **residual_stats,
    }
    save_reports(Path(args.output_dir), headers, written_sorted, unfilled, still_missing,
                 observed_changed, unmatched, summary, profile.key_columns)
    logger.info("writeback summary: %s", summary)
    if args.strict_fill and (unfilled or still_missing or observed_changed):
        raise RuntimeError(
            "strict fill failed: "
            f"unfilled={len(unfilled)} still_missing={len(still_missing)} observed_changed={len(observed_changed)}"
        )
    return summary


def save_reports(output_dir: Path, headers, written, unfilled, still_missing,
                 observed_changed, unmatched, summary, key_columns: Tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    header_by_idx = {int(h["trace_idx"]): h["key"] for h in headers}
    for name, indices in (
        ("filled_missing_keys.csv", written),
        ("unfilled_missing_keys.csv", unfilled),
        ("still_missing_after_write_keys.csv", still_missing),
        ("observed_changed_keys.csv", observed_changed),
    ):
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trace_idx", *key_columns])
            for idx in indices:
                default_key = ("",) * len(key_columns)
                writer.writerow([idx, *header_by_idx.get(int(idx), default_key)])
    with open(output_dir / "unmatched_prediction_keys.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(key_columns)
        writer.writerows(unmatched)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPM V3 H5 inference and fill missing traces into SEGY")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5_irregular", default=None, help="Deprecated: no longer used, kept for backward compatibility")
    parser.add_argument("--h5_regular", required=True)
    parser.add_argument("--h5_mask", required=True, help="H5 with missing traces; same trace keys as h5_regular")
    parser.add_argument("--mask_path", required=True, help="SEGY with missing traces; only zero traces are replaced")
    parser.add_argument("--label_segy", default=None, help="Ground truth SEGY for residual computation (optional)")
    parser.add_argument("--output_dir", default="gen_fill_results")
    parser.add_argument("--output_segy", default=None)
    parser.add_argument("--output_residual_segy", default=None, help="Residual SEGY output path (default: {output_dir}/residual.sgy)")
    parser.add_argument("--segy_profile", choices=profile_names(), default=DEFAULT_PROFILE.name,
                        help="SEG-Y header/key profile used for mask_path and H5 key alignment")
    parser.add_argument("--device", default="cuda:0",
                        help="Single-GPU device (only used outside torchrun; under torchrun LOCAL_RANK is used)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--inference_batch_size", type=int, default=1, help="Batch size for inference (pads groups to max traces in batch)")
    parser.add_argument("--max_traces_per_chunk", type=int, default=0,
                        help="Max traces per model forward; groups larger than this are split into overlapping spatial chunks (0 = no chunking)")
    parser.add_argument("--chunk_overlap", type=int, default=32,
                        help="Overlap between adjacent spatial chunks in number of traces")
    parser.add_argument("--max_total_traces_per_batch", type=int, default=0,
                        help="Max total traces (batch_size × n_traces) per forward pass; batches exceeding this are split into sub-batches (0 = no limit)")
    parser.add_argument(
        "--gather_mode",
        default=None,
        help="H5 sample grouping; valid values depend on --segy_profile",
    )
    parser.add_argument("--time_ps", type=int, default=None)
    parser.add_argument("--missing_eps", type=float, default=1e-10)
    parser.add_argument("--h5_missing_eps", type=float, default=None, help="trace is missing in h5_mask if all abs(data) <= this; defaults to missing_eps")
    parser.add_argument("--header_mode", choices=["fixed", "self_computed"], default=None)
    parser.add_argument("--non_strict_load", dest="strict_load", action="store_false")
    parser.add_argument("--strict_fill", dest="strict_fill", action="store_true")
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(strict_load=True, strict_fill=False, progress=True)

    parser.add_argument("--visualize", type=str2bool, default=False, help="Enable per-batch visualization of masked input, prediction, and ground truth")
    parser.add_argument("--vis_batches", type=int, default=0, help="Max number of batches to visualize (0 = all)")
    parser.add_argument("--model_type", default="gated")
    parser.add_argument("--gated_d_model", type=int, default=1080)
    parser.add_argument("--gated_d_ff", type=int, default=2048)
    parser.add_argument("--gated_nhead", type=int, default=6)
    parser.add_argument("--gated_encoder_layers", type=int, default=4)
    parser.add_argument("--gated_bottleneck_layers", type=int, default=4)
    parser.add_argument("--gated_decoder_layers", type=int, default=4)
    parser.add_argument("--chunk_length_flow", type=int, default=256)
    parser.add_argument("--chunk_overlap_flow", type=float, default=0.5)
    parser.add_argument("--use_energy_mlp", type=str2bool, default=False)
    parser.add_argument("--use_missing_embedding", type=str2bool, default=True)
    parser.add_argument("--headwise_attn_output_gate", type=str2bool, default=True)
    parser.add_argument("--elementwise_attn_output_gate", type=str2bool, default=False)
    parser.add_argument("--geom_mode", choices=["source", "receiver", "relative"], default="source")
    parser.add_argument("--use_p_scale", type=str2bool, default=False, help="apply p_scale to gated model RoPE coordinates")

    parser.add_argument("--path_type", choices=["Linear", "GVP", "VP"], default="Linear")
    parser.add_argument("--prediction", choices=["velocity", "score", "noise"], default="velocity")
    parser.add_argument("--loss_weight", type=none_or_str, default=None)
    parser.add_argument("--sampling_method", choices=["ode", "sde"], default="ode")
    parser.add_argument("--ode_sampling_method", default="dopri5")
    parser.add_argument("--ode_num_steps", type=int, default=50)
    parser.add_argument("--ode_atol", type=float, default=1e-6)
    parser.add_argument("--ode_rtol", type=float, default=1e-3)
    parser.add_argument("--sde_sampling_method", default="Euler")
    parser.add_argument("--sde_num_steps", type=int, default=250)
    args = parser.parse_args()

    profile = get_segy_profile(args.segy_profile)
    if args.gather_mode is None:
        args.gather_mode = profile.default_gather_mode
    elif args.gather_mode not in profile.gather_modes:
        valid = ", ".join(sorted(profile.gather_modes))
        parser.error(f"--gather_mode {args.gather_mode!r} is not valid for --segy_profile {profile.name!r}; choose one of: {valid}")
    if args.header_mode is None:
        args.header_mode = profile.default_header_mode

    args.output_dir = str(Path(args.output_dir).resolve())
    args.output_segy = args.output_segy or str(Path(args.output_dir) / "filled_missing.sgy")
    args.output_residual_segy = args.output_residual_segy or str(Path(args.output_dir) / "residual.sgy")
    args.h5_missing_eps = args.missing_eps if args.h5_missing_eps is None else args.h5_missing_eps
    return args


def main() -> None:
    args = parse_args()
    profile = get_segy_profile(args.segy_profile)
    rank, local_rank, world_size = setup_ddp()

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        logger = setup_logger(Path(args.output_dir)) if rank == 0 else logging.getLogger("gen_infer")
        if rank != 0:
            logger.setLevel(logging.WARNING)
    else:
        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        logger = setup_logger(Path(args.output_dir))

    if rank == 0:
        logger.info("args: %s", vars(args))
        logger.info(
            "data path: local H5 DatasetH5_interp-compatible | segy_profile=%s | gather_mode=%s | key=%s",
            profile.name,
            args.gather_mode,
            profile.key_columns,
        )

    total_start = time.perf_counter()
    mask_data = read_segy_data(args.mask_path)
    headers = read_segy_headers(args.mask_path, args.header_mode, profile)
    if len(headers) != mask_data.shape[0]:
        raise ValueError(f"header_count={len(headers)} but segy_traces={mask_data.shape[0]}")
    missing_global = np.all(np.abs(mask_data) <= args.missing_eps, axis=1)
    if rank == 0:
        logger.info(
            "template SEGY: traces=%d samples=%d missing=%d header_mode=%s",
            mask_data.shape[0],
            mask_data.shape[1],
            int(missing_global.sum()),
            args.header_mode,
        )

    label_data = None
    if args.label_segy:
        label_data = read_segy_data(args.label_segy)
        if label_data.shape != mask_data.shape:
            raise ValueError(f"label_segy shape {label_data.shape} != mask_segy shape {mask_data.shape}")
        if rank == 0:
            logger.info("label SEGY loaded: %s shape=%s", args.label_segy, label_data.shape)

    dataset = build_dataset(args, logger if rank == 0 else logging.getLogger("gen_infer"), profile)
    if rank == 0:
        logger.info("use_p_scale=%s p_scale=%s", args.use_p_scale, getattr(dataset, 'p_scale', None))
        logger.info("inference_batch_size=%d world_size=%d", args.inference_batch_size, world_size)

    backbone = make_backbone(args).to(device).eval()
    load_checkpoint(backbone, args.checkpoint, args.strict_load,
                    logger if rank == 0 else logging.getLogger("gen_infer"))
    if rank == 0:
        logger.info("model=%s params=%d device=%s", args.model_type, sum(p.numel() for p in backbone.parameters()), device)

    # Create FlowMatchingModel once — trace_num/time_steps are placeholders;
    # the true shape is inferred from x_cond at each sample() call.
    fpm = FlowMatchingModel(
        model=backbone,
        trace_num=2048,
        time_steps=2048,
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=None,
        sample_eps=None,
        sample_num=1,
        device=device,
        sup_mode="all",
        use_coherence=False,
        sigma_obs=0.001,
        use_bayesian=False,
        sampling_method=args.sampling_method,
        ode_sampling_method=args.ode_sampling_method,
        ode_num_steps=args.ode_num_steps,
        ode_atol=args.ode_atol,
        ode_rtol=args.ode_rtol,
        sde_sampling_method=args.sde_sampling_method,
        sde_num_steps=args.sde_num_steps,
    ).eval()

    if world_size > 1:
        # Inference-only DDP: each rank processes its own data subset independently.
        # No gradient sync needed — the model runs raw.  Only dist.all_gather_object
        # below is used to merge predictions across ranks.
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if rank == 0:
            logger.info("DDP: world_size=%d rank=%d local_rank=%d", world_size, rank, local_rank)
    else:
        sampler = None

    log_interval = max(1, (len(dataset) // world_size) // 10) if world_size > 1 else max(1, len(dataset) // 10)
    pred_sum, pred_count, inference_seconds, infer_stats = infer_h5_dataset(
        args, fpm, device, logger, ns=mask_data.shape[1], profile=profile, dataset=dataset,
        sampler=sampler, rank=rank, world_size=world_size,
        log_interval=log_interval,
    )

    # Gather predictions across all ranks
    if world_size > 1:
        dist.barrier()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (pred_sum, dict(pred_count), infer_stats))
        if rank == 0:
            merged_sum, merged_count = {}, defaultdict(int)
            merged_stats = {"dataset_samples": 0, "dataset_traces": 0, "dataset_missing": 0, "prediction_keys": 0}
            for ps, pc, st in gathered:
                for k, v in ps.items():
                    if k not in merged_sum:
                        merged_sum[k] = v.copy()
                    else:
                        merged_sum[k] += v
                for k, v in pc.items():
                    merged_count[k] += v
                for sk in merged_stats:
                    merged_stats[sk] += st.get(sk, 0)
            pred_sum, pred_count = merged_sum, merged_count
            infer_stats = merged_stats
        dist.barrier()
    
    if rank == 0:
        print('begin fill segy ')
        summary = fill_and_verify(args, headers, missing_global, pred_sum, pred_count,
                                  logger, profile=profile, label_data=label_data)
        summary.update({k: v for k, v in infer_stats.items() if k in ("dataset_traces", "dataset_missing", "prediction_keys")})
        summary["inference_seconds"] = round(inference_seconds, 3)
        summary["num_gpus"] = world_size
        summary["total_seconds"] = round(time.perf_counter() - total_start, 3)
        (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("done in %.2fs | output=%s", summary["total_seconds"], args.output_segy)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
