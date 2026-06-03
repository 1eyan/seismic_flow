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
import hashlib
import json
import logging
import os
import pickle
import shutil
import struct
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataset import config as ds_config
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
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


def read_segy_data(path: str) -> np.ndarray:
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        return f.trace.raw[:].astype(np.float32)


def _read_segy_sort_keys(path: str, profile: SegyProfile) -> List[Tuple[int, ...]]:
    """Read sort key fields from SEGY headers using segyio."""
    byte_pos = profile.byte_pos
    keys = []
    with open(path, "rb") as f:
        f.seek(3200)
        bin_hdr = f.read(400)
        ns_bin = struct.unpack(">H", bin_hdr[20:22])[0]
        fmt = struct.unpack(">H", bin_hdr[24:26])[0]
        bps = 4 if fmt in (1, 2, 5) else 2
        f.seek(3600)
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            ns = struct.unpack(">H", hdr[114:116])[0] or ns_bin
            key = tuple(struct.unpack(">i", hdr[byte_pos[k] - 1 : byte_pos[k] + 3])[0]
                        for k in profile.sort_keys)
            keys.append(key)
            f.seek(ns * bps, 1)
    return keys


def write_segy_data(template_path: str, output_path: str, data: np.ndarray) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)
    with segyio.open(output_path, "r+", strict=False, ignore_geometry=True) as f:
        if f.tracecount != data.shape[0]:
            raise ValueError(f"SEGY tracecount={f.tracecount}, data traces={data.shape[0]}")
        for i in range(f.tracecount):
            f.trace[i] = data[i].astype(np.float32)


def write_sorted_segy(template_path: str, output_path: str, order: np.ndarray, data: np.ndarray) -> None:
    """Write a SEG-Y file with trace headers and trace samples sorted together."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)
    with segyio.open(template_path, "r", strict=False, ignore_geometry=True) as src, \
            segyio.open(output_path, "r+", strict=False, ignore_geometry=True) as dst:
        if src.tracecount != data.shape[0]:
            raise ValueError(f"SEGY tracecount={src.tracecount}, data traces={data.shape[0]}")
        for new_idx, old_idx in enumerate(order):
            old_idx = int(old_idx)
            dst.header[new_idx] = src.header[old_idx]
            dst.trace[new_idx] = data[old_idx].astype(np.float32)


def _read_segy_ns(path: str) -> int:
    """Read number of time samples from SEG-Y binary header (bytes 3220-3221)."""
    with open(path, "rb") as f:
        f.seek(3200)
        bin_hdr = f.read(400)
        return struct.unpack(">H", bin_hdr[20:22])[0]


def write_segy_data_incremental(output_path: str, trace_indices: np.ndarray, trace_data: np.ndarray) -> None:
    """Write specific traces to an existing SEG-Y file (r+ mode, no template copy)."""
    with segyio.open(output_path, "r+", strict=False, ignore_geometry=True) as f:
        for i in range(len(trace_indices)):
            f.trace[int(trace_indices[i])] = trace_data[i]


def fit_trace(trace: np.ndarray, ns: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    if trace.size > ns:
        return trace[trace.size - ns:]
    if trace.size < ns:
        return np.pad(trace, (ns - trace.size, 0)).astype(np.float32)
    return trace


def add_prediction(pred_sum, pred_count, key, trace) -> None:
    if key not in pred_sum:
        pred_sum[key] = np.zeros_like(trace, dtype=np.float64)
    pred_sum[key] += trace.astype(np.float64)
    pred_count[key] += 1


# ---------------------------------------------------------------------------
# SEG-Y header reading & geometry-key lookup
# ---------------------------------------------------------------------------

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
    """Parse SEG-Y trace headers and extract geometry keys.

    Returns a list of dicts, one per trace:
      {"trace_idx": int, "key": tuple, "coords": (sx, sy, rx, ry), "ns": int}
    """
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
            headers.append({
                "trace_idx": trace_idx,
                "key": tuple(int(x) for x in key),
                "coords": (sx, sy, rx, ry),
                "ns": int(ns_trace),
            })
            f.seek(int(ns_trace) * bps, os.SEEK_CUR)
            trace_idx += 1
    return headers


def build_lookup(headers: List[dict]) -> Dict[Tuple[int, ...], List[int]]:
    """Build {geometry_key: [trace_idx, ...]} lookup from parsed headers."""
    lookup: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for h in headers:
        lookup[h["key"]].append(int(h["trace_idx"]))
    return dict(lookup)


def partial_fill(args, headers, missing_global, pred_sum, pred_count, logger, profile, batch_idx: int) -> None:
    """Write current pred_sum/pred_count snapshot to output SEGY (intermediate result).

    Uses incremental write: first call copies template, subsequent calls only
    update traces that have predictions (avoids full-file rewrite).
    """
    ns = _read_segy_ns(args.mask_path)
    lookup = build_lookup(headers)
    seen_indices = {}
    for key, total in pred_sum.items():
        indices = lookup.get(key)
        if not indices:
            continue
        trace = fit_trace(total / max(pred_count[key], 1), ns)
        for idx in indices:
            if idx < len(missing_global) and missing_global[idx]:
                seen_indices[int(idx)] = trace.astype(np.float32)

    if not seen_indices:
        logger.info("partial_fill: batch=%d nothing to write", batch_idx)
        return

    output_path = args.output_segy
    if not Path(output_path).exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.mask_path, output_path)

    write_segy_data_incremental(
        output_path,
        np.array(list(seen_indices.keys()), dtype=np.intp),
        np.array(list(seen_indices.values()), dtype=np.float32),
    )
    logger.info("partial_fill: batch=%d pred_keys=%d traces_written=%d",
                batch_idx, len(pred_sum), len(seen_indices))


def collate_patches(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    stacked = {}
    for key in ('data', 'masked_patch', 'trace_mask',
                'sx_patch', 'sy_patch', 'rx_patch', 'ry_patch', 'key_values', 'trace_indices'):
        if key in batch[0]:
            stacked[key] = np.stack([b[key] for b in batch])
    stacked['amp_scale'] = np.array([float(b.get('amp_scale', 1.0)) for b in batch], dtype=np.float32)
    stacked['line_group_idx'] = np.array([int(b.get('line_group_idx', 0)) for b in batch], dtype=np.int64)
    return stacked


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_fk_spectrum(data: np.ndarray) -> np.ndarray:
    fk = np.fft.fft2(data)
    fk = np.fft.fftshift(fk)
    return 20 * np.log10(np.abs(fk) + 1e-10).T


def compute_snr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    missing = mask < 0.5
    gt_m = gt[missing]
    pred_m = pred[missing]
    noise = gt_m - pred_m
    signal_power = np.sum(gt_m ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power < 1e-30:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def visualize_batch(
    masked_patch: np.ndarray,
    pred: np.ndarray,
    data: np.ndarray,
    trace_mask: np.ndarray,
    sample_idx: int,
    vis_dir: Path,
) -> None:
    """Visualize masked input, prediction, ground truth, residual, FK spectra, and SNR."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    missing = trace_mask < 0.5
    residual = pred - data
    snr_val = compute_snr(data, pred, trace_mask)
    vmax = float(max(masked_patch.max(), pred.max(), data.max()))
    res_abs_max = float(max(abs(residual.min()), abs(residual.max()), 1e-6))
    n_traces = trace_mask.shape[0]
    n_missing = int(missing.sum())

    fk_mask = compute_fk_spectrum(masked_patch)
    fk_pred = compute_fk_spectrum(pred)
    fk_gt = compute_fk_spectrum(data)
    fk_vmax = max(fk_mask.max(), fk_pred.max(), fk_gt.max())
    fk_vmin = fk_vmax - 60

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
    for ax, img, title, vmn, vmx in zip(
        axes[0],
        [masked_patch, pred, data, residual],
        ["Masked Input", "FPM Prediction", "Ground Truth", "Residual (Pred - GT)"],
        [-vmax, -vmax, -vmax, -vmax],
        [vmax, vmax, vmax, vmax],
    ):
        ax.imshow(img.T, aspect="auto", cmap="seismic", vmin=vmn, vmax=vmx, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time Sample")

    for ax, img, title in zip(
        axes[1],
        [fk_mask, fk_pred, fk_gt],
        ["FK: Masked Input", "FK: Prediction", "FK: Ground Truth"],
    ):
        nf, nk = img.shape
        extent = [-0.5, 0.5, -0.5, 0.5]
        im = ax.imshow(img, aspect="auto", cmap="jet", vmin=fk_vmin, vmax=fk_vmax,
                       origin="lower", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("Wavenumber (cycle/sample)")
        ax.set_ylabel("Frequency (cycle/sample)")
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes[1, 3].axis("off")

    fig.suptitle(
        f"Sample {sample_idx} | {n_traces} traces | {n_missing} missing | SNR={snr_val:.2f} dB",
        fontsize=12,
    )
    fig.savefig(vis_dir / f"batch_{sample_idx:04d}.png", dpi=120)
    plt.close(fig)


def make_backbone(args: argparse.Namespace) -> torch.nn.Module:
    if "gated_encdec" in args.model_type:
        from models.gated_seisdit_gen_encdec import create_gated_seisdit_gen_encdec

        return create_gated_seisdit_gen_encdec(
            image_channels=2,
            d_model=args.gated_d_model,
            d_ff=args.gated_d_ff,
            nhead=args.gated_nhead,
            num_encoder_layers=args.gated_encoder_layers,
            num_bottleneck_layers=args.gated_bottleneck_layers,
            num_memory_bottleneck_layers=args.gated_encdec_mem_bottleneck_layers,
            num_decoder_layers=args.gated_decoder_layers,
            chunk_length=args.chunk_length_flow,
            chunk_overlap=args.chunk_overlap_flow,
            use_energy_stats=args.use_energy_mlp,
            use_missing_embed=args.use_missing_embedding,
            headwise_attn_output_gate=args.headwise_attn_output_gate,
            elementwise_attn_output_gate=args.elementwise_attn_output_gate,
            use_qk_norm=True,
            qkv_bias=False,
            use_relative_bias=True,
            use_encoder_relative_bias=False,
            use_decoder_self_relative_bias=False,
            use_cross_relative_bias=True,
            use_time_adaln=True,
            use_cross_query_gate=True,
            num_attn_res_blocks=2,
        )
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


def build_dataset(args, logger, profile: SegyProfile):
    from dataset.datasets_interp_v2 import DatasetH5_interp
    dataset = DatasetH5_interp(
        h5File_irregular=args.h5_mask,
        h5File_regular=args.h5_regular,
        train=False,
        missing_eps=args.h5_missing_eps,
        time_ps=args.time_ps,
        trace_ps=getattr(args, 'trace_ps', None),
        overlap_ratio=getattr(args, 'overlap_ratio', 0.5),
        use_p_scale=args.use_p_scale,
        profile=profile,
    )
    logger.info(
        "dataset ready: patches=%d time_ps=%d trace_ps=%d stride=%d overlap=%.2f",
        len(dataset), dataset.time_ps, dataset.trace_ps, dataset.stride, dataset.overlap_ratio,
    )
    return dataset


def infer_h5_dataset(args, fpm, device, logger, ns: int, profile: SegyProfile, dataset=None,
                     sampler=None, rank: int = 0, world_size: int = 1,
                     log_interval: int = 1,
                     headers=None, missing_global=None):
    if dataset is None:
        dataset = build_dataset(args, logger, profile)

    loader = DataLoader(
        dataset,
        batch_size=args.inference_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_patches,
    )

    pred_sum, pred_count = {}, defaultdict(int)
    # Incremental dicts: only accumulate predictions since the last backfill
    # This keeps all_gather_object small — just delta, not the entire accumulated dict.
    pred_sum_inc, pred_count_inc = {}, defaultdict(int)
    # Rank 0 only: persistent merged state across backfill intervals
    merged_full_sum = None
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
        current_bs = int(to_numpy(batch["amp_scale"]).shape[0])

        x_all = to_numpy(batch["masked_patch"]).astype(np.float32)
        coords_all = np.stack([
            to_numpy(batch["sx_patch"]),
            to_numpy(batch["sy_patch"]),
            to_numpy(batch["rx_patch"]),
            to_numpy(batch["ry_patch"]),
        ], axis=2).astype(np.float32)
        trace_mask_all = to_numpy(batch["trace_mask"]).astype(np.float32)
        scale_all = to_numpy(batch["amp_scale"]).astype(np.float32)
        data_all = to_numpy(batch["data"]).astype(np.float32)
        key_values_all = to_numpy(batch["key_values"]).astype(np.int64)

        x_for_model = x_all[:, None, :, :].astype(np.float32)
        coords_for_model = coords_all.astype(np.float32)

        pred_all = flow_sample(fpm, x_for_model, coords_for_model, device)

        for s in range(current_bs):
            trace_mask = trace_mask_all[s].astype(np.float32)
            scale = float(scale_all[s]) if scale_all.ndim > 0 else float(scale_all)
            pred = pred_all[s] * scale
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
                    key = tuple(int(v) for v in key_values_all[s, j])
                    trace = fit_trace(pred[j], ns)
                    add_prediction(pred_sum, pred_count, key, trace)
                    add_prediction(pred_sum_inc, pred_count_inc, key, trace)

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

        # Periodic partial backfill: gather only incremental data, rank 0 writes merged result
        if (args.backfill_interval > 0 and headers is not None
                and missing_global is not None
                and (batch_idx + 1) % args.backfill_interval == 0):
            if world_size > 1:
                dist.barrier()
                # all_gather_object only the incremental dicts (small — just delta since last backfill)
                gathered = [None] * world_size
                dist.all_gather_object(gathered, (dict(pred_sum_inc), dict(pred_count_inc)))
                if rank == 0:
                    if merged_full_sum is None:
                        merged_full_sum, merged_full_count = {}, defaultdict(int)
                    for ps, pc in gathered:
                        for k, v in ps.items():
                            if k not in merged_full_sum:
                                merged_full_sum[k] = v.copy()
                            else:
                                merged_full_sum[k] += v
                        for k, v in pc.items():
                            merged_full_count[k] += v
                    partial_fill(args, headers, missing_global, merged_full_sum, merged_full_count,
                                 logger, profile, batch_idx)
                dist.barrier()
            else:
                merged_full_sum = pred_sum
                merged_full_count = pred_count
                partial_fill(args, headers, missing_global, merged_full_sum, merged_full_count,
                             logger, profile, batch_idx)
            # Clear incremental dicts on all ranks
            pred_sum_inc.clear()
            pred_count_inc.clear()

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time
    logger.info(
        "inference finished: %.2fs | patches=%d traces=%d missing=%d prediction_keys=%d",
        seconds,
        len(dataset),
        total_traces,
        total_missing,
        len(pred_sum),
    )
    return pred_sum, pred_count, seconds, {
        "dataset_samples": int(global_sample_idx),
        "dataset_traces": int(total_traces),
        "dataset_missing": int(total_missing),
        "prediction_keys": int(len(pred_sum)),
    }


def fill_and_verify(args, headers, missing_global, pred_sum, pred_count, logger,
                    profile: SegyProfile, label_data=None) -> dict:
    mask_data = read_segy_data(args.mask_path)
    out = mask_data.copy()
    ns = mask_data.shape[1]
    lookup = build_lookup(headers)
    written = set()
    unmatched_indices = []  # trace indices where prediction key maps to non-missing trace
    unmatched_keys = []     # prediction keys that matched no SEGY trace

    for key, total in pred_sum.items():
        indices = lookup.get(key)
        if not indices:
            unmatched_keys.append(key)
            continue
        trace = fit_trace(total / max(pred_count[key], 1), ns)
        for idx in indices:
            if not missing_global[idx]:
                unmatched_indices.append(idx)
                continue
            out[idx] = trace
            written.add(idx)

    write_start = time.perf_counter()
    write_segy_data(args.mask_path, args.output_segy, out)
    writeback_seconds = time.perf_counter() - write_start

    written_sorted = sorted(written)
    missing_indices = set(np.flatnonzero(missing_global).tolist())
    unfilled = sorted(missing_indices - written)
    # In-memory validation (avoids disk readback)
    still_missing = np.flatnonzero(
        missing_global & np.all(np.abs(out) <= args.missing_eps, axis=1)
    ).tolist()
    observed_changed = np.flatnonzero(
        (~missing_global) & np.any(np.abs(out - mask_data) > args.missing_eps, axis=1)
    ).tolist()

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
        "unmatched_prediction_keys": int(len(unmatched_indices)),
        "unmatched_geometry_keys": int(len(unmatched_keys)),
        "writeback_seconds": round(writeback_seconds, 3),
        "output_segy": args.output_segy,
        **residual_stats,
    }
    if args.sort_output:
        out_path = Path(args.output_segy)
        sorted_path = str(out_path.with_name(f"{out_path.stem}_sorted{out_path.suffix}"))
        sort_keys = np.asarray(_read_segy_sort_keys(args.output_segy, profile), dtype=np.int64)
        if sort_keys.ndim == 1:
            order = np.argsort(sort_keys)
        else:
            order = np.lexsort(sort_keys.T[::-1])
        write_sorted_segy(args.output_segy, sorted_path, order, out)
        summary["output_segy_sorted"] = sorted_path
        logger.info("sorted SEGY written: %s", sorted_path)

    save_reports(Path(args.output_dir), written_sorted, unfilled, still_missing,
                 observed_changed, unmatched_indices, summary)
    logger.info("writeback summary: %s", summary)
    if args.strict_fill and (unfilled or still_missing or observed_changed):
        raise RuntimeError(
            "strict fill failed: "
            f"unfilled={len(unfilled)} still_missing={len(still_missing)} observed_changed={len(observed_changed)}"
        )
    return summary


def save_reports(output_dir: Path, written, unfilled, still_missing,
                 observed_changed, unmatched, summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, indices in (
        ("filled_missing_traces.csv", written),
        ("unfilled_missing_traces.csv", unfilled),
        ("still_missing_after_write_traces.csv", still_missing),
        ("observed_changed_traces.csv", observed_changed),
    ):
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trace_idx"])
            for idx in indices:
                writer.writerow([idx])
    with open(output_dir / "unmatched_prediction_traces.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trace_idx"])
        for idx in unmatched:
            writer.writerow([idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPM V3 H5 inference and fill missing traces into SEGY")
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--inference_batch_size", type=int, default=1, help="Patches per forward pass (reduce if OOM)")
    parser.add_argument("--time_ps", type=int, default=None)
    parser.add_argument("--trace_ps", type=int, default=None,
                        help="Patch trace count (default from data_config.py: 128)")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                        help="Sliding window overlap ratio for patch-based inference")
    parser.add_argument("--missing_eps", type=float, default=1e-10)
    parser.add_argument("--h5_missing_eps", type=float, default=None, help="trace is missing in h5_mask if all abs(data) <= this; defaults to missing_eps")
    parser.add_argument("--non_strict_load", dest="strict_load", action="store_false")
    parser.add_argument("--strict_fill", dest="strict_fill", action="store_true")
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.add_argument("--sort_output", type=str2bool, default=False,
                        help="Sort output SEGY traces by profile.sort_keys")
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
    parser.add_argument("--gated_encdec_mem_bottleneck_layers", type=int, default=2, help="memory bottleneck layers for gated_encdec model")

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

    parser.add_argument("--backfill_interval", type=int, default=0,
                        help="Partial SEGY write every N batches during inference (0 = only at end)")
    parser.add_argument("--header_mode", choices=["fixed", "self_computed"], default=None,
                        help="SEG-Y header mode (default from profile)")

    args = parser.parse_args()

    profile = get_segy_profile(args.segy_profile)
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
            "data path: local H5 DatasetH5_interp-compatible | segy_profile=%s | key=%s",
            profile.name,
            profile.key_columns,
        )

    total_start = time.perf_counter()
    mask_data = read_segy_data(args.mask_path)
    if len(mask_data) == 0:
        raise ValueError(f"SEGY file has no traces: {args.mask_path}")
    missing_global = np.all(np.abs(mask_data) <= args.missing_eps, axis=1)
    if rank == 0:
        logger.info(
            "template SEGY: traces=%d samples=%d missing=%d",
            mask_data.shape[0],
            mask_data.shape[1],
            int(missing_global.sum()),
        )

    # Read SEG-Y trace headers for geometry-key-based matching
    headers = read_segy_headers(args.mask_path, args.header_mode, profile)
    if rank == 0:
        n_keys = len({tuple(h["key"]) for h in headers})
        logger.info("SEG-Y headers parsed: traces=%d unique_geometry_keys=%d mode=%s",
                     len(headers), n_keys, args.header_mode)

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
        trace_num=args.trace_ps,
        time_steps=args.time_ps,
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
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if rank == 0:
            logger.info("DDP: world_size=%d rank=%d local_rank=%d", world_size, rank, local_rank)
    else:
        sampler = None

    log_interval = max(1, (len(dataset) // world_size) // 10) if world_size > 1 else max(1, len(dataset) // 10)
    pred_sum, pred_count, inference_seconds, infer_stats = infer_h5_dataset(
        args, fpm, device, logger, ns=mask_data.shape[1], profile=profile, dataset=dataset,
        sampler=sampler, rank=rank, world_size=world_size,
        log_interval=log_interval,
        headers=headers, missing_global=missing_global,
    )

    # ---- File-based DDP merge (avoids NCCL all_gather_object timeout) ----
    if world_size > 1:
        dist.barrier()  # lightweight sync — all ranks finished inference
        _RANK_TMP = Path(tempfile.gettempdir()) / (
            "infer_merge_" + hashlib.md5(args.output_dir.encode()).hexdigest()[:12]
        )
        _rank_dir = _RANK_TMP / f"rank_{rank}"
        _rank_dir.mkdir(parents=True, exist_ok=True)

        # Save pred_sum: keys via pickle, arrays via npz
        _r_keys = list(pred_sum.keys())
        with open(_rank_dir / "pred_keys.pkl", "wb") as _f:
            pickle.dump(_r_keys, _f, protocol=pickle.HIGHEST_PROTOCOL)
        _npz_dict = {f"arr_{i}": pred_sum[k] for i, k in enumerate(_r_keys)}
        np.savez(_rank_dir / "pred_sum.npz", **_npz_dict)

        # Save pred_count: stringify tuple keys with "__" separator
        with open(_rank_dir / "pred_count.json", "w") as _f:
            json.dump({"__".join(map(str, k)): int(v) for k, v in pred_count.items()}, _f)

        # Save infer_stats
        with open(_rank_dir / "infer_stats.json", "w") as _f:
            json.dump(infer_stats, _f)

        # File-based barrier: signal completion
        (_RANK_TMP / f".rank_{rank}_done").touch()

        if rank == 0:
            _max_wait = 86400  # 24h
            logger.info("file barrier: waiting for %d ranks (tmp=%s)...", world_size, _RANK_TMP)
            _waited = 0
            _pending = set(range(world_size))
            while _pending:
                time.sleep(2)
                _waited += 2
                _pending = {r for r in _pending
                            if not (_RANK_TMP / f".rank_{r}_done").exists()}
                if _waited % 60 == 0:
                    logger.info("still waiting for ranks %s (%ds)...", sorted(_pending), _waited)
                if _waited > _max_wait:
                    raise RuntimeError(
                        f"File barrier timed out after {_max_wait}s. "
                        f"Missing ranks: {sorted(_pending)}."
                    )

            logger.info("all ranks done, merging %d result files...", world_size)
            merged_sum, merged_count = {}, defaultdict(int)
            merged_stats = {"dataset_samples": 0, "dataset_traces": 0,
                            "dataset_missing": 0, "prediction_keys": 0}
            for r in range(world_size):
                _rd = _RANK_TMP / f"rank_{r}"
                with open(_rd / "pred_keys.pkl", "rb") as _f:
                    _r_keys = pickle.load(_f)
                with np.load(_rd / "pred_sum.npz") as _rd_npz:
                    for i, k in enumerate(_r_keys):
                        arr = _rd_npz[f"arr_{i}"]
                        if k in merged_sum:
                            merged_sum[k] += arr
                        else:
                            merged_sum[k] = arr.copy()
                with open(_rd / "pred_count.json") as _f:
                    for k_str, v in json.load(_f).items():
                        merged_count[tuple(int(x) for x in k_str.split("__"))] += v
                with open(_rd / "infer_stats.json") as _f:
                    for sk, sv in json.load(_f).items():
                        merged_stats[sk] = merged_stats.get(sk, 0) + sv

            pred_sum, pred_count = merged_sum, merged_count
            merged_stats["prediction_keys"] = len(merged_sum)
            infer_stats = merged_stats
            # Cleanup temp files (rank 0 only, after merge completes)
            shutil.rmtree(_RANK_TMP, ignore_errors=True)
            logger.info("merge complete: %d unique keys", len(pred_sum))
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
