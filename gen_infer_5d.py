#!/usr/bin/env python3
"""FPM V3 inference CLI — queryctx mode.

Supports single-GPU and multi-GPU (via torchrun).

Usage:
    # Single GPU
    python infer_cli.py --checkpoint results/checkpoints/model-20.pth \
        --h5_irregular data/raw5d_data.h5 --h5_regular data/reg5dbin_label.h5 \
        --h5_mask data/reg5dbin_label_binning.h5 --mask_segy mask.sgy \
        --dataset_neighbors_infer data/infer_query_context.npz

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=2 infer_cli.py \
        --checkpoint ... --h5_irregular ... --dataset_neighbors_infer ...
"""

import argparse
import csv
import json
import logging
import os
import struct
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from dataset.dataset_reg import DatasetH5_all_queryctx, DatasetH5_all_queryctxV2
from models.seisdit_trace_axis import SeisDiTRopeV2
from FPM import FlowMatchingModel
from gen_infer_tool5d import run_queryctx_inference, add_prediction, fit_trace
from gen_infer import (
    read_segy_headers,
    read_segy_data,
    write_segy_data,
    write_segy_data_incremental,
    build_lookup,
    str2bool,
    none_or_str,
)
from segy_schema import get_segy_profile

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kw):
        return iterable


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("queryctx_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "infer.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def setup_ddp() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world_size


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool, logger: logging.Logger) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = (
        ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        if isinstance(ckpt, dict) else ckpt
    )
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    result = model.load_state_dict(state, strict=strict)
    logger.info("checkpoint loaded: missing=%s unexpected=%s",
                result.missing_keys, result.unexpected_keys)


def load_training_config(checkpoint_path: str) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    config_path = ckpt.parent.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    config_path = ckpt.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _first_of(*keys: str, cfg: dict, default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    ds = cfg.get("dataset", {})
    for k in keys:
        if k in ds:
            return ds[k]
    dsa = cfg.get("dataset_args", {})
    for k in keys:
        if k in dsa:
            return dsa[k]
    return default


def save_reports(output_dir: Path, headers, key_columns, written, unfilled, still_missing,
                 observed_changed, unmatched, summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
                writer.writerow([idx, *header_by_idx.get(int(idx), ("", "", "", ""))])
    with open(output_dir / "unmatched_prediction_keys.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(key_columns)
        writer.writerows(unmatched)


def _make_periodic_fill_callback(
    output_segy: str,
    mask_path: str,
    mask_data: np.ndarray,
    headers: list,
    missing_global: np.ndarray,
    time_ps: int,
    logger: logging.Logger,
):
    """Return a closure that checkpoints current predictions into *output_segy*.

    First invocation copies *mask_path* → *output_segy* (template).
    Subsequent ones open the existing file in ``r+`` and write only traces
    that are marked missing and have accumulated predictions.

    Signature ``(pred_sum, pred_count, flush_count)`` matches the
    ``flush_callback`` expected by ``run_queryctx_inference``.
    """
    import shutil as _shutil
    from pathlib import Path as _Path

    lookup = build_lookup(headers)
    out = mask_data.copy()
    ns = mask_data.shape[1]
    _initialized = False

    def _callback(pred_sum, pred_count, flush_count):
        nonlocal _initialized

        if not _initialized:
            _Path(output_segy).parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(mask_path, output_segy)
            _initialized = True

        seen: dict = {}
        for key, total in pred_sum.items():
            for trace_idx in lookup.get(key, []):
                if missing_global[trace_idx]:
                    avg = total / max(pred_count[key], 1)
                    trace = fit_trace(avg, ns, time_ps=time_ps)
                    out[trace_idx] = trace
                    seen[trace_idx] = trace

        if seen:
            write_segy_data_incremental(
                output_segy,
                np.array(list(seen.keys()), dtype=np.intp),
                np.array(list(seen.values()), dtype=np.float32),
            )
        logger.info("periodic fill [flush %d]: wrote %d traces to %s",
                     flush_count, len(seen), output_segy)

    return _callback


def fill_segy(args, headers, missing_global, pred_sum, pred_count, logger,
              key_columns, label_data=None, time_ps: int = None) -> dict:
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
        trace = fit_trace(total / max(pred_count[key], 1), ns, time_ps=time_ps)
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
        logger.info("residual SEGY: %s | max_abs=%.6g mean_abs=%.6g",
                     args.output_residual_segy,
                     residual_stats["residual_max_abs"],
                     residual_stats["residual_mean_abs"])

    summary = {
        "key_columns": list(key_columns),
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

    save_reports(Path(args.output_dir), headers, key_columns, written_sorted, unfilled,
                 still_missing, observed_changed, unmatched, summary)
    logger.info("writeback summary: %s", summary)

    if args.strict_fill and (unfilled or still_missing or observed_changed):
        raise RuntimeError(
            f"strict fill failed: unfilled={len(unfilled)} "
            f"still_missing={len(still_missing)} observed_changed={len(observed_changed)}"
        )
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPM V3 queryctx inference and SEGY fill")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5_irregular", required=True)
    parser.add_argument("--h5_regular", required=True)
    parser.add_argument("--h5_mask", required=True)
    parser.add_argument("--mask_path", required=True,
                        help="SEGY with missing traces (zero traces are replaced)")
    parser.add_argument("--label_segy", default=None,
                        help="Ground truth SEGY for residual (optional)")
    parser.add_argument("--dataset_neighbors_infer", required=True,
                        help="infer_query_context.npz path")

    parser.add_argument("--output_dir", default="gen_fill_results")
    parser.add_argument("--output_segy", default=None)
    parser.add_argument("--output_residual_segy", default=None)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--fill_interval", type=int, default=0,
                        help="Periodic SEGY checkpoint every N batch flushes "
                             "(0=disabled). Only rank 0 writes in DDP mode.")
    parser.add_argument("--time_ps", type=int, default=1256)
    parser.add_argument("--trace_ps", type=int, default=128)
    parser.add_argument("--missing_eps", type=float, default=1e-10)
    parser.add_argument("--header_mode", choices=["fixed", "self_computed"], default="fixed")
    parser.add_argument("--allow_coord_stats_fallback", type=str2bool, default=False,
                        help="Fallback to compute_coord_stats if coord_norm_stats.npz missing")

    parser.add_argument("--strict_load", action="store_false", dest="strict_load")
    parser.add_argument("--strict_fill", action="store_true", default=False)

    parser.add_argument("--use_missing_embedding", type=str2bool, default=False)
    parser.add_argument("--use_p_scale", type=str2bool, default=False)

    parser.add_argument("--model_type", default="trace_axis")

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

    parser.add_argument("--segy_config", type=str, default="field1031",
                        help="SEG-Y profile name (field1031, sw06, segc3)")
    parser.add_argument("--queryctx_variant", type=str, default="v1", choices=["v1", "v2"],
                        help="Dataset variant: v1=supervised, v2=self-supervised")
    parser.add_argument("--mlp_ratio", type=float, default=2.5,
                        help="MLP ratio for DiT blocks (must match training)")
    parser.add_argument("--num_bands", type=int, default=16,
                        help="Fourier frequency bands (must match training)")
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--vis_batches", type=int, default=0)

    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.output_segy = args.output_segy or str(Path(args.output_dir) / "filled_missing.sgy")
    args.output_residual_segy = args.output_residual_segy or str(Path(args.output_dir) / "residual.sgy")
    return args


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        logger = setup_logger(Path(args.output_dir)) if is_main else logging.getLogger("infer")
        if not is_main:
            logger.setLevel(logging.WARNING)
    else:
        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        logger = setup_logger(Path(args.output_dir))

    # Load profile from segy_schema
    profile = get_segy_profile(args.segy_config)
    key_columns = profile.key_columns

    if is_main:
        logger.info("args: %s", vars(args))
        logger.info("world_size=%d profile=%s key_columns=%s", world_size, profile.name, key_columns)

    # Compute shared temp directory for cross-rank result exchange.
    # Uses local storage (typically /tmp) rather than output_dir to avoid
    # NFS I/O pressure from 8 concurrent writers.
    import hashlib as _hashlib
    import tempfile as _tempfile
    _RANK_TMP = Path(_tempfile.gettempdir()) / (
        "infer_merge_" + _hashlib.md5(args.output_dir.encode()).hexdigest()[:12]
    )
    # Cleanup stale temp dir from previous crashed run (all ranks)
    if _RANK_TMP.exists():
        import shutil as _shutil
        if is_main:
            logger.info("cleaning up stale rank results: %s", _RANK_TMP)
        _shutil.rmtree(_RANK_TMP, ignore_errors=True)

    total_start = time.perf_counter()

    # Read SEGY (only rank 0 needs it for fill_segy)
    if is_main:
        mask_data = read_segy_data(args.mask_path)
        headers = read_segy_headers(args.mask_path, args.header_mode, profile)
        if len(headers) != mask_data.shape[0]:
            raise ValueError(f"header_count={len(headers)} != segy_traces={mask_data.shape[0]}")
        missing_global = np.all(np.abs(mask_data) <= args.missing_eps, axis=1)
        logger.info("template SEGY: traces=%d samples=%d missing=%d",
                     mask_data.shape[0], mask_data.shape[1], int(missing_global.sum()))

        label_data = None
        if args.label_segy:
            label_data = read_segy_data(args.label_segy)
            if label_data.shape != mask_data.shape:
                raise ValueError(f"label shape {label_data.shape} != mask shape {mask_data.shape}")
            logger.info("label SEGY loaded: %s shape=%s", args.label_segy, label_data.shape)
    else:
        mask_data = None
        headers = None
        missing_global = None
        label_data = None

    # Load training config from checkpoint for auto-detection
    train_cfg = load_training_config(args.checkpoint)
    if train_cfg and is_main:
        logger.info("loaded training config from checkpoint directory")

    # ---- Dataset ----
    use_p_scale = args.use_p_scale
    time_ps = args.time_ps
    if train_cfg:
        use_p_scale = bool(_first_of(
            "use_p_scale", cfg=train_cfg, default=use_p_scale
        ))
        tps = _first_of("time_ps", cfg=train_cfg)
        if tps is not None:
            time_ps = int(tps)

    # ---- Dataset (V1 or V2) ----
    variant = getattr(args, 'queryctx_variant', 'v1')
    if variant == 'v2':
        DatasetCls = DatasetH5_all_queryctxV2
    else:
        DatasetCls = DatasetH5_all_queryctx
    logger.info("building %s inference dataset", DatasetCls.__name__)
    dataset = DatasetCls(
        h5File=args.h5_irregular,
        h5File_regular=args.h5_regular,
        dataset_neighbors=args.dataset_neighbors_infer,
        train=False,
        key_columns=key_columns,
        use_p_scale=use_p_scale,
        time_ps=time_ps,
        trace_ps=args.trace_ps,
        allow_coord_stats_fallback=args.allow_coord_stats_fallback,
    )
    logger.info("queryctx dataset ready: samples=%d time_ps=%d", len(dataset), dataset.time_ps)

    # ---- Model (aligned with training) ----
    backbone = SeisDiTRopeV2(
            image_channels=2,
            n_channels=32,
            num_layers=8,
            d_model=384,
            nhead=8,
            mlp_ratio=4,
            num_bands=16,
            missing_focus_adapter=args.use_missing_embedding,
        ).to(device).eval()

    load_checkpoint(backbone, args.checkpoint, not args.strict_load, logger)
    if is_main:
        logger.info("model=SeisDiTRopeV2 params=%d device=%s",
                     sum(p.numel() for p in backbone.parameters()), device)

    fpm = FlowMatchingModel(
        model=backbone,
        trace_num=dataset.trace_ps,
        time_steps=dataset.time_ps,
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
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

    # ---- Periodic fill callback (optional) ----
    fill_callback = None
    if is_main and args.fill_interval > 0:
        fill_callback = _make_periodic_fill_callback(
            output_segy=args.output_segy,
            mask_path=args.mask_path,
            mask_data=mask_data,
            headers=headers,
            missing_global=missing_global,
            time_ps=time_ps,
            logger=logger,
        )

    # ---- Inference ----
    pred_sum, pred_count, inference_seconds, infer_stats = run_queryctx_inference(
        dataset=dataset,
        fpm=fpm,
        device=device,
        batch_size=args.batch_size,
        visualize=args.visualize,
        vis_dir=str(Path(args.output_dir) / "vis"),
        vis_max=args.vis_batches,
        progress=is_main,
        logger=logger,
        rank=rank,
        world_size=world_size,
        flush_callback=fill_callback,
        flush_interval=args.fill_interval,
    )

    # ---- DDP gather via file-based merge (avoids NCCL timeout) ----
    if world_size > 1:
        import pickle as _pickle
        import time as _time
        import shutil as _shutil

        _rank_dir = _RANK_TMP / f"rank_{rank}"
        _rank_dir.mkdir(parents=True, exist_ok=True)

        # Save pred_sum: store arrays in npz (keyed by index) + tuple keys via pickle.
        # Use np.savez (uncompressed) -- compression gains ~0 on per-trace arrays
        # and adds CPU overhead. Data is read immediately on the same machine.
        _r_keys = list(pred_sum.keys())
        with open(_rank_dir / "pred_keys.pkl", "wb") as _f:
            _pickle.dump(_r_keys, _f, protocol=_pickle.HIGHEST_PROTOCOL)
        _npz_dict = {f"arr_{i}": pred_sum[k] for i, k in enumerate(_r_keys)}
        np.savez(_rank_dir / "pred_sum.npz", **_npz_dict)
        # Save pred_count with stringified keys (JSON-compatible)
        _r_count = {"__".join(map(str, k)): int(v) for k, v in pred_count.items()}
        with open(_rank_dir / "pred_count.json", "w") as _f:
            json.dump(_r_count, _f)

        # File-based barrier: each rank signals completion by touching a done file
        (_RANK_TMP / f".rank_{rank}_done").touch()

        if is_main:
            _max_wait = 86400  # 24h timeout for file barrier
            logger.info("waiting for all ranks to finish inference (file barrier, timeout=%d h, tmp=%s)...",
                        _max_wait // 3600, _RANK_TMP)
            _waited = 0
            _pending = set(range(world_size))
            while _pending:
                _time.sleep(2)
                _waited += 2
                _pending = {r for r in _pending
                            if not (_RANK_TMP / f".rank_{r}_done").exists()}
                if _waited % 60 == 0:
                    logger.info("still waiting for ranks %s (%.0f s elapsed)...",
                                sorted(_pending), _waited)
                if _waited > _max_wait:
                    raise RuntimeError(
                        f"File barrier timed out after {_max_wait}s. "
                        f"Missing ranks: {sorted(_pending)}. "
                        f"The missing ranks may have crashed (OOM, segfault). "
                        f"Check the logs for errors on ranks {sorted(_pending)}."
                    )
            logger.info("all ranks done, merging %d result files...", world_size)

            merged_sum, merged_count = {}, defaultdict(int)
            for r in range(world_size):
                _rd = _RANK_TMP / f"rank_{r}"
                # Load pred_sum arrays + keys
                with open(_rd / "pred_keys.pkl", "rb") as _f:
                    _r_keys = _pickle.load(_f)
                with np.load(_rd / "pred_sum.npz") as _rd_npz:
                    for i, k in enumerate(_r_keys):
                        arr = _rd_npz[f"arr_{i}"]
                        if k in merged_sum:
                            merged_sum[k] += arr
                        else:
                            merged_sum[k] = arr.copy()
                # Load pred_count
                with open(_rd / "pred_count.json") as _f:
                    _r_counts = json.load(_f)
                for k_str, v in _r_counts.items():
                    kt = tuple(int(x) for x in k_str.split("__"))
                    merged_count[kt] += v

            pred_sum, pred_count = merged_sum, merged_count
            # Cleanup temp files
            _shutil.rmtree(_RANK_TMP)
            logger.info("merge complete: %d unique keys", len(pred_sum))

    # ---- SEGY fill ----
    if is_main:
        summary = fill_segy(args, headers, missing_global, pred_sum, pred_count,
                            logger, key_columns, label_data=label_data, time_ps=time_ps)
        # Merge infer_stats, but keep the global prediction_keys from fill_segy
        _global_keys = len(pred_sum)
        summary.update(infer_stats)
        summary["prediction_keys"] = int(_global_keys)
        summary["num_gpus"] = world_size
        summary["total_seconds"] = round(time.perf_counter() - total_start, 3)

        (Path(args.output_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("done in %.2fs | output=%s", summary["total_seconds"], args.output_segy)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()