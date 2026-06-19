"""Query-Context inference engine (self-contained).

Core inference loop for DatasetH5_all_queryctx. Works with any FlowMatchingModel
that implements ``sample(condL, x_cond, time_axis)``.

Usage:
    from queryctx_module import DatasetH5_all_queryctx, build_coord_config
    from queryctx_module.infer import run_queryctx_inference, flow_sample, add_prediction

    dataset = DatasetH5_all_queryctx(
        h5File="irregular.h5",
        h5File_regular="regular.h5",
        dataset_neighbors="infer_query_context.npz",
        train=False,
        use_p_scale=True,
        time_ps=1256,
        trace_ps=128,
    )

    # fpm is your loaded FlowMatchingModel in eval mode
    pred_sum, pred_count, metadata = run_queryctx_inference(
        dataset=dataset,
        fpm=fpm,
        device=torch.device("cuda:0"),
        batch_size=4,
        progress=True,
    )

    # pred_sum: dict[(shot_line, shot_stake, recv_line, recv_stake) -> np.ndarray]
    # pred_count: dict[key -> int] number of predictions per key
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def add_prediction(pred_sum, pred_count, key, trace):
    """Accumulate a prediction trace into the running per-key sum.

    Args:
        pred_sum: dict[key -> np.ndarray] mutable accumulator
        pred_count: dict[key -> int] mutable counter
        key: trace identity, e.g. (shot_line, shot_stake, recv_line, recv_stake)
        trace: (T,) float32 array
    """
    # float64 for stable accumulation; asarray already copies (float32->float64)
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if key not in pred_sum:
        pred_sum[key] = trace
    else:
        pred_sum[key] += trace
    pred_count[key] += 1


def fit_trace(trace: np.ndarray, ns: int, time_ps: Optional[int] = None) -> np.ndarray:
    """Resize a trace to exactly *ns* samples, inverting ``_crop_or_pad_time``.

    When *time_ps* is provided, the trace is assumed to be *time_ps* samples long
    and this function recovers the original *ns*-sample layout by undoing the
    dataset time transformation:

    - ``ns > time_ps``: dataset kept the **last** *time_ps* samples
      (crop from left).  Inverse: prepend zeros on the left.
    - ``ns < time_ps``: dataset **left-padded** with zeros.
      Inverse: keep the **last** *ns* samples.

    When *time_ps* is ``None`` (legacy / simple resize), trims from the left
    or right-pads.
    """
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)

    if time_ps is not None and trace.size != ns:
        if ns > time_ps:
            # dataset: traces[:, diff:]  →  dropped first diff, kept last
            return np.pad(trace, (ns - time_ps, 0), constant_values=0).astype(np.float32)
        elif ns < time_ps:
            # dataset: left-padded (time_ps - ns) zeros at beginning
            return trace[time_ps - ns:].astype(np.float32)
        return trace.astype(np.float32)

    # Simple length-matching (legacy path)
    if trace.size > ns:
        return trace[:ns]
    if trace.size < ns:
        return np.pad(trace, (0, ns - trace.size), constant_values=0).astype(np.float32)
    return trace


# ---------------------------------------------------------------------------
# Flow-matching forward
# ---------------------------------------------------------------------------


def flow_sample(fpm, x_norm: np.ndarray, coords: np.ndarray, device) -> np.ndarray:
    """Run flow-matching inference on a batch.

    Args:
        fpm: FlowMatchingModel (must have ``sample`` method)
        x_norm: (B, max_tr, T) normalized masked input,
                or (B, 1, max_tr, T), or (1, max_tr, T), or (max_tr, T).
        coords: (B, max_tr, 4) normalized coords [sx, sy, rx, ry].
        device: torch device.

    Returns:
        pred: (B, max_tr, T) float32 numpy array (normalized).
    """
    import torch

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


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------


def run_queryctx_inference(
    dataset,
    fpm,
    device,
    batch_size: int = 1,
    visualize: bool = False,
    vis_dir: Optional[str] = None,
    vis_max: int = 0,
    progress: bool = True,
    logger=None,
    rank: int = 0,
    world_size: int = 1,
    flush_callback=None,
    flush_interval: int = 0,
) -> Tuple[Dict, Dict, Dict[str, Any]]:
    """Run inference on a DatasetH5_all_queryctx in inference mode.

    The dataset must be created with ``train=False`` and a
    ``dataset_neighbors`` npz containing ``infer_query_context`` keys.

    Args:
        dataset: DatasetH5_all_queryctx (train=False).
        fpm: FlowMatchingModel loaded with checkpoint weights, in eval mode.
        device: torch device.
        batch_size: number of samples per forward pass (>1 packs unequal-size
            patches via zero-padding to max-traces within the batch).
        visualize: if True, save per-sample visualization PNGs.
        vis_dir: output directory for visualization.
        vis_max: max samples to visualize (0 = all).
        progress: show tqdm progress bar.
        logger: optional logging.Logger.

    Returns:
        (pred_sum, pred_count, metadata)

        ``pred_sum``: dict[(shot_line, shot_stake, recv_line, recv_stake) -> np.ndarray]
            Accumulated prediction trace arrays.
        ``pred_count``: dict[key -> int]
            Number of predictions accumulated per key.
        ``metadata``: dict with statistics (dataset_samples, dataset_traces,
            dataset_missing, prediction_keys, inference_seconds).
    """
    import torch

    pred_sum: Dict = {}
    pred_count = defaultdict(int)
    total_missing = 0
    total_traces = 0
    is_main = rank == 0

    vis_path = None
    if vis_dir is not None:
        vis_path = Path(vis_dir)
        vis_path.mkdir(parents=True, exist_ok=True)
    vis_limit = vis_max if vis_max > 0 else float("inf")
    _flush_count = 0

    all_indices = list(range(len(dataset)))
    # DDP: each rank processes only its shard
    if world_size > 1:
        all_indices = all_indices[rank::world_size]
    batch_size = max(1, int(batch_size))

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    sample_buf: list = []
    total_samples = len(all_indices)
    desc = f"queryctx inference [rank {rank}]" if world_size > 1 else "queryctx inference"
    iterator = tqdm(
        all_indices,
        desc=desc,
        unit="sample",
        disable=not (progress and is_main),
        smoothing=0.01,
    )

    

    # ------------------------------------------------------------------
    # Batch builder + flusher
    # ------------------------------------------------------------------

    def _build_batch(samples: list):
        """Pack variable-size dataset items into a padded batch.

        Returns: (x_batch, c_batch, scales, valid, meta_list)
        """
        B = len(samples)
        max_tr = max(s[0]["data"].shape[0] for s in samples)
        T = samples[0][0]["data"].shape[1]

        x_batch = np.zeros((B, max_tr, T), dtype=np.float32)
        c_batch = np.zeros((B, max_tr, 4), dtype=np.float32)
        valid = np.zeros((B, max_tr), dtype=np.float32)
        scales = np.zeros(B, dtype=np.float32)
        meta_list = []

        for b in range(B):
            s = samples[b][0]
            n_tr = int(s["data"].shape[0])
            valid[b, :n_tr] = 1.0
            x_batch[b, :n_tr] = np.asarray(s["masked_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 0] = np.asarray(s["sx_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 1] = np.asarray(s["sy_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 2] = np.asarray(s["rx_patch"], dtype=np.float32)
            c_batch[b, :n_tr, 3] = np.asarray(s["ry_patch"], dtype=np.float32)
            scales[b] = float(s["amp_scale"])
            meta_list.append(
                {
                    "n_tr": n_tr,
                    "is_query": np.asarray(s["is_query"], dtype=bool),
                    "trace_obs": (~np.asarray(s["is_query"], dtype=bool)).astype(
                        np.float32
                    ),
                    "masked_patch_raw": np.asarray(
                        s.get("masked_patch_raw", s["masked_patch"]), dtype=np.float32
                    ),
                    "data_raw": np.asarray(s.get("data_raw", s["data"]), dtype=np.float32),
                    "patch_info": s.get("patch_info", {}),
                    "sample_idx": samples[b][1],
                }
            )

        return x_batch, c_batch, scales, valid, meta_list

    def _flush():
        nonlocal total_missing, total_traces, pred_sum, pred_count, _flush_count

        if not sample_buf:
            return

        x_batch, c_batch, scales, valid, meta_list = _build_batch(sample_buf)
        B = x_batch.shape[0]
        _, max_tr, T = x_batch.shape
        # Update model shape info (required by some FlowMatchingModel impls)
        #if hasattr(fpm.model, "trace_num"):
        fpm.trace_num = max_tr
        fpm.time_steps = T
        fpm.sample_num = B

        pred = flow_sample(fpm, x_batch, c_batch, device)
        pred = pred * scales[:, None, None]

        for b in range(B):
            m = meta_list[b]
            n_tr = m["n_tr"]
            is_query = m["is_query"]
            trace_obs = m["trace_obs"]
            pred_b = pred[b, :n_tr]
            masked_raw_b = m["masked_patch_raw"][:n_tr]
            data_raw_b = m["data_raw"][:n_tr]

            missing_count = int(is_query.sum())
            total_missing += missing_count
            total_traces += n_tr

            if visualize and is_main and m["sample_idx"] < vis_limit:
                _visualize_sample(
                    masked_raw_b,
                    pred_b,
                    data_raw_b,
                    trace_obs,
                    int(m["sample_idx"]),
                    vis_path,
                )

            pi = m["patch_info"]
            # Extract key arrays once per sample (from dataset.key_columns)
            _key_cols = dataset.key_columns
            key_arrays = {col: pi.get(col) for col in _key_cols}
            for j in range(n_tr):
                if is_query[j]:
                    key = tuple(
                        int(key_arrays[col][j]) if key_arrays[col] is not None else 0
                        for col in _key_cols
                    )
                    add_prediction(pred_sum, pred_count, key, pred_b[j])

        sample_buf.clear()
        _flush_count += 1
        if (
            flush_callback is not None
            and flush_interval > 0
            and _flush_count % flush_interval == 0
        ):
            flush_callback(pred_sum, pred_count, _flush_count)


    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    for idx in iterator:
        sample = dataset[idx]
        sample_buf.append((sample, idx))
        if len(sample_buf) >= batch_size:
            _flush()
        if progress and is_main:
            n_tr = sample["data"].shape[0]
            is_query = np.asarray(sample["is_query"], dtype=bool)
            iterator.set_postfix(sample=idx, traces=n_tr, missing=int(is_query.sum()))

    _flush()  # remaining

    if flush_callback is not None and flush_interval > 0:
        flush_callback(pred_sum, pred_count, _flush_count)

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time

    if logger is not None:
        logger.info(
            "queryctx inference done: %.2fs | samples=%d traces=%d missing=%d keys=%d",
            seconds, total_samples, total_traces, total_missing, len(pred_sum),
        )

    return pred_sum, pred_count, seconds, {
        "dataset_samples": int(len(dataset)),
        "dataset_traces": int(total_traces),
        "dataset_missing": int(total_missing),
        "prediction_keys": int(len(pred_sum)),
    }


# ---------------------------------------------------------------------------
# Internal visualization
# ---------------------------------------------------------------------------


def _visualize_sample(
    masked_patch: np.ndarray,
    pred: np.ndarray,
    data: np.ndarray,
    trace_obs: np.ndarray,
    sample_idx: int,
    vis_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = pred - data
    vmax = float(
        max(
            abs(masked_patch.min()),
            abs(masked_patch.max()),
            abs(pred.min()),
            abs(pred.max()),
            abs(data.min()),
            abs(data.max()),
            abs(residual.min()),
            abs(residual.max()),
            1e-6,
        )
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    im = None
    for ax, img, title in zip(
        axes,
        [masked_patch, pred, data, residual],
        ["Masked Input", "FPM Prediction", "Ground Truth", "Residual (Pred - GT)"],
    ):
        im = ax.imshow(
            img.T, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, origin="upper"
        )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time Sample")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    n_missing = int((trace_obs < 0.5).sum())
    fig.suptitle(
        f"Sample {sample_idx} | {trace_obs.shape[0]} traces | {n_missing} missing",
        fontsize=12,
    )
    fig.savefig(vis_dir / f"sample_{sample_idx:04d}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: load training config for model setup
# ---------------------------------------------------------------------------


def load_training_config(checkpoint_path: str) -> Dict[str, Any]:
    """Load training_config.json from the checkpoint's results folder.

    Expected layout::

        resultsFPM/xxx/
        ├── checkpoints/
        │   └── model-N.pth   ← checkpoint_path
        └── logs/
            └── training_config.json

    Returns empty dict if not found.
    """
    ckpt = Path(checkpoint_path)
    # New layout: checkpoints/model-N.pth → results/logs/training_config.json
    config_path = ckpt.parent.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Old layout: checkpoint directly in results/
    config_path = ckpt.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


import json  # noqa: E402 (module-level import at bottom for clarity)