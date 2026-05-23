import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .config import args
    from .datasets_interp import DatasetH5_interp, apply_random_missing
except ImportError:
    from config import args
    from datasets_interp import DatasetH5_interp, apply_random_missing

try:
    from ..generate_py.ovt_masking import build_support_index, dispatch_ovt_mask
except ImportError:
    import sys
    from pathlib import Path
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from generate_py.ovt_masking import build_support_index, dispatch_ovt_mask


def _load_optional_json(json_path: Optional[str]) -> Optional[Any]:
    if json_path is None:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_ovt_train_mixture() -> Sequence[Dict[str, Any]]:
    return [
        {
            "type": "random_bin",
            "prob": 0.35,
            "params": {
                "missing_ratio": 0.35,
                "scope": {"type": "global"},
            },
        },
        {
            "type": "azimuth_sector",
            "prob": 0.15,
            "params": {
                "scope": {"type": "local", "patch_ratio": 0.55},
                "sectors": [{"start": -180.0, "width": 180.0}],
                "angle_unit": "degree",
                "reciprocal_pair": True,
            },
        },
        {
            "type": "offset_truncation",
            "prob": 0.30,
            "params": {
                "scope": {"type": "local", "patch_ratio": 0.25},
                "truncation_mode": "remove_far",
                "far_quantile": 0.8,
                "quantile_scope": "per_midpoint_cell",
            },
        },
        {
            "type": "midpoint_block",
            "prob": 0.20,
            "params": {
                "width": 8,
                "height": 8,
                "scope": {"type": "global"},
            },
        },
    ]


class DatasetH5_ovt_interp(DatasetH5_interp):
    """
    基于 DatasetH5_interp 的 OVT 在线 masking 版本。

    - 训练：在每个 patch 内根据 OVT 几何做在线 mixed masking
    - 测试：沿用 DatasetH5_interp 的现有逻辑
    """

    def __init__(
        self,
        h5File_irregular: str,
        h5File_regular: Optional[str] = None,
        train_idx_np: Optional[str] = None,
        train: bool = True,
        survey_line_key: str = "recv_line",
        missing_ratio_range: Tuple[float, float] = (0.4, 0.7),
        ovt_mask_mode: str = args.ovt_mask_mode,
        ovt_default_mode: str = args.ovt_mask_default_mode,
        ovt_config: Optional[Dict[str, Any]] = None,
        ovt_mixture: Optional[Sequence[Dict[str, Any]]] = None,
        ovt_mixture_json: Optional[str] = args.ovt_mask_mixture_json,
        ovt_seed: int = args.ovt_mask_seed,
        ovt_min_keep_cells: int = args.ovt_mask_min_keep_cells,
        fallback_to_random_missing: bool = args.ovt_mask_fallback_random,
    ):
        super().__init__(
            h5File_irregular=h5File_irregular,
            h5File_regular=h5File_regular,
            train_idx_np=train_idx_np,
            train=train,
            survey_line_key=survey_line_key,
            missing_ratio_range=missing_ratio_range,
        )
        self.ovt_mask_mode = ovt_mask_mode
        self.ovt_default_mode = ovt_default_mode
        self.ovt_seed = int(ovt_seed)
        self.ovt_min_keep_cells = int(max(1, ovt_min_keep_cells))
        self.fallback_to_random_missing = bool(fallback_to_random_missing)
        self.ovt_config = dict(ovt_config or {})
        self.ovt_mixture = list(ovt_mixture) if ovt_mixture is not None else None
        if self.ovt_mixture is None:
            self.ovt_mixture = _load_optional_json(ovt_mixture_json)
        if self.ovt_mask_mode == "train" and self.ovt_mixture is None:
            self.ovt_mixture = _default_ovt_train_mixture()

    def _build_patch_geometry_table(self, selected: np.ndarray) -> Dict[str, np.ndarray]:
        h5 = self.h5_data
        table: Dict[str, np.ndarray] = {
            "trace_index": np.arange(len(selected), dtype=np.int64),
            "global_trace_index": selected.astype(np.int64),
            "sx": h5["sx"][selected],
            "sy": h5["sy"][selected],
            "rx": h5["rx"][selected],
            "ry": h5["ry"][selected],
        }

        optional_keys = [
            "mx", "my", "hx", "hy",
            "imx", "imy", "ihx", "ihy",
            "mx_center", "my_center", "hx_center", "hy_center",
            "fold", "offset_mag", "azimuth",
        ]
        for key in optional_keys:
            if key in h5:
                table[key] = h5[key][selected]
        return table

    def _sort_patch_by_ovt(self, selected: np.ndarray):
        patch_table = pd.DataFrame(self._build_patch_geometry_table(selected))
        support = build_support_index(
            patch_table,
            source_type="dataframe",
        )
        trace_df = support["trace_df"].sort_values(
            by=["imx", "imy", "ihx", "ihy", "trace_index"]
        ).reset_index(drop=True)
        selected_sorted = trace_df["global_trace_index"].to_numpy(dtype=np.int64)
        return selected_sorted, trace_df

    def _apply_online_ovt_mask(self, data_full: np.ndarray, patch_table: pd.DataFrame, seed: int):
        result = dispatch_ovt_mask(
            patch_table,
            source_type="dataframe",
            mode=self.ovt_default_mode,
            mask_mode=self.ovt_mask_mode,
            config=self.ovt_config,
            mixture=self.ovt_mixture,
            seed=seed,
            min_keep_cells=self.ovt_min_keep_cells,
        )

        keep_local = np.zeros(len(patch_table), dtype=bool)
        keep_local[result["kept_trace_indices"]] = True
        if not np.any(keep_local) and self.fallback_to_random_missing:
            missing_ratio = np.random.uniform(*self.missing_ratio_range)
            return (*apply_random_missing(data_full, missing_ratio), None, None)

        trace_mask = keep_local.astype(np.float32)[:, None]
        mask = np.repeat(trace_mask, data_full.shape[1], axis=1)
        masked_patch = data_full * mask
        trace_mask_table = result["trace_mask_table"]
        trace_mask_table = trace_mask_table.sort_values("trace_index").reset_index(drop=True)
        return masked_patch, mask.astype(np.float32), result, trace_mask_table

    def _get_train_item(self, idx: int) -> Dict[str, Any]:
        np.random.seed(idx)

        if self.kept_indices is not None:
            all_indices = self.kept_indices
        else:
            all_indices = np.arange(len(self.h5_data["data"]))

        n_total = len(all_indices)
        if n_total <= self.trace_ps:
            selected = all_indices
        else:
            start = np.random.randint(0, n_total - self.trace_ps + 1)
            selected = all_indices[start:start + self.trace_ps]

        selected, patch_trace_df = self._sort_patch_by_ovt(selected)

        data_full = self.h5_data["data"][selected]
        rx_full = self.h5_data["rx"][selected]
        ry_full = self.h5_data["ry"][selected]
        sx_full = self.h5_data["sx"][selected]
        sy_full = self.h5_data["sy"][selected]
        data_full = self._crop_or_pad_time(data_full)

        ovt_seed = self.ovt_seed + int(idx)
        masked_patch, mask, ovt_result, trace_mask_table = self._apply_online_ovt_mask(
            data_full,
            patch_table=patch_trace_df,
            seed=ovt_seed,
        )

        obs = masked_patch[mask > 0]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else 1e-2
        std_val = max(std_val, 1e-2)

        thres = np.percentile(np.abs(masked_patch), 99.5) if obs.size > 0 else 1e-6
        thres = max(thres, 1e-6)
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_full, -thres, thres) / thres

        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_full, sy_full, rx_full, ry_full)
        time_axis_2d = self._time_axis_2d(len(selected))

        sample = {
            "data": data_patch.astype(np.float32),
            "masked_patch": masked_patch.astype(np.float32),
            "mask": mask.astype(np.float32),
            "rx_patch": rx_n.astype(np.float32),
            "ry_patch": ry_n.astype(np.float32),
            "sx_patch": sx_n.astype(np.float32),
            "sy_patch": sy_n.astype(np.float32),
            "time_axis_2d": time_axis_2d.astype(np.float32),
            "std_val": std_val,
            "trace_indices": selected.astype(np.int64),
        }

        for key in ["mx", "my", "hx", "hy", "imx", "imy", "ihx", "ihy", "offset_mag", "azimuth", "fold"]:
            if key in patch_trace_df.columns:
                sample[f"{key}_patch"] = patch_trace_df[key].to_numpy()

        if ovt_result is not None:
            sample["ovt_keep_trace_mask"] = trace_mask_table["keep"].to_numpy(dtype=np.float32)
            sample["ovt_drop_trace_mask"] = trace_mask_table["drop"].to_numpy(dtype=np.float32)
            sample["ovt_mask_stats"] = ovt_result["stats"]
            sample["ovt_applied_modes"] = np.array(ovt_result["applied_modes"], dtype=object)
            sample["ovt_sort_keys"] = np.array(["imx", "imy", "ihx", "ihy"], dtype=object)

        return sample


if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt

    def plot_patch_ovt_geometry(sample: Dict[str, Any], output_dir: str = ".", prefix: str = "ovt_patch"):
        has_ovt_geom = all(
            key in sample for key in ["mx_patch", "my_patch", "hx_patch", "hy_patch", "offset_mag_patch", "azimuth_patch"]
        )
        if not has_ovt_geom:
            return

        if "ovt_drop_trace_mask" in sample:
            drop_mask = sample["ovt_drop_trace_mask"].astype(bool)
        else:
            drop_mask = sample["mask"][:, 0] <= 0
        keep_mask = ~drop_mask

        mx = np.asarray(sample["mx_patch"], dtype=np.float64)
        my = np.asarray(sample["my_patch"], dtype=np.float64)
        hx = np.asarray(sample["hx_patch"], dtype=np.float64)
        hy = np.asarray(sample["hy_patch"], dtype=np.float64)
        h = np.asarray(sample["offset_mag_patch"], dtype=np.float64)
        azimuth = np.asarray(sample["azimuth_patch"], dtype=np.float64)
        fold = np.asarray(sample["fold_patch"], dtype=np.float64) if "fold_patch" in sample else np.ones_like(h)

        azimuth_deg = np.rad2deg(azimuth)
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        ax.scatter(mx[keep_mask], my[keep_mask], s=12, alpha=0.7, c="tab:blue", label="kept")
        if np.any(drop_mask):
            ax.scatter(mx[drop_mask], my[drop_mask], s=26, alpha=0.95, c="tab:red", marker="x", label="masked")
        ax.set_title("Midpoint distribution with masked traces")
        ax.set_xlabel("mx")
        ax.set_ylabel("my")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.scatter(hx[keep_mask], hy[keep_mask], s=12, alpha=0.7, c="tab:green", label="kept")
        if np.any(drop_mask):
            ax.scatter(hx[drop_mask], hy[drop_mask], s=26, alpha=0.95, c="tab:red", marker="x", label="masked")
        ax.set_title("Offset-vector distribution with masked traces")
        ax.set_xlabel("hx")
        ax.set_ylabel("hy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_geometry.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(h, bins=24, alpha=0.55, color="tab:blue", label="all")
        if np.any(drop_mask):
            axes[0].hist(h[drop_mask], bins=24, alpha=0.8, color="tab:red", label="masked")
        axes[0].set_title("Offset magnitude distribution")
        axes[0].set_xlabel("offset_mag")
        axes[0].set_ylabel("trace count")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].legend()

        axes[1].hist(azimuth_deg, bins=24, alpha=0.55, color="tab:green", label="all")
        if np.any(drop_mask):
            axes[1].hist(azimuth_deg[drop_mask], bins=24, alpha=0.8, color="tab:red", label="masked")
        axes[1].set_title("Azimuth distribution")
        axes[1].set_xlabel("azimuth (degree)")
        axes[1].set_ylabel("trace count")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        axes[1].legend()

        axes[2].hist(fold, bins=24, alpha=0.55, color="tab:purple", label="all")
        if np.any(drop_mask):
            axes[2].hist(fold[drop_mask], bins=24, alpha=0.8, color="tab:red", label="masked")
        axes[2].set_title("Fold distribution")
        axes[2].set_xlabel("fold")
        axes[2].set_ylabel("trace count")
        axes[2].grid(True, linestyle="--", alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_distributions.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        scatter = axes[0].scatter(mx, my, c=h, s=18, cmap="viridis", alpha=0.8)
        if np.any(drop_mask):
            axes[0].scatter(mx[drop_mask], my[drop_mask], s=34, facecolors="none", edgecolors="tab:red", linewidths=1.1)
        axes[0].set_title("Midpoint colored by offset_mag")
        axes[0].set_xlabel("mx")
        axes[0].set_ylabel("my")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label="offset_mag")

        scatter = axes[1].scatter(hx, hy, c=azimuth_deg, s=18, cmap="twilight", alpha=0.8)
        if np.any(drop_mask):
            axes[1].scatter(hx[drop_mask], hy[drop_mask], s=34, facecolors="none", edgecolors="tab:red", linewidths=1.1)
        axes[1].set_title("Offset-vector colored by azimuth")
        axes[1].set_xlabel("hx")
        axes[1].set_ylabel("hy")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label="azimuth (degree)")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_annotated.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    data_dir = "/home/chengzhitong/5d_regular/seis_flow_data12V2"
    h5_irregular = os.path.join(data_dir, "generate_py/h5/segc3na/segc3na_2.h5")
    idx_file = os.path.join(data_dir, "generate_py/h5/segc3/segc3_3_info/kept_trace_indices_random_0.5.npy")

    ds = DatasetH5_ovt_interp(
        h5File_irregular=h5_irregular,
        train_idx_np=None,
        train=True,
        ovt_mask_mode="eval",
        ovt_default_mode="offset_truncation",
        ovt_config={
            "scope": {"type": "local", "patch_ratio": 0.70},
            "truncation_mode": "remove_far",
            "far_quantile": 0.9,
            'near_quantile': 0.1,
            "quantile_scope": "per_midpoint_cell",
        },
    )
    sample = ds[0]
    print("Sample keys:", sample.keys())
    print("data:", sample["data"].shape)
    print("masked_patch:", sample["masked_patch"].shape)
    print("mask:", sample["mask"].shape)
    if "ovt_mask_stats" in sample:
        print("ovt_mask_stats:", sample["ovt_mask_stats"])
    plt.imshow(sample["data"].T, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.savefig("data_ovt.png")
    plt.close()
    plt.imshow(sample["masked_patch"].T, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.savefig("masked_patch_ovt.png")
    plt.close()
    plt.imshow(sample["mask"].T, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.savefig("mask_ovt.png")
    plt.close()
    plot_patch_ovt_geometry(sample, output_dir=".", prefix="patch_ovt")
