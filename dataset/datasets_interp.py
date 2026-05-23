import torch
import numpy as np
from h5py import File
from typing import Optional, Tuple, Dict, Any
try:
    from .config import args
except ImportError:
    from config import args

def apply_random_missing(traces, missing_ratio):
    n_traces, n_samples = traces.shape
    trace_mask = np.random.choice(
        [0, 1], size=(n_traces, 1),
        p=[missing_ratio, 1 - missing_ratio], replace=True
    )
    mask = np.ones((n_traces, n_samples), dtype=np.float32) * trace_mask
    return traces * mask, mask


def amplitude_metadata(thres: float, clip_percentile: float = 99.5) -> Dict[str, Any]:
    thres = float(max(float(thres), 1e-6))
    return {
        "amp_scale": np.float32(thres),
        "amp_clip": np.float32(thres),
        "amp_clip_percentile": np.float32(clip_percentile),
    }


class DatasetH5_interp(torch.utils.data.Dataset):
    """
    地震数据重建数据集，支持：
    - 训练：读取 irregular H5（按 split_core.py 生成的索引），self-supervised masking
    - 测试：按「测线 × 炮集」输入网络（同一 recv_line 上不同炮线/炮桩拆成不同样本）

    索引文件格式：kept_trace_indices_{mode}_{keep_ratio}.npy
    """

    def __init__(
        self,
        h5File_irregular: str,
        h5File_regular: Optional[str] = None,
        train_idx_np: Optional[str] = None,
        train: bool = True,
        survey_line_key: str = "recv_line",  # "shot_line" or "recv_line"
        missing_ratio_range: Tuple[float, float] = (0.4, 0.7),
        use_p_scale: bool = False,
    ):
        """
        Args:
            h5File_irregular: 不规则采样 H5（训练时的输入）
            h5File_regular: 规则网格 H5（测试时的 target）
            train_idx_np: split_core.py 生成的索引文件路径
            train: True=训练模式，False=测试模式
            survey_line_key: 测线维度，"shot_line" 或 "recv_line"
            missing_ratio_range: 训练时随机缺失率范围
        """
        super().__init__()
        print(f"Loading dataset (train={train})...")

        self.h5File_irregular = h5File_irregular
        self.h5File_regular = h5File_regular
        self.train = train
        self.survey_line_key = survey_line_key
        self.missing_ratio_range = missing_ratio_range
        self.use_p_scale = use_p_scale
        self.time_ps = args.time_ps
        self.trace_ps = args.trace_ps
        self.dt_ms = 4
        self.t0_ms = 0
        self.std_val = None

        # ---- 加载不规则 H5 ----
        self.h5_data = self._load_h5_group(h5File_irregular)
        print(f"Irregular H5: {self.h5_data['data'].shape}")

        # ---- 加载训练索引（如果有）----
        self.kept_indices = None
        if train_idx_np is not None:
            self.kept_indices = np.load(train_idx_np)
            print(f"Loaded kept_indices: {len(self.kept_indices)} traces")

        # ---- 加载规则 H5（测试模式）----
        self.h5_data_regular = None
        self.h5_data_tgt = None
        if not train and h5File_regular is not None:
            self.h5_data_regular = self._load_h5_group(h5File_regular)
            self.h5_data_tgt = self.h5_data_regular
            print(f"Regular H5: {self.h5_data_regular['data'].shape}")

        # ---- 计算坐标统计 ----
        self.coord_stats = self.compute_coord_stats()
        print("coord_stats computed")

        # ---- 测试模式：构建测线索引 ----
        if not train:
            self._build_survey_lines()

    def _load_h5_group(self, h5_path: str) -> Dict[str, Any]:
        """加载 H5 的第一个 group（数据组）"""
        h5 = {}
        with File(h5_path, "r") as f:
            for key in f:
                node = f[key]
                if hasattr(node, "keys") and "data" in node:
                    for k in node.keys():
                        h5[k] = node[k][:]
                    break
        return h5

    @staticmethod
    def _row_to_group_key(row: np.ndarray) -> Tuple[int, ...]:
        """将 np.unique 的一行转为可哈希的 group key（测线[, 炮线, 炮桩]）。"""
        r = np.asarray(row, dtype=np.int64).ravel()
        return tuple(int(x) for x in r)

    def _build_survey_lines(self):
        """按测线 × 炮集划分：在 survey_line_key 对应测线内，用 (shot_line, shot_stake) 区分不同炮集。"""
        line_key = self.survey_line_key
        h5 = self.h5_data_regular
        if line_key not in h5:
            raise ValueError(f"Key '{line_key}' not found in regular H5")

        lines = np.asarray(h5[line_key], dtype=np.int64)
        n = len(lines)
        use_shot_split = "shot_line" in h5

        if use_shot_split:
            sl = np.asarray(h5["shot_line"], dtype=np.int64)
            if "shot_stake" in h5:
                ss = np.asarray(h5["shot_stake"], dtype=np.int64)
            elif "shot_no" in h5:
                ss = np.asarray(h5["shot_no"], dtype=np.int64)
            else:
                ss = np.zeros(n, dtype=np.int64)
            keys = np.column_stack([lines, sl, ss])
            uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        else:
            uniq_keys, inv = np.unique(lines, return_inverse=True)
            uniq_keys = uniq_keys.reshape(-1, 1)

        line_indices: Dict[Tuple[int, ...], np.ndarray] = {}
        for k in range(len(uniq_keys)):
            gkey = self._row_to_group_key(uniq_keys[k])
            line_indices[gkey] = np.where(inv == k)[0]

        kept_keys = sorted(line_indices.keys(), key=lambda t: t)
        self.line_indices = {g: line_indices[g] for g in kept_keys}
        self.survey_lines = np.asarray(kept_keys, dtype=object)
        self.num_lines = len(kept_keys)

        if use_shot_split:
            print(f"Survey groups: {self.num_lines} ({line_key} × shot_line × shot_stake)")
        else:
            print(f"Survey lines: {self.num_lines} ({line_key})")
        if self.num_lines == 0:
            raise ValueError(f"No survey groups; check {line_key} and H5 content.")

    def typical_grid_step(self, arr, eps=1e-9):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u
        d = np.diff(u)
        d = d[d > eps]
        if d.size == 0:
            return None, u
        return float(np.median(d)), u

    def compute_coord_stats(self):
        """计算坐标归一化统计量"""

        sx_all = self.h5_data['sx']
        sy_all = self.h5_data['sy']
        rx_all = self.h5_data['rx']
        ry_all = self.h5_data['ry']

        sx_all = np.clip(sx_all, np.percentile(sx_all, 0.01), np.percentile(sx_all, 99.99))
        sy_all = np.clip(sy_all, np.percentile(sy_all, 0.01), np.percentile(sy_all, 99.99))
        rx_all = np.clip(rx_all, np.percentile(rx_all, 0.01), np.percentile(rx_all, 99.99))
        ry_all = np.clip(ry_all, np.percentile(ry_all, 0.01), np.percentile(ry_all, 99.99))

        stats = {
            "sx_min": sx_all.min(), "sx_max": sx_all.max(),
            "sy_min": sy_all.min(), "sy_max": sy_all.max(),
            "rx_min": rx_all.min(), "rx_max": rx_all.max(),
            "ry_min": ry_all.min(), "ry_max": ry_all.max(),
        }

        # 计算 p_scale (每轴 RoPE 频率缩放因子)
        dsx, _ = self.typical_grid_step(sx_all)
        dsy, _ = self.typical_grid_step(sy_all)
        drx, _ = self.typical_grid_step(rx_all)
        dry, _ = self.typical_grid_step(ry_all)
        self.scale = {}
        if dsx is not None and (stats["sx_max"] - stats["sx_min"]) > 0:
            self.scale["sx"] = float((stats["sx_max"] - stats["sx_min"]) / (2 * dsx))
        if dsy is not None and (stats["sy_max"] - stats["sy_min"]) > 0:
            self.scale["sy"] = float((stats["sy_max"] - stats["sy_min"]) / (2 * dsy))
        if drx is not None and (stats["rx_max"] - stats["rx_min"]) > 0:
            self.scale["rx"] = float((stats["rx_max"] - stats["rx_min"]) / (2 * drx))
        if dry is not None and (stats["ry_max"] - stats["ry_min"]) > 0:
            self.scale["ry"] = float((stats["ry_max"] - stats["ry_min"]) / (2 * dry))

        # 在归一化前应用 p_scale：将 stats 的 min/max 乘以 p_scale
        if self.use_p_scale and self.scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.scale.get(name)
                if s is not None:
                    stats[f"{name}_min"] *= s
                    stats[f"{name}_max"] *= s
            print(f"[DatasetH5_interp] p_scale applied to coord_stats: {self.scale}")

        return stats

    def _normalize_coords(self, sx, sy, rx, ry) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stats = self.coord_stats
        sx_n = (sx - stats['sx_min']) / (stats['sx_max'] - stats['sx_min'] + 1e-12)
        sy_n = (sy - stats['sy_min']) / (stats['sy_max'] - stats['sy_min'] + 1e-12)
        rx_n = (rx - stats['rx_min']) / (stats['rx_max'] - stats['rx_min'] + 1e-12)
        ry_n = (ry - stats['ry_min']) / (stats['ry_max'] - stats['ry_min'] + 1e-12)
        return sx_n, sy_n, rx_n, ry_n

    def _time_axis_2d(self, n_trace: int) -> np.ndarray:
        time_idx = np.arange(0, self.time_ps, dtype=np.float32)
        time_axis = self.t0_ms + time_idx * self.dt_ms
        return np.tile(time_axis[None, :], (n_trace, 1))

    def _crop_or_pad_time(self, traces: np.ndarray) -> np.ndarray:
        """时间维度裁剪或填充"""
        if traces.shape[1] > self.time_ps:
            return traces[:, traces.shape[1] - self.time_ps:]
        if traces.shape[1] < self.time_ps:
            pad = self.time_ps - traces.shape[1]
            return np.pad(traces, ((0, 0), (0, pad)), 'constant', constant_values=0)
        return traces

    def __len__(self):
        if self.train:
            # 训练时按 patch 数量
            if self.kept_indices is not None:
                return len(self.kept_indices)
            return len(self.h5_data['data'])
        else:
            # 测试时按测线数量
            return self.num_lines

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.train:
            return self._get_train_item(idx)
        else:
            return self._get_test_item(idx)

    def _get_train_item(self, idx: int) -> Dict[str, Any]:
        """训练：self-supervised learning"""
        np.random.seed(idx)

        if self.kept_indices is not None:
            # 使用预计算的保留索引
            all_indices = self.kept_indices
        else:
            all_indices = np.arange(len(self.h5_data['data']))

        # 随机选取 trace_ps 条道
        n_total = len(all_indices)
        if n_total <= self.trace_ps:
            selected = all_indices
        else:
            start = np.random.randint(0, n_total - self.trace_ps + 1)
            selected = all_indices[start:start + self.trace_ps]

        # 加载数据
        data_full = self.h5_data['data'][selected]
        rx_full = self.h5_data['rx'][selected]
        ry_full = self.h5_data['ry'][selected]
        sx_full = self.h5_data['sx'][selected]
        sy_full = self.h5_data['sy'][selected]

        # 时间裁剪
        data_full = self._crop_or_pad_time(data_full)

        # Self-supervised masking：在已知道中随机缺失
        missing_ratio = np.random.uniform(*self.missing_ratio_range)
        masked_patch, mask = apply_random_missing(data_full, missing_ratio)

        obs = masked_patch[mask > 0]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else 1e-2
        std_val = max(std_val, 1e-2)

        thres = self.global_amp_thres
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_full, -thres, thres) / thres

        # 坐标归一化
        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_full, sy_full, rx_full, ry_full)
        rx_patch = rx_n.astype(np.float32)
        ry_patch = ry_n.astype(np.float32)
        sx_patch = sx_n.astype(np.float32)
        sy_patch = sy_n.astype(np.float32)

        time_axis_2d = self._time_axis_2d(len(selected))

        return {
            'data': data_patch.astype(np.float32),       # 原始数据（target）
            'masked_patch': masked_patch.astype(np.float32),  # masked 输入
            'mask': mask.astype(np.float32),              # 缺失 mask
            'rx_patch': rx_patch,
            'ry_patch': ry_patch,
            'sx_patch': sx_patch,
            'sy_patch': sy_patch,
            'time_axis_2d': time_axis_2d.astype(np.float32),
            'std_val': std_val,
            'trace_indices': selected.astype(np.int64),
            **amplitude_metadata(thres),
        }

    def _get_test_item(self, idx: int) -> Dict[str, Any]:
        """测试：按「测线 × 炮集」加载，用于全观测系统推理。
        - target: 该炮集内完整道数据
        - input: 仅保留 kept_indices 中的道，其余置 0（模拟缺失）
        - line_val: survey_line_key 对应标量（与旧版兼容）
        - gather_key: (测线, 炮线, 炮桩) 或单元素 (测线,)
        - line_group_idx: 本数据集内唯一下标，用于输出文件名避免同测线多炮集冲突
        """
        gkey = self.survey_lines[idx]
        if not isinstance(gkey, tuple):
            gkey = tuple(np.asarray(gkey, dtype=np.int64).ravel().tolist())
        line_trace_idx = self.line_indices[gkey]
        line_val_scalar = int(gkey[0])

        # 加载该测线的全部数据
        data_full = self.h5_data_regular['data'][line_trace_idx]
        tgt_full = self.h5_data_regular['data'][line_trace_idx]
        rx_full = self.h5_data_regular['rx'][line_trace_idx]
        ry_full = self.h5_data_regular['ry'][line_trace_idx]
        sx_full = self.h5_data_regular['sx'][line_trace_idx]
        sy_full = self.h5_data_regular['sy'][line_trace_idx]
        shot_line_full = self.h5_data_regular['shot_line'][line_trace_idx]
        # shot_stake/recv_stake: fixed 模式直接读 H5 的 shot_stake/recv_stake 字段；
        # self_computed 模式 H5 中对应字段为 shot_no/recv_no（见 Segy2H5.py）
        shot_stake_full = (
            self.h5_data_regular['shot_stake'][line_trace_idx]
            if 'shot_stake' in self.h5_data_regular
            else self.h5_data_regular['shot_no'][line_trace_idx]
        )
        recv_line_full = self.h5_data_regular['recv_line'][line_trace_idx]
        recv_stake_full = (
            self.h5_data_regular['recv_stake'][line_trace_idx]
            if 'recv_stake' in self.h5_data_regular
            else self.h5_data_regular['recv_no'][line_trace_idx]
        )

        # 时间处理
        data_full = self._crop_or_pad_time(data_full)
        tgt_full = self._crop_or_pad_time(tgt_full)
        # 将不在 kept_indices 中的道置 0（作为缺失）
        if self.kept_indices is not None:
            # line_trace_idx 中哪些在 kept_indices 中
            kept_set = set(self.kept_indices)
            trace_mask = np.array([idx in kept_set for idx in line_trace_idx])
            data_input = data_full.copy()
            data_input[~trace_mask] = 0.0
        else:
            trace_mask = np.ones(len(line_trace_idx), dtype=bool)
            data_input = data_full
        data_raw = tgt_full.astype(np.float32, copy=True)
        masked_raw = data_input.astype(np.float32, copy=True)

        # 归一化（仅基于观测到的非零道，不依赖 label）
        obs = data_input[trace_mask]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else 1e-2
        std_val = max(std_val, 1e-2)

        thres = np.percentile(np.abs(obs), 99.5) if obs.size > 0 else 1e-6
        thres = max(thres, 1e-6)
        data_patch = np.clip(data_input, -thres, thres) / thres
        tgt_patch = np.clip(tgt_full, -thres, thres) / thres

        # 坐标归一化
        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_full, sy_full, rx_full, ry_full)

        time_axis_2d = self._time_axis_2d(len(line_trace_idx))

        return {
            'data': tgt_patch.astype(np.float32),     # target (完整数据)
            'masked_patch': data_patch.astype(np.float32),  # input (缺失道置 0)
            'trace_mask': trace_mask.astype(np.float32),
            'rx_patch': rx_n.astype(np.float32),
            'ry_patch': ry_n.astype(np.float32),
            'sx_patch': sx_n.astype(np.float32),
            'sy_patch': sy_n.astype(np.float32),
            'shot_line_patch': shot_line_full.astype(np.int32),
            'shot_stake_patch': shot_stake_full.astype(np.int32),
            'recv_line_patch': recv_line_full.astype(np.int32),
            'recv_stake_patch': recv_stake_full.astype(np.int32),
            'time_axis_2d': time_axis_2d.astype(np.float32),
            'std_val': std_val,
            'line_val': line_val_scalar,
            'gather_key': gkey,
            'line_group_idx': int(idx),
            'trace_indices': line_trace_idx.astype(np.int64),
            'data_raw': data_raw,
            'masked_patch_raw': masked_raw,
            **amplitude_metadata(thres),
        }


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = "/home/chengzhitong/5d_regular/seis_flow_data12V2"
    out_dir = os.path.join(data_dir, "test_output")
    os.makedirs(out_dir, exist_ok=True)

    h5_irregular = os.path.join(data_dir, "generate_py/h5/segc3/segc3_3.h5")
    h5_regular = os.path.join(data_dir, "generate_py/h5/segc3/segc3_3.h5")
    idx_file = os.path.join(data_dir, "generate_py/h5/segc3/segc3_3_info/kept_trace_indices_random_0.5.npy")

    # === 训练模式测试 ===
    print("=" * 50)
    print("Training mode test")
    print("=" * 50)

    ds_train = DatasetH5_interp(
        h5File_irregular=h5_irregular,
        train_idx_np=idx_file,
        train=True,
    )
    print(f"Dataset length: {len(ds_train)}")

    sample = ds_train[0]
    print("Sample keys:", sample.keys())
    print(f"  data shape:        {sample['data'].shape}")
    print(f"  masked_patch shape:{sample['masked_patch'].shape}")
    print(f"  mask shape:        {sample['mask'].shape}")
    print(f"  rx_patch range:    [{sample['rx_patch'].min():.3f}, {sample['rx_patch'].max():.3f}]")
    print(f"  ry_patch range:    [{sample['ry_patch'].min():.3f}, {sample['ry_patch'].max():.3f}]")
    print(f"  sx_patch range:    [{sample['sx_patch'].min():.3f}, {sample['sx_patch'].max():.3f}]")
    print(f"  sy_patch range:    [{sample['sy_patch'].min():.3f}, {sample['sy_patch'].max():.3f}]")
    print(f"  std_val:           {sample['std_val']:.6f}")

    # ---- 可视化：整条测线/_patch 的 2D 地震剖面 ----
    def plot_seismic_section(data, title, vmin=None, vmax=None, cmap="seismic", xlabel="Trace", ylabel="Time", save_path=None):
        """绘制 2D 地震剖面"""
        fig, ax = plt.subplots(figsize=(12, 6))
        if vmin is None:
            vmin = np.percentile(data, 2)
        if vmax is None:
            vmax = np.percentile(data, 98)
        im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig, ax

    # 训练 Patch 2D 剖面
    print("Generating train patch seismic sections...")

    # 按 sx 排序形成炮集剖面
    sort_idx = np.argsort(sample['sx_patch'])
    data_sorted = sample['data'][sort_idx]
    masked_sorted = sample['masked_patch'][sort_idx]
    mask_sorted = sample['mask'][sort_idx]
    rx_sorted = sample['rx_patch'][sort_idx]
    ry_sorted = sample['ry_patch'][sort_idx]
    sx_sorted = sample['sx_patch'][sort_idx]
    sy_sorted = sample['sy_patch'][sort_idx]
    trace_idx_sorted = sample['trace_indices'][sort_idx]

    vrange = np.percentile(np.abs(data_sorted), 99)

    # 原始剖面
    fig, ax = plot_seismic_section(
        data_sorted, f"Train Patch - Original (N={len(data_sorted)} traces)",
        vmin=-vrange, vmax=vrange,
        save_path=os.path.join(out_dir, "train_patch_original.png")
    )
    ax.set_title(f"Train Patch - Original (N={len(data_sorted)} traces, sorted by sx)")
    plt.close(fig)

    # Masked 剖面
    fig, ax = plot_seismic_section(
        masked_sorted, f"Train Patch - Masked Input",
        vmin=-vrange, vmax=vrange,
        save_path=os.path.join(out_dir, "train_patch_masked.png")
    )
    plt.close(fig)

    # Mask 剖面
    fig, ax = plot_seismic_section(
        mask_sorted, f"Train Patch - Missing Mask",
        vmin=0, vmax=1, cmap="gray_r",
        save_path=os.path.join(out_dir, "train_patch_mask.png")
    )
    plt.close(fig)

    # 合并对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, d, title, vrange in zip(
        axes,
        [data_sorted, masked_sorted, mask_sorted],
        ["Original", "Masked Input", "Missing Mask"],
        [vrange, vrange, 1]
    ):
        cmap = "seismic" if "Mask" not in title or title == "Missing Mask" else "seismic"
        vmin = -vrange if "Mask" not in title else 0
        vmax = vrange if "Mask" not in title else 1
        im = ax.imshow(d.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace (sorted by sx)")
        ax.set_ylabel("Time sample")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_patch_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Train patch figures saved to {out_dir}/")

    # ---- 可视化：观测系统（检波点空间分布 + 缺失着色）----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 检波点空间分布，颜色=该点所在道的最大振幅
    ax = axes[0]
    amp = np.max(np.abs(data_sorted), axis=1)
    sc = ax.scatter(rx_sorted, ry_sorted, c=amp, s=5, cmap="seismic", vmin=0, vmax=np.percentile(amp, 95))
    ax.set_title("Receiver positions (color = max amplitude)")
    ax.set_xlabel("rx (normalized)")
    ax.set_ylabel("ry (normalized)")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax)

    # 缺失模式着色
    ax = axes[1]
    keep_rate = mask_sorted.mean(axis=1)  # 每道保留比例
    sc = ax.scatter(rx_sorted, ry_sorted, c=keep_rate, s=5, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Missing pattern (green=kept, red=missing)")
    ax.set_xlabel("rx (normalized)")
    ax.set_ylabel("ry (normalized)")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_observation_system.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Observation system figure saved to {out_dir}/train_observation_system.png")

    # === 测试模式测试 ===
    print("\n" + "=" * 50)
    print("Test mode test (by survey line)")
    print("=" * 50)

    ds_test = DatasetH5_interp(
        h5File_irregular=h5_irregular,
        h5File_regular=h5_regular,
        train_idx_np=idx_file,
        train=False,
        survey_line_key="recv_line",
    )
    print(f"Dataset length (num lines): {len(ds_test)}")

    # 遍历所有测线，统计信息
    for i in range(min(3, len(ds_test))):
        sample_t = ds_test[i]
        print(
            f"\nGroup {i}: line_val={sample_t['line_val']}, gather_key={sample_t.get('gather_key')}, "
            f"traces={len(sample_t['trace_indices'])}"
        )
        print(f"  data shape:        {sample_t['data'].shape}")
        print(f"  rx range:          [{sample_t['rx_patch'].min():.3f}, {sample_t['rx_patch'].max():.3f}]")
        print(f"  ry range:          [{sample_t['ry_patch'].min():.3f}, {sample_t['ry_patch'].max():.3f}]")

    # ---- 可视化：测试 - 整条测线的 2D 剖面 ----
    print("\nGenerating test survey line seismic sections...")

    sample_t = ds_test[0]
    vrange_t = np.percentile(np.abs(sample_t['data']), 99)

    # 单条测线：Target vs Input（缺失道为 0）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    im = ax.imshow(sample_t['data'].T, aspect='auto', cmap='seismic', vmin=-vrange_t, vmax=vrange_t)
    ax.set_title(f"Target - Full (N={len(sample_t['trace_indices'])} traces)")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time sample")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(sample_t['masked_patch'].T, aspect='auto', cmap='seismic', vmin=-vrange_t, vmax=vrange_t)
    ax.set_title(f"Input - Missing traces set to 0 (N={len(sample_t['trace_indices'])} traces)")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time sample")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"test_group_{sample_t['line_group_idx']}_target_vs_input.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 缺失 mask 可视化：哪些道被置 0
    input_flat = sample_t['masked_patch'].mean(axis=1)
    missing_mask = (np.abs(input_flat) < 1e-6).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(sample_t['masked_patch'].T, aspect='auto', cmap='seismic', vmin=-vrange_t, vmax=vrange_t)
    ax.set_title(f"Input (group {sample_t['line_group_idx']}, recv_line={sample_t['line_val']})")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time sample")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    sc = ax.scatter(np.arange(len(sample_t['trace_indices'])), np.zeros(len(sample_t['trace_indices'])),
                    c=missing_mask, s=20, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Missing traces (green=observed, red=missing)")
    ax.set_xlabel("Trace index")
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"test_group_{sample_t['line_group_idx']}_missing_mask.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 多条测线对比
    n_lines_show = min(3, len(ds_test))
    fig, axes = plt.subplots(n_lines_show, 3, figsize=(18, 4 * n_lines_show))
    if n_lines_show == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_lines_show):
        sample_i = ds_test[i]
        vrange_i = np.percentile(np.abs(sample_i['data']), 99)
        for col, (d, title) in enumerate(zip(
            [sample_i['masked_patch'], sample_i['data'], sample_i['data'] - sample_i['masked_patch']],
            ["Input (missing=0)", "Target", "Difference"]
        )):
            ax = axes[i, col]
            if title == "Difference":
                vrange_diff = np.percentile(np.abs(d), 99)
                im = ax.imshow(d.T, aspect='auto', cmap='seismic', vmin=-vrange_diff, vmax=vrange_diff)
            else:
                im = ax.imshow(d.T, aspect='auto', cmap='seismic', vmin=-vrange_i, vmax=vrange_i)
            ax.set_title(f"g{sample_i['line_group_idx']} recv_line={sample_i['line_val']} - {title}")
            ax.set_xlabel("Trace index")
            ax.set_ylabel("Time sample")
            plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "test_lines_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 全部测线空间分布（检波点着色：是否缺失）
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(ds_test)):
        sample_i = ds_test[i]
        input_flat_i = sample_i['masked_patch'].mean(axis=1)
        missing_i = (np.abs(input_flat_i) < 1e-6).astype(float)
        ax.scatter(sample_i['rx_patch'], sample_i['ry_patch'],
                   c=missing_i, s=0.5, alpha=0.5, cmap="RdYlGn", vmin=0, vmax=1,
                   label=f"g{sample_i['line_group_idx']} rl={sample_i['line_val']}" if i < 5 else None)
    ax.set_title("All Survey Lines - Missing Pattern (green=observed, red=missing)")
    ax.set_xlabel("rx (normalized)")
    ax.set_ylabel("ry (normalized)")
    ax.set_aspect("equal", adjustable="box")
    if len(ds_test) <= 5:
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "test_all_lines_spatial.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nTest figures saved to {out_dir}/")
    print(f"  - test_group_{{line_group_idx}}_target_vs_input.png (目标 vs 输入对比)")
    print(f"  - test_group_{{line_group_idx}}_missing_mask.png    (缺失道标记)")
    print(f"  - test_lines_comparison.png                  (多条测线对比)")
    print(f"  - test_all_lines_spatial.png                (全部测线空间分布)")
    print(f"\nAll outputs saved to: {out_dir}/")
