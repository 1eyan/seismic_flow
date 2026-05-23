import torch
import numpy as np
from h5py import File
from typing import Optional, Tuple, Dict, Any, Mapping
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
        gather_mode: str = "survey_shot",
        missing_ratio_range: Tuple[float, float] = (0.4, 0.7),
        patch_amp_percentile: float = 99.5,
        time_ps: Optional[int] = None,
        trace_ps: Optional[int] = None,
        overlap_ratio: float = 0.5,
        use_p_scale: bool = False,
        profile = None,
        missing_eps: float = 1e-10,
    ):
        """
        Args:
            h5File_irregular: 不规则采样 H5（训练时的输入；测试时为 mask H5）
            h5File_regular: 规则网格 H5（测试时的 target）
            train_idx_np: split_core.py 生成的索引文件路径
            train: True=训练模式，False=测试模式
            survey_line_key: 测线维度，"shot_line" 或 "recv_line"
            gather_mode: 测试分组方式，"survey_shot"、"shot"、"receiver" 或 "survey_line"
            missing_ratio_range: 训练时随机缺失率范围
            profile: SEG-Y 配置（测试模式必需，用于 per-trace 身份字段和 gather_key）
            missing_eps: 测试模式缺失道判定阈值
        """
        super().__init__()
        print(f"Loading dataset (train={train})...")

        self.h5File_irregular = h5File_irregular
        self.h5File_regular = h5File_regular
        self.train = train
        self.survey_line_key = survey_line_key
        self.gather_mode = gather_mode
        self.missing_ratio_range = missing_ratio_range
        self.use_p_scale = use_p_scale
        self.patch_amp_percentile = float(patch_amp_percentile)
        self.time_ps = int(time_ps if time_ps is not None else args.time_ps)
        self.trace_ps = int(trace_ps if trace_ps is not None else args.trace_ps)
        self.overlap_ratio = float(overlap_ratio)
        self.stride = max(1, int(self.trace_ps * (1.0 - self.overlap_ratio)))
        self.dt_ms = 4
        self.t0_ms = 0
        self.std_val = None
        self.profile = profile
        self.missing_eps = float(missing_eps)

        # ---- 加载不规则 H5 ----
        self.h5_data = self._load_h5_group(h5File_irregular)
        print(f"Irregular H5: {self.h5_data['data'].shape}")

        # ---- 加载训练索引（如果有）----
        self.kept_indices = None
        if train_idx_np is not None:
            self.kept_indices = np.load(train_idx_np)
            print(f"Loaded kept_indices: {len(self.kept_indices)} traces")

        # ---- 加载规则 H5（坐标统计 + 测试 target）----
        self.h5_data_regular = None
        self.h5_data_tgt = None
        if h5File_regular is not None:
            self.h5_data_regular = self._load_h5_group(h5File_regular)
            self.h5_data_tgt = self.h5_data_regular
            print(f"Regular H5: {self.h5_data_regular['data'].shape}")

        # ---- 计算坐标统计 ----
        self.coord_stats = self.compute_coord_stats()
        print("coord_stats computed")

        # ---- 按 profile 排序（训练 & 测试共用）----
        self.sort_order = None
        self.inv_order = None
        if self.profile is not None:
            # 对 h5_data 排序并记录映射（用于 trace_indices）
            sort_fields = []
            for k in self.profile.sort_keys:
                fallback = self.profile.h5_fallback.get(k)
                sort_fields.append(self._h5_int_array(self.h5_data, k, fallback=fallback))
            self.sort_order = np.lexsort(sort_fields[::-1])
            self.inv_order = np.argsort(self.sort_order)

            n = len(self.sort_order)
            for key in list(self.h5_data.keys()):
                arr = self.h5_data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n:
                    self.h5_data[key] = arr[self.sort_order]

            if self.kept_indices is not None:
                self.kept_indices = self.inv_order[self.kept_indices]
                self.kept_indices.sort()

            # 对 regular/tgt 数据集各自独立排序
            for d in [self.h5_data_regular, self.h5_data_tgt]:
                if d is None:
                    continue
                sort_fields_reg = []
                for k in self.profile.sort_keys:
                    fallback = self.profile.h5_fallback.get(k)
                    sort_fields_reg.append(self._h5_int_array(d, k, fallback=fallback))
                order_reg = np.lexsort(sort_fields_reg[::-1])
                m = len(order_reg)
                for key in list(d.keys()):
                    arr = d[key]
                    if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == m:
                        d[key] = arr[order_reg]

            print(f"[DatasetH5_interp] sorted {n} traces by {self.profile.sort_keys}")

        # ---- 设置 regular_data ----
        if self.h5_data_regular is not None:
            self.regular_data = self.h5_data_regular['data']
        else:
            print('no regualr data !')

        # ---- 测试模式：patch-based 滑动窗口 ----
        if not train:
            n_total = len(self.h5_data['data'])
            if n_total >= self.trace_ps:
                starts = list(range(0, n_total - self.trace_ps + 1, self.stride))
                if starts and starts[-1] != n_total - self.trace_ps:
                    starts.append(n_total - self.trace_ps)
                self._test_starts = starts
                self.n_test_patches = len(starts)
            else:
                self._test_starts = []
                self.n_test_patches = 0

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

    @staticmethod
    def _h5_int_array(h5: Dict[str, Any], key: str, fallback: Optional[str] = None, default: int = 0) -> np.ndarray:
        if key in h5:
            return np.asarray(h5[key], dtype=np.int64)
        if fallback is not None and fallback in h5:
            return np.asarray(h5[fallback], dtype=np.int64)
        return np.full(len(h5["data"]), int(default), dtype=np.int64)

    
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
        """计算坐标归一化统计量 + p_scale（RoPE 频率缩放）。坐标范围基于规则网格。"""
        coord_h5 = self.h5_data_regular if self.h5_data_regular is not None else self.h5_data
        sx_all = coord_h5['sx']
        sy_all = coord_h5['sy']
        rx_all = coord_h5['rx']
        ry_all = coord_h5['ry']

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

        # p_scale: per-axis RoPE frequency scaling
        deltas = {}
        for name, arr in [("sx", sx_all), ("sy", sy_all), ("rx", rx_all), ("ry", ry_all)]:
            ds, _ = self.typical_grid_step(arr)
            lo, hi = stats[f"{name}_min"], stats[f"{name}_max"]
            if ds is not None and (hi - lo) > 0:
                deltas[name] = float((hi - lo) / (2 * ds))
        self.scale = deltas

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
        """时间维度裁剪或填充：前面（浅部）填零，统一保留深部"""
        traces = np.asarray(traces)
        if traces.ndim != 2:
            raise ValueError(f"traces must be 2D [N, T], got {traces.shape}")
        diff = traces.shape[1] - self.time_ps
        if diff > 0:
            return traces[:, diff:]
        if diff < 0:
            return np.pad(traces, ((0, 0), (-diff, 0)), "constant", constant_values=0)
        return traces

    def __len__(self):
        if self.train:
            if self.kept_indices is not None:
                n_total = len(self.kept_indices)
            else:
                n_total = len(self.h5_data['data'])
            return max(1, (n_total - self.trace_ps) // self.stride + 1) * 6
        else:
            return self.n_test_patches

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.train:
            return self._get_train_item(idx)
        else:
            return self._get_test_item(idx)

    def _get_train_item(self, idx: int) -> Dict[str, Any]:
        """训练：self-supervised learning（顺序窗口 + 重叠）"""
        if self.kept_indices is not None:
            all_indices = self.kept_indices
        else:
            all_indices = np.arange(len(self.h5_data['data']))

        n_total = len(all_indices)
        # idx -> 顺序窗口起始位置（重叠由 stride 控制）
        idx_wrap = idx % max(1, (n_total - self.trace_ps) // self.stride + 1)
        start = idx_wrap * self.stride
        end = start + self.trace_ps

        if end <= n_total:
            selected = all_indices[start:end]
        else:
            # 末尾不足：补齐到 trace_ps
            selected = all_indices[start:n_total]
            if len(selected) < self.trace_ps:
                pad = np.random.choice(all_indices, self.trace_ps - len(selected), replace=False)
                selected = np.concatenate([selected, pad])

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

        # 归一化（基于 mask 前的原始数据）
        obs = data_full[np.isfinite(data_full)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else 1e-2
        std_val = max(std_val, 1e-2)

        thres = np.percentile(np.abs(data_full), 99.5) if obs.size > 0 else 1e-6
        thres = max(thres, 1e-6)
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
            'trace_mask': np.any(mask > 0, axis=1).astype(np.float32),
            'rx_patch': rx_patch,
            'ry_patch': ry_patch,
            'sx_patch': sx_patch,
            'sy_patch': sy_patch,
            'time_axis_2d': time_axis_2d.astype(np.float32),
            'std_val': std_val,
            'trace_indices': (self.sort_order[selected] if self.sort_order is not None
                              else selected).astype(np.int64),
            **amplitude_metadata(thres, self.patch_amp_percentile),
        }

    def _get_test_item(self, idx: int) -> Dict[str, Any]:
        n_total = len(self.h5_data['data'])

        # gen_patches_torch 风格：只生成有效 patch，最后窗口左对齐末尾
        if n_total < self.trace_ps:
            selected = np.arange(n_total)
        else:
            max_start = n_total - self.trace_ps
            start = min(idx * self.stride, max_start)
            selected = np.arange(start, start + self.trace_ps)

        observed = self.h5_data['data'][selected]
        target = self.regular_data[selected]

        observed = self._crop_or_pad_time(observed)
        target = self._crop_or_pad_time(target)

        trace_mask = np.any(np.abs(observed) > self.missing_eps, axis=1).astype(np.float32)

        obs = observed[trace_mask > 0]
        obs = obs[np.isfinite(obs)]
        thres = np.percentile(np.abs(obs), self.patch_amp_percentile) if obs.size > 0 else 1e-6
        thres = max(thres, 1e-6)

        masked_patch = np.clip(observed, -thres, thres) / thres
        data_patch = np.clip(target, -thres, thres) / thres

        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(
            self.h5_data['sx'][selected],
            self.h5_data['sy'][selected],
            self.h5_data['rx'][selected],
            self.h5_data['ry'][selected],
        )

        return {
            'data': data_patch.astype(np.float32),
            'masked_patch': masked_patch.astype(np.float32),
            'trace_mask': trace_mask,
            'sx_patch': sx_n.astype(np.float32),
            'sy_patch': sy_n.astype(np.float32),
            'rx_patch': rx_n.astype(np.float32),
            'ry_patch': ry_n.astype(np.float32),
            'amp_scale': np.float32(thres),
            'line_group_idx': np.int64(idx),
            'trace_indices': (self.sort_order[selected] if self.sort_order is not None
                              else selected).astype(np.int64),
        }


class DatasetH5_interp_v2(DatasetH5_interp):
    """
    统一口径版本（不覆盖旧类）：
    - 振幅：全局 thres（irregular 全体 |amp| p99.5）
    - 训练缺失：保持随机整道缺失
    - 组织：保留 H5 / 测线×炮集
    - 坐标：对齐 build_masked_dataset（全局 min-max，零跨度 -> 0.5）
    - gather_key：每道四元组 (shot_line, shot_no, recv_line, recv_no)
    """

    def __init__(
        self,
        h5File_irregular: str,
        h5File_regular: Optional[str] = None,
        train_idx_np: Optional[str] = None,
        train: bool = True,
        survey_line_key: str = "recv_line",
        gather_mode: str = "survey_shot",
        missing_ratio_range: Tuple[float, float] = (0.4, 0.7),
        global_amp_percentile: float = 99.5,
        time_ps: Optional[int] = None,
        trace_ps: Optional[int] = None,
        overlap_ratio: float = 0.5,
        use_p_scale: bool = False,
    ):
        self.global_amp_percentile = float(global_amp_percentile)
        super().__init__(
            h5File_irregular=h5File_irregular,
            h5File_regular=h5File_regular,
            train_idx_np=train_idx_np,
            train=train,
            survey_line_key=survey_line_key,
            gather_mode=gather_mode,
            missing_ratio_range=missing_ratio_range,
            time_ps=time_ps,
            trace_ps=trace_ps,
            overlap_ratio=overlap_ratio,
            use_p_scale=use_p_scale,
        )
        self.global_amp_thres = float(
            np.percentile(np.abs(self.h5_data["data"]), self.global_amp_percentile)
        )
        self.global_amp_thres = max(self.global_amp_thres, 1e-6)
        print(
            f"Global amplitude threshold: p{self.global_amp_percentile:.1f}="
            f"{self.global_amp_thres:.6g}"
        )

    def compute_coord_stats(self):
        """坐标口径对齐 build_masked_dataset：全局 min/max，不做百分位裁剪。坐标范围基于规则网格。"""
        coord_h5 = self.h5_data_regular 
        sx_all = coord_h5["sx"].astype(np.float64)
        sy_all = coord_h5["sy"].astype(np.float64)
        rx_all = coord_h5["rx"].astype(np.float64)
        ry_all = coord_h5["ry"].astype(np.float64)
        stats = {
            "sx_min": float(np.min(sx_all)), "sx_max": float(np.max(sx_all)),
            "sy_min": float(np.min(sy_all)), "sy_max": float(np.max(sy_all)),
            "rx_min": float(np.min(rx_all)), "rx_max": float(np.max(rx_all)),
            "ry_min": float(np.min(ry_all)), "ry_max": float(np.max(ry_all)),
        }

        # p_scale: per-axis RoPE frequency scaling
        deltas = {}
        for name, arr in [("sx", sx_all), ("sy", sy_all), ("rx", rx_all), ("ry", ry_all)]:
            ds, _ = self.typical_grid_step(arr)
            lo, hi = stats[f"{name}_min"], stats[f"{name}_max"]
            if ds is not None and (hi - lo) > 0:
                deltas[name] = float((hi - lo) / (2 * ds))
        self.scale = deltas

        # 在归一化前应用 p_scale：将 stats 的 min/max 乘以 p_scale
        if self.use_p_scale and self.scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.scale.get(name)
                if s is not None:
                    stats[f"{name}_min"] *= s
                    stats[f"{name}_max"] *= s
            print(f"[DatasetH5_interp_v2] p_scale applied to coord_stats: {self.scale}")

        return stats

    def _normalize_coords(self, sx, sy, rx, ry) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """每维全局 min-max，零跨度维设为 0.5（与 build_masked_dataset 一致）。"""
        stats = self.coord_stats

        def _norm(arr, lo, hi):
            arr = np.asarray(arr, dtype=np.float64)
            if (hi - lo) > 1e-8:
                return (arr - lo) / (hi - lo)
            return np.full(arr.shape, 0.5, dtype=np.float64)

        sx_n = _norm(sx, stats["sx_min"], stats["sx_max"])
        sy_n = _norm(sy, stats["sy_min"], stats["sy_max"])
        rx_n = _norm(rx, stats["rx_min"], stats["rx_max"])
        ry_n = _norm(ry, stats["ry_min"], stats["ry_max"])
        return sx_n, sy_n, rx_n, ry_n

    def _get_train_item(self, idx: int) -> Dict[str, Any]:
        """训练：保持随机整道缺失；振幅归一化使用全局 thres。"""
        if self.kept_indices is not None:
            all_indices = self.kept_indices
        else:
            all_indices = np.arange(len(self.h5_data["data"]))

        n_total = len(all_indices)
        idx_wrap = idx % max(1, (n_total - self.trace_ps) // self.stride + 1)
        start = idx_wrap * self.stride
        end = start + self.trace_ps

        if end <= n_total:
            selected = all_indices[start:end]
        else:
            selected = all_indices[start:n_total]
            if len(selected) < self.trace_ps:
                pad = np.random.choice(all_indices, self.trace_ps - len(selected), replace=False)
                selected = np.concatenate([selected, pad])

        data_full = self.h5_data["data"][selected]
        rx_full = self.h5_data["rx"][selected]
        ry_full = self.h5_data["ry"][selected]
        sx_full = self.h5_data["sx"][selected]
        sy_full = self.h5_data["sy"][selected]

        data_full = self._crop_or_pad_time(data_full)
        missing_ratio = np.random.uniform(*self.missing_ratio_range)
        masked_patch, mask = apply_random_missing(data_full, missing_ratio)

        obs = masked_patch[mask > 0]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else 1e-2
        std_val = max(std_val, 1e-2)

        thres = self.global_amp_thres
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_full, -thres, thres) / thres

        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_full, sy_full, rx_full, ry_full)
        time_axis_2d = self._time_axis_2d(len(selected))

        return {
            "data": data_patch.astype(np.float32),
            "masked_patch": masked_patch.astype(np.float32),
            "mask": mask.astype(np.float32),
            "trace_mask": np.any(mask > 0, axis=1).astype(np.float32),
            "rx_patch": rx_n.astype(np.float32),
            "ry_patch": ry_n.astype(np.float32),
            "sx_patch": sx_n.astype(np.float32),
            "sy_patch": sy_n.astype(np.float32),
            "time_axis_2d": time_axis_2d.astype(np.float32),
            "std_val": std_val,
            "trace_indices": selected.astype(np.int64),
            **amplitude_metadata(thres, self.global_amp_percentile),
        }

    


if __name__ == "__main__":
    import os, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from segy_schema import get_segy_profile

    out_dir = "./dataset_v2_test_output"
    os.makedirs(out_dir, exist_ok=True)

    h5_regular = "/cloud/cloud-s3fs/reg5dbin_label1031.h5"
    h5_mask = "/cloud/cloud-s3fs/reg5dbin_label1031_binning.h5"
    h5_irr ='/cloud/cloud-s3fs/raw5d_data1104.h5'
    
    profile = get_segy_profile("field1031")

    import h5py
    for label, path in [("regular", h5_regular), ("mask", h5_mask)]:
        with h5py.File(path, "r") as f:
            groups = list(f.keys())
            print(f"H5 {label}: groups={groups}")
            for g in groups:
                if "data" in f[g]:
                    d = f[g]["data"]
                    print(f"  group '{g}': data.shape={d.shape}, dtype={d.dtype}")
                    for key in list(profile.sort_keys) + ["sx", "sy", "rx", "ry"]:
                        if key in f[g]:
                            print(f"    {key}: shape={f[g][key].shape}, range=[{f[g][key][:].min()}, {f[g][key][:].max()}]")
                    break
    print()

    # === Patch-based 推理模式 ===
    print("=" * 50)
    print("Inference mode test")
    print("=" * 50)

    ds = DatasetH5_interp(
        h5File_irregular=h5_irr,
        h5File_regular=h5_regular,
        train=True,
        time_ps=1256,
        trace_ps=128,
        overlap_ratio=0.25,
        use_p_scale=True,
        profile=profile,
        missing_eps=1e-10,
    )
    print(f"Patches: {len(ds)}, time_ps={ds.time_ps}, trace_ps={ds.trace_ps}, stride={ds.stride}")

    # ---- 取前 3 个 patch 查看 ----
    for i in range(min(3, len(ds))):
        s = ds[i]
        n = len(s['trace_indices'])
        n_miss = int((s['trace_mask'] < 0.5).sum())
        print(f"Patch {i}: traces={n} missing={n_miss} "
              f"amp_scale={s['amp_scale']:.3f} "
              f"rx=[{s['rx_patch'].min():.3f},{s['rx_patch'].max():.3f}] "
              f"ry=[{s['ry_patch'].min():.3f},{s['ry_patch'].max():.3f}] "
              f"sx=[{s['sx_patch'].min():.3f},{s['sx_patch'].max():.3f}] "
              f"sy=[{s['sy_patch'].min():.3f},{s['sy_patch'].max():.3f}]")

    # ---- 可视化第 0 个 patch ----
    s0 = ds[0]
    vmax = float(max(abs(s0['data']).max(), abs(s0['masked_patch']).max()) or 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, d, title in zip(
        axes,
        [s0['data'], s0['masked_patch'], s0['data'] - s0['masked_patch']],
        ["Target (regular)", "Input (masked)", "Difference"]
    ):
        v = vmax if "Difference" not in title else float(max(abs(d).max() or 1e-6, 1e-6))
        im = ax.imshow(d.T, aspect='auto', cmap='seismic', vmin=-v, vmax=v, origin='upper')
        ax.set_title(f"{title}\n({s0['data'].shape[0]} traces × {s0['data'].shape[1]} samples)")
        ax.set_xlabel("Trace index")
        ax.set_ylabel("Time sample")
        plt.colorbar(im, ax=ax)
    fig.savefig(os.path.join(out_dir, "patch_inference.png"), dpi=150)
    plt.close(fig)

    # ---- 坐标空间分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(s0['rx_patch'], s0['ry_patch'], c=s0['trace_mask'], s=10,
                    cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title(f"Receiver positions (green=observed, red=missing)")
    ax.set_xlabel("rx (normalized)")
    ax.set_ylabel("ry (normalized)")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax)

    ax = axes[1]
    sc = ax.scatter(s0['sx_patch'], s0['sy_patch'], c=s0['trace_mask'], s=10,
                    cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title(f"Source positions (green=observed, red=missing)")
    ax.set_xlabel("sx (normalized)")
    ax.set_ylabel("sy (normalized)")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax)

    fig.savefig(os.path.join(out_dir, "patch_coordinates.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutputs saved to {out_dir}/")
