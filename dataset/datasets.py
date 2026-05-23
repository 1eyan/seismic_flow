import datetime
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np
import math
from h5py import File
from .config import args
from typing import Union, Any, List, Sequence

try:
    from ..reg_tool.patch_sampler import diverse_topk
except ImportError:
    import sys
    from pathlib import Path
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from reg_tool.patch_sampler import diverse_topk

ArrayLike = Union[np.ndarray, "Any"]
##
def sample_missing_ratio(a=2.0, b=5.0, min_r=args.min_r, max_r=args.max_r):
    """采样缺失率"""
    r = np.random.beta(a, b)
    return min_r + (max_r - min_r) * r

def apply_random_missing(traces, missing_ratio):
    n_traces, n_samples = traces.shape
    trace_mask = np.random.choice(
        [0, 1], size=(n_traces, 1),
        p=[missing_ratio, 1 - missing_ratio], replace=True
    )
    mask = np.ones((n_traces, n_samples), dtype=np.float32) * trace_mask
    return traces * mask, mask

def apply_block_missing(traces,):
    n_traces, n_samples = traces.shape
    missing_ratio = np.random.uniform(0.1, 0.3)
    mask = np.ones((n_traces, n_samples), dtype=np.float32)
    n_missing = int(n_traces * missing_ratio)
    if n_missing > 0:
        start = np.random.randint(0, max(1, n_traces - n_missing))
        mask[start:start + n_missing, :] = 0.0
    return traces * mask, mask

def apply_mixed_mask(traces, missing_ratio, block_prob=0.4):
    """混合缺失模式"""
    if np.random.rand() < block_prob:
        return apply_block_missing(traces,)
    else:
        return apply_random_missing(traces, missing_ratio)


def amplitude_metadata(thres: float, clip_percentile: float = 99.5) -> Dict[str, Any]:
    thres = float(max(float(thres), 1e-6))
    return {
        "amp_scale": np.float32(thres),
        "amp_clip": np.float32(thres),
        "amp_clip_percentile": np.float32(clip_percentile),
    }


def _augment_coords(rx, ry, sx, sy, jitter=0.05, rot_scale=True, center_prob=0.5):
    """坐标增强：旋转+缩放+随机中心化 + 可选 jitter"""
    rx = rx.copy()
    ry = ry.copy()
    sx = sx.copy()
    sy = sy.copy()

    # 轻微 jitter
    rx += np.random.uniform(-jitter, jitter, size=rx.shape)
    ry += np.random.uniform(-jitter, jitter, size=ry.shape)
    sx += np.random.uniform(-jitter, jitter, size=sx.shape)
    sy += np.random.uniform(-jitter, jitter, size=sy.shape)

    if rot_scale:
        # 随机中心化
        if np.random.rand() < center_prob:
            if np.random.rand() < 0.5:
                dx, dy = np.random.choice(rx), np.random.choice(ry)
            else:
                dx, dy = np.random.choice(sx), np.random.choice(sy)
            rx -= dx
            ry -= dy
            sx -= dx
            sy -= dy

        # 旋转
        theta = np.random.rand() * 2.0 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        rx_, ry_ = rx*c - ry*s, rx*s + ry*c
        sx_, sy_ = sx*c - sy*s, sx*s + sy*c
        rx, ry, sx, sy = rx_, ry_, sx_, sy_

        # 缩放
        scale = np.random.uniform(0.8, 1.2)
        rx *= scale
        ry *= scale
        sx *= scale
        sy *= scale

    # clip
    rx = np.clip(rx, -1.5, 1.5)
    ry = np.clip(ry, -1.5, 1.5)
    sx = np.clip(sx, -1.5, 1.5)
    sy = np.clip(sy, -1.5, 1.5)
    return rx, ry, sx, sy



class DatasetH5_all_queryctx(torch.utils.data.Dataset):

    def __init__(
        self,
        h5File=None,
        h5File_regular=None,
        h5File_tgt=None,
        dataset_neighbors=None,
        train=None,
        train_num_query: int = 16,
        train_context_size: Optional[int] = None,
        patch_beta: float = 0.3,
        patch_metric_weights=None,
        force_anchor_query: bool = False,
        trace_sort_keys: Tuple[str, ...] = ("rx", "ry", "sx", "sy"),
        use_p_scale: bool = False,
    ):
        super().__init__()
        print('Loading dataset...')

        self.h5File = h5File
        self.h5File_regular = h5File_regular
        self.h5File_tgt = h5File_tgt
        self.time_ps = args.time_ps
        self.trace_ps = args.trace_ps
        self.train = train
        self._rng = np.random.default_rng(123)
        self.std_val = None
        self.train_num_query = int(max(1, train_num_query))
        self.train_context_size = (
            None if train_context_size is None else int(max(1, train_context_size))
        )
        self.patch_beta = float(patch_beta)
        self.patch_metric_weights = patch_metric_weights
        self.force_anchor_query = bool(force_anchor_query)
        self.trace_sort_keys = tuple(trace_sort_keys)
        self.use_p_scale = use_p_scale

        self.dt_ms = 4
        self.t0_ms = 0
        self.scale = None
        
        h5 = File(self.h5File, "r")
        self.h5_data = {}
        for key in h5:
            node = h5[key]
            if hasattr(node, "keys") and "data" in node:
                break
        for key in node.keys():
            self.h5_data[key] = node[key]
        self.h5_data_regular = {}
        h5_regular = File(self.h5File_regular, "r")
        for key in h5_regular:
            node = h5_regular[key]
            if hasattr(node, "keys") and "data" in node:
                break
        for key in node.keys():
            self.h5_data_regular[key] = node[key]
        self.h5_data_tgt = {}
        h5_tgt = File(self.h5File_tgt, "r")
        for key in h5_tgt:
            node = h5_tgt[key]
            if hasattr(node, "keys") and "data" in node:
                break
        for key in node.keys():
            self.h5_data_tgt[key] = node[key]
        print(self.h5_data_regular["data"].shape)
        print(self.h5_data["data"].shape)
        print("loading data")
        
        self.coord_stats = self.compute_coord_stats(regular=not self.train)
        print('coord_stats computed')
        self.patch_meta = self._load_patch_metadata(dataset_neighbors)
        self.patch_mode = self.patch_meta["mode"]
        print(self.patch_mode)
        self.num_samples = int(self.patch_meta["num_samples"])
        print(f"patch metadata mode: {self.patch_mode}, samples: {self.num_samples}")

    def typical_grid_step(self,arr, eps=1e-9):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u  # 无法估步长
        d = np.diff(u)
        d = d[d > eps]     # 去掉重复和数值噪声
        if d.size == 0:
            return None, u
        return float(np.median(d)), u    

    def __len__(self):
        return self.num_samples

    def _load_patch_metadata(self, path: Optional[str]) -> Dict[str, Any]:
        if path is None:
            raise ValueError("dataset_neighbors is required")
        raw = np.load(path, allow_pickle=True)
        if hasattr(raw, "files"):
            arrays = {k: raw[k] for k in raw.files}
            raw.close()
        else:
            arrays = {"0": raw}

        if "grid_query_idx_list" in arrays and (
            "context_idx_list" in arrays or "patch_idx_list" in arrays
        ):
            num_samples = len(arrays["grid_query_idx_list"])
            return {
                "mode": "infer_query_context",
                "num_samples": int(num_samples),
                "grid_query_idx_list": arrays["grid_query_idx_list"],
                "context_idx_list": arrays.get(
                    "context_idx_list",
                    arrays.get("patch_idx_list"),
                ),
                "block_id": arrays.get("block_id"),
                "block_center_grid_idx": arrays.get("block_center_grid_idx"),
                "anchor_grid_idx_list": arrays.get("anchor_grid_idx_list"),
            }

        if "pool_idx_2d" in arrays:
            pool_idx_2d = np.asarray(arrays["pool_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(pool_idx_2d.shape[0]),
                "pool_idx_2d": pool_idx_2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        if "patch_idx_2d" in arrays and self.train:
            patch_idx_2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(patch_idx_2d.shape[0]),
                "pool_idx_2d": patch_idx_2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        if "patch_idx_2d" in arrays and (not self.train):
            patch_idx_2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "legacy",
                "num_samples": int(patch_idx_2d.shape[0]),
                "patch_idx_2d": patch_idx_2d,
            }

        if "0" in arrays:
            patch_idx_2d = np.asarray(arrays["0"], dtype=np.int64)
            return {
                "mode": "legacy",
                "num_samples": int(patch_idx_2d.shape[0]),
                "patch_idx_2d": patch_idx_2d,
            }

        raise ValueError(
            "Unsupported dataset_neighbors format. Expected legacy ['0'], "
            "train pool keys, or infer query/context keys."
        )

    def _index_row(self, storage: np.ndarray, idx: int) -> np.ndarray:
        row = np.asarray(storage[idx], dtype=np.int64).reshape(-1)
        return row[row >= 0]

    def _take_rows(self, dataset, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            sample = np.asarray(dataset[:1])
            if sample.ndim == 1:
                return np.zeros((0,), dtype=sample.dtype)
            return np.zeros((0, sample.shape[1]), dtype=sample.dtype)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        out = np.asarray(dataset[sorted_idx])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return out[inv]

    def _crop_or_pad_time(self, traces: np.ndarray) -> np.ndarray:
        traces = np.asarray(traces)
        if traces.ndim != 2:
            raise ValueError(f"traces must be 2D [N, T], got {traces.shape}")
        diff = traces.shape[1] - self.time_ps
        if diff > 0:
            return traces[:, diff:]
        if diff < 0:
            return np.pad(traces, ((0, 0), (-diff, 0)), 'constant', constant_values=0)
        return traces

    def _time_axis_2d(self, n_trace: int) -> np.ndarray:
        time_idx_1d = np.arange(0, self.time_ps, dtype=np.int32)
        time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
        return np.tile(time_axis_1d[None, :], (int(n_trace), 1)).astype(np.float32)

    def _scale_pair(
        self,
        data_patch: np.ndarray,
        masked_patch: np.ndarray,
        is_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
        obs = masked_patch[~is_query]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else np.float32(0.0)
        std_val = np.float32(max(std_val, 1e-2))
        ref = np.abs(obs) if obs.size > 0 else np.abs(masked_patch[np.isfinite(masked_patch)])
        thres = np.percentile(ref, 99.5) if ref.size > 0 else 1e-6
        thres = float(max(thres, 1e-6))
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_patch, -thres, thres) / thres
        self.std_val = std_val
        return (
            data_patch.astype(np.float32),
            masked_patch.astype(np.float32),
            std_val,
            np.float32(thres),
        )

    def _sample_rng(self, idx: int) -> np.random.Generator:
        seed = int(self._rng.integers(0, 2**31 - 1)) ^ int(idx)
        return np.random.default_rng(seed)

    _COORD_COL = {"sx": 0, "sy": 1, "rx": 2, "ry": 3}

    def _sort_traces(
        self,
        data_patch: np.ndarray,
        is_query: np.ndarray,
        coords_patch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按 self.trace_sort_keys 指定的坐标列排序道序.

        coords_patch 列序: 0=sx, 1=sy, 2=rx, 3=ry.
        np.lexsort 最后一个 key 为主排序键, 所以反转 keys 传入.
        Returns: (data_sorted, is_query_sorted, coords_sorted, order)
        """
        if not self.trace_sort_keys:
            order = np.arange(data_patch.shape[0])
            return data_patch, is_query, coords_patch, order
        cols = [coords_patch[:, self._COORD_COL[k]] for k in reversed(self.trace_sort_keys)]
        order = np.lexsort(cols)
        return data_patch[order], is_query[order], coords_patch[order], order

    def _build_train_query_context_sample(self, idx: int) -> Dict[str, Any]:
        pool_idx = self._index_row(self.patch_meta["pool_idx_2d"], idx)
        if pool_idx.size < 2:
            raise RuntimeError("train pool must contain at least 2 traces")
        anchor_idx = None
        if self.patch_meta.get("anchor_idx") is not None:
            anchor_idx = int(np.asarray(self.patch_meta["anchor_idx"])[idx])

        data_pool = self._crop_or_pad_time(self._take_rows(self.h5_data['data'], pool_idx)).astype(np.float32)
        rx_pool = self._take_rows(self.h5_data['rx'], pool_idx).astype(np.float32)
        ry_pool = self._take_rows(self.h5_data['ry'], pool_idx).astype(np.float32)
        sx_pool = self._take_rows(self.h5_data['sx'], pool_idx).astype(np.float32)
        sy_pool = self._take_rows(self.h5_data['sy'], pool_idx).astype(np.float32)

        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_pool, sy_pool, rx_pool, ry_pool)
        coords_pool = np.stack([sx_n, sy_n, rx_n, ry_n], axis=1).astype(np.float32)

        rng = self._sample_rng(idx)
        q_eff = min(self.train_num_query, int(pool_idx.size) - 1)
        if q_eff < 1:
            raise RuntimeError("effective train query count must be >= 1")
        k_ctx_target = (
            max(1, self.trace_ps - q_eff)
            if self.train_context_size is None
            else self.train_context_size
        )
        k_ctx = min(int(k_ctx_target), int(pool_idx.size) - q_eff)
        if k_ctx < 1:
            raise RuntimeError("effective train context count must be >= 1")

        perm = rng.permutation(pool_idx.size)
        if (
            self.force_anchor_query
            and anchor_idx is not None
            and np.any(pool_idx == anchor_idx)
        ):
            anchor_local = int(np.flatnonzero(pool_idx == anchor_idx)[0])
            rest = perm[perm != anchor_local]
            extra = rest[: max(0, q_eff - 1)]
            query_local = np.concatenate(
                [np.asarray([anchor_local], dtype=np.int64), extra.astype(np.int64)],
                axis=0,
            )
        else:
            query_local = perm[:q_eff].astype(np.int64, copy=False)
        candidate_local = np.asarray(
            [i for i in range(pool_idx.size) if i not in set(query_local.tolist())],
            dtype=np.int64,
        )
        center_coord = np.mean(coords_pool[query_local], axis=0).astype(np.float32, copy=False)
        context_local = diverse_topk(
            center_coord=center_coord,
            candidate_idx=candidate_local,
            all_coords=coords_pool,
            k=k_ctx,
            metric_weights=self.patch_metric_weights,
            beta=self.patch_beta,
        ).astype(np.int64, copy=False)
        if context_local.size == 0:
            raise RuntimeError("failed to build non-empty training context from pool")

        patch_local = np.concatenate([query_local, context_local], axis=0)
        data_patch = data_pool[patch_local].astype(np.float32, copy=False)
        is_query_orig = np.zeros((patch_local.size,), dtype=bool)
        is_query_orig[: query_local.size] = True

        coords_patch = coords_pool[patch_local].astype(np.float32, copy=False)
        data_patch, is_query, coords_patch, _ = self._sort_traces(
            data_patch, is_query_orig, coords_patch,
        )

        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
        data_patch, masked_patch, std_val, thres = self._scale_pair(data_patch, masked_patch, is_query)
        return {
            'data': data_patch,
            'masked_patch': masked_patch,
            'rx_patch': coords_patch[:, 2].astype(np.float32, copy=False),
            'ry_patch': coords_patch[:, 3].astype(np.float32, copy=False),
            'sx_patch': coords_patch[:, 0].astype(np.float32, copy=False),
            'sy_patch': coords_patch[:, 1].astype(np.float32, copy=False),
            'time_axis_2d': self._time_axis_2d(patch_local.size),
            'std_val': std_val,
            'is_query': is_query,
            'query_count': np.int64(query_local.size),
            'context_count': np.int64(context_local.size),
            'query_global_idx': pool_idx[query_local].astype(np.int64, copy=False),
            'context_global_idx': pool_idx[context_local].astype(np.int64, copy=False),
            'pool_global_idx': pool_idx.astype(np.int64, copy=False),
            'anchor_global_idx': np.int64(-1 if anchor_idx is None else anchor_idx),
            **amplitude_metadata(thres),
        }

    def _build_infer_query_context_sample(self, idx: int) -> Dict[str, Any]:
        query_idx = self._index_row(self.patch_meta["grid_query_idx_list"], idx)
        context_idx = self._index_row(self.patch_meta["context_idx_list"], idx)
        if query_idx.size == 0 or context_idx.size == 0:
            raise RuntimeError("infer sample must contain non-empty query and context")

        query_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data_regular['data'], query_idx)
        ).astype(np.float32)
        context_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data['data'], context_idx)
        ).astype(np.float32)
        data_patch = np.concatenate([query_data, context_data], axis=0).astype(np.float32, copy=False)
        is_query_orig = np.zeros((data_patch.shape[0],), dtype=bool)
        is_query_orig[: query_idx.size] = True

        rx_q = self._take_rows(self.h5_data_regular['rx'], query_idx).astype(np.float32)
        ry_q = self._take_rows(self.h5_data_regular['ry'], query_idx).astype(np.float32)
        sx_q = self._take_rows(self.h5_data_regular['sx'], query_idx).astype(np.float32)
        sy_q = self._take_rows(self.h5_data_regular['sy'], query_idx).astype(np.float32)
        rx_c = self._take_rows(self.h5_data['rx'], context_idx).astype(np.float32)
        ry_c = self._take_rows(self.h5_data['ry'], context_idx).astype(np.float32)
        sx_c = self._take_rows(self.h5_data['sx'], context_idx).astype(np.float32)
        sy_c = self._take_rows(self.h5_data['sy'], context_idx).astype(np.float32)
        sx_qn, sy_qn, rx_qn, ry_qn = self._normalize_coords(sx_q, sy_q, rx_q, ry_q)
        sx_cn, sy_cn, rx_cn, ry_cn = self._normalize_coords(sx_c, sy_c, rx_c, ry_c)

        coords_patch = np.stack([
            np.concatenate([sx_qn, sx_cn]),
            np.concatenate([sy_qn, sy_cn]),
            np.concatenate([rx_qn, rx_cn]),
            np.concatenate([ry_qn, ry_cn]),
        ], axis=1).astype(np.float32)
        data_patch, is_query, coords_patch, _order = self._sort_traces(
            data_patch, is_query_orig, coords_patch,
        )
        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
        data_raw = data_patch.astype(np.float32, copy=True)
        masked_raw = masked_patch.astype(np.float32, copy=True)
        data_patch, masked_patch, std_val, thres = self._scale_pair(data_patch, masked_patch, is_query)

        # patch_info: Q+K 道，按 _order 排序，与 data 一致
        sl_q = self._take_rows(self.h5_data_regular['shot_line'], query_idx)
        ss_q = self._take_rows(self.h5_data_regular['shot_stake'], query_idx)
        rl_q = self._take_rows(self.h5_data_regular['recv_line'], query_idx)
        rs_q = self._take_rows(self.h5_data_regular['recv_stake'], query_idx)
        sl_c = self._take_rows(self.h5_data['shot_line'], context_idx)
        ss_c = self._take_rows(self.h5_data['shot_stake'], context_idx)
        rl_c = self._take_rows(self.h5_data['recv_line'], context_idx)
        rs_c = self._take_rows(self.h5_data['recv_stake'], context_idx)
        patch_info = {
            'shot_line': np.concatenate([sl_q, sl_c])[_order],
            'shot_stake': np.concatenate([ss_q, ss_c])[_order],
            'recv_line': np.concatenate([rl_q, rl_c])[_order],
            'recv_stake': np.concatenate([rs_q, rs_c])[_order],
        }
        out = {
            'data': data_patch,
            'masked_patch': masked_patch,
            'rx_patch': coords_patch[:, 2].astype(np.float32, copy=False),
            'ry_patch': coords_patch[:, 3].astype(np.float32, copy=False),
            'sx_patch': coords_patch[:, 0].astype(np.float32, copy=False),
            'sy_patch': coords_patch[:, 1].astype(np.float32, copy=False),
            'time_axis_2d': self._time_axis_2d(data_patch.shape[0]),
            'std_val': std_val,
            'is_query': is_query,
            'query_count': np.int64(query_idx.size),
            'context_count': np.int64(context_idx.size),
            'grid_query_idx': query_idx.astype(np.int64, copy=False),
            'context_idx': context_idx.astype(np.int64, copy=False),
            'patch_info': patch_info,
            'data_raw': data_raw,
            'masked_patch_raw': masked_raw,
            **amplitude_metadata(thres),
        }
        if self.patch_meta.get("block_id") is not None:
            out['block_id'] = np.int64(np.asarray(self.patch_meta["block_id"])[idx])
        if self.patch_meta.get("block_center_grid_idx") is not None:
            out['block_center_grid_idx'] = np.int64(
                np.asarray(self.patch_meta["block_center_grid_idx"])[idx]
            )
        if self.patch_meta.get("anchor_grid_idx_list") is not None:
            out['anchor_grid_idx'] = self._index_row(self.patch_meta["anchor_grid_idx_list"], idx)
        return out
    
    def _normalize_coords(self, sx, sy, gx, gy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """归一化坐标到 [-1, 1]"""
        stats = self.coord_stats
        sx_n = 2 * (sx - stats['sx_min']) / (stats['sx_max'] - stats['sx_min']) - 1
        sy_n = 2 * (sy - stats['sy_min']) / (stats['sy_max'] - stats['sy_min']) - 1
        gx_n = 2 * (gx - stats['rx_min']) / (stats['rx_max'] - stats['rx_min']) - 1
        gy_n = 2 * (gy - stats['ry_min']) / (stats['ry_max'] - stats['ry_min']) - 1
        return sx_n, sy_n, gx_n, gy_n

    def compute_coord_stats(self, regular: bool = True):
        # 统计范围一律基于规则网格坐标（与 _normalize_coords 一致）；regular 仅保留接口兼容。
        #if regular:
        sx_all = np.clip(self.h5_data_regular['sx'], np.percentile(self.h5_data_regular['sx'], 0.5), np.percentile(self.h5_data_regular['sx'], 99.5))
        sy_all = np.clip(self.h5_data_regular['sy'], np.percentile(self.h5_data_regular['sy'], 0.5), np.percentile(self.h5_data_regular['sy'], 99.5))
        rx_all = np.clip(self.h5_data_regular['rx'], np.percentile(self.h5_data_regular['rx'], 0.5), np.percentile(self.h5_data_regular['rx'], 99.5))
        ry_all = np.clip(self.h5_data_regular['ry'], np.percentile(self.h5_data_regular['ry'], 0.5), np.percentile(self.h5_data_regular['ry'], 99.5))
        #else:
        #    sx_all = np.clip(self.h5_data['sx'], np.percentile(self.h5_data['sx'], 0.5), np.percentile(self.h5_data['sx'], 99.5))
        #    sy_all = np.clip(self.h5_data['sy'], np.percentile(self.h5_data['sy'], 0.5), np.percentile(self.h5_data['sy'], 99.5))
        #    rx_all = np.clip(self.h5_data['rx'], np.percentile(self.h5_data['rx'], 0.5), np.percentile(self.h5_data['rx'], 99.5))
        #    ry_all = np.clip(self.h5_data['ry'], np.percentile(self.h5_data['ry'], 0.5), np.percentile(self.h5_data['ry'], 99.5))
        dsx, sx_u = self.typical_grid_step(sx_all)
        dsy, sy_u = self.typical_grid_step(sy_all)
        drx, rx_u = self.typical_grid_step(rx_all)
        dry, ry_u = self.typical_grid_step(ry_all)

        # 用同一套范围（这里用 unique 的 min/max；也可以换成分位数）
        sx_min, sx_max = float(sx_u.min()), float(sx_u.max())
        sy_min, sy_max = float(sy_u.min()), float(sy_u.max())
        rx_min, rx_max = float(rx_u.min()), float(rx_u.max())
        ry_min, ry_max = float(ry_u.min()), float(ry_u.max())
        #print('sx_min, sx_max, sy_min, sy_max, rx_min, rx_max, ry_min, ry_max:',sx_min, sx_max, sy_min, sy_max, rx_min, rx_max, ry_min, ry_max)
        #print('dsx, dsy, drx, dry:',dsx, dsy, drx, dry)
        deltas = {}
        if dsx is not None and (sx_max - sx_min) > 0:
            deltas["sx"] = float((sx_max - sx_min)/(2*dsx))
        if dsy is not None and (sy_max - sy_min) > 0:
            deltas["sy"] = float((sy_max - sy_min)/(2*dsy))
        if drx is not None and (rx_max - rx_min) > 0:
            deltas["rx"] = float((rx_max - rx_min)/(2*drx))
        if dry is not None and (ry_max - ry_min) > 0:
            deltas["ry"] = float((ry_max - ry_min)/(2*dry))
        self.scale = deltas
        stats = {
            "sx_min": sx_all.min(), "sx_max": sx_all.max(),
            "sy_min": sy_all.min(), "sy_max": sy_all.max(),
            "rx_min": rx_all.min(), "rx_max": rx_all.max(),
            "ry_min": ry_all.min(), "ry_max": ry_all.max(),
        }
        
        stats["Lx"] = 0.5 * max(stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"])
        stats["Ly"] = 0.5 * max(stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"])

        # 在归一化前应用 p_scale：将 stats 的 min/max 乘以 p_scale
        if self.use_p_scale and self.scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.scale.get(name)
                if s is not None:
                    stats[f"{name}_min"] *= s
                    stats[f"{name}_max"] *= s
            print(f"[DatasetH5_all_queryctx] p_scale applied to coord_stats: {self.scale}")

        return stats

    def __getitem__(self, idx):
        # train_pool：与 self.train 无关，便于用 pool npz 做可视化/一致性检查
        if self.patch_mode == "train_pool":
            return self._build_train_query_context_sample(idx)

        if (not self.train) and self.patch_mode == "infer_query_context":
            return self._build_infer_query_context_sample(idx)

        raise NotImplementedError(
            f"DatasetH5_all_queryctx: 不支持 train={self.train!r}, patch_mode={self.patch_mode!r}"
        )

      


class DatasetH5_all(torch.utils.data.Dataset):

    def __init__(
        self,
        h5File=None,
        h5File_regular=None,
        h5File_tgt=None,
        dataset_neighbors=None,
        train=None,
        trace_sort_keys: Tuple[str, ...] = ("sx", "sy", "rx", "ry"),
        use_p_scale: bool = False,
    ):
        super().__init__()
        print('Loading dataset...')

        self.h5File = h5File
        self.h5File_regular = h5File_regular
        self.h5File_tgt = h5File_tgt
        self.dataset_neighbors = self._load_legacy_neighbors(dataset_neighbors)
        self.time_ps = args.time_ps
        self.trace_ps = args.trace_ps
        self.train = train
        self._rng = np.random.default_rng(123)
        self.std_val = None
        self.trace_sort_keys = tuple(trace_sort_keys)
        self.use_p_scale = use_p_scale

        self.dt_ms = 4
        self.t0_ms = 0
        self.scale = None

        h5 = File(self.h5File, "r")
        self.h5_data = {}
        for key in h5:
            node = h5[key]
            if hasattr(node, "keys") and "data" in node:
                break
        for key in node.keys():
            self.h5_data[key] = node[key]

        self.h5_data_regular = {}
        h5_regular = File(self.h5File_regular, "r")
        for key in h5_regular:
            node = h5_regular[key]
            if hasattr(node, "keys") and "data" in node:
                break
        for key in node.keys():
            self.h5_data_regular[key] = node[key]

        self.h5_data_tgt = {}
        h5_tgt = File(self.h5File_tgt, "r")
        for key in h5_tgt:
            node = h5_tgt[key]
            if hasattr(node, "keys") and "data" in node:
                break
        for key in node.keys():
            self.h5_data_tgt[key] = node[key]

        print(self.h5_data_regular["data"].shape)
        print(self.h5_data["data"].shape)
        print("loading data")

        self.coord_stats = self.compute_coord_stats(regular=not self.train)
        print('coord_stats computed')

    def _load_legacy_neighbors(self, path: Optional[str]) -> np.ndarray:
        if path is None:
            raise ValueError("dataset_neighbors is required")
        raw = np.load(path, allow_pickle=True)
        if hasattr(raw, "files"):
            if "0" in raw.files:
                arr = raw["0"]
            elif "patch_idx_2d" in raw.files:
                arr = raw["patch_idx_2d"]
            elif "pool_idx_2d" in raw.files:
                arr = raw["pool_idx_2d"]
            else:
                raw.close()
                raise ValueError(
                    "legacy DatasetH5_all expects ['0'] / patch_idx_2d / pool_idx_2d in dataset_neighbors"
                )
            raw.close()
            return np.asarray(arr)
        return np.asarray(raw)

    def _index_row(self, storage: np.ndarray, idx: int) -> np.ndarray:
        row = np.asarray(storage[idx], dtype=np.int64).reshape(-1)
        return row[row >= 0]

    def typical_grid_step(self, arr, eps=1e-9):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u
        d = np.diff(u)
        d = d[d > eps]
        if d.size == 0:
            return None, u
        return float(np.median(d)), u

    def __len__(self):
        return len(self.dataset_neighbors)

    def _normalize_coords(self, sx, sy, gx, gy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stats = self.coord_stats
        sx_n = 2 * (sx - stats['sx_min']) / (stats['sx_max'] - stats['sx_min']) - 1
        sy_n = 2 * (sy - stats['sy_min']) / (stats['sy_max'] - stats['sy_min']) - 1
        gx_n = 2 * (gx - stats['rx_min']) / (stats['rx_max'] - stats['rx_min']) - 1
        gy_n = 2 * (gy - stats['ry_min']) / (stats['ry_max'] - stats['ry_min']) - 1
        return sx_n, sy_n, gx_n, gy_n

    def compute_coord_stats(self, regular: bool = True):
        sx_all = np.clip(self.h5_data_regular['sx'], np.percentile(self.h5_data_regular['sx'], 0.5), np.percentile(self.h5_data_regular['sx'], 99.5))
        sy_all = np.clip(self.h5_data_regular['sy'], np.percentile(self.h5_data_regular['sy'], 0.5), np.percentile(self.h5_data_regular['sy'], 99.5))
        rx_all = np.clip(self.h5_data_regular['rx'], np.percentile(self.h5_data_regular['rx'], 0.5), np.percentile(self.h5_data_regular['rx'], 99.5))
        ry_all = np.clip(self.h5_data_regular['ry'], np.percentile(self.h5_data_regular['ry'], 0.5), np.percentile(self.h5_data_regular['ry'], 99.5))
        dsx, sx_u = self.typical_grid_step(sx_all)
        dsy, sy_u = self.typical_grid_step(sy_all)
        drx, rx_u = self.typical_grid_step(rx_all)
        dry, ry_u = self.typical_grid_step(ry_all)

        sx_min, sx_max = float(sx_u.min()), float(sx_u.max())
        sy_min, sy_max = float(sy_u.min()), float(sy_u.max())
        rx_min, rx_max = float(rx_u.min()), float(rx_u.max())
        ry_min, ry_max = float(ry_u.min()), float(ry_u.max())
        deltas = {}
        if dsx is not None and (sx_max - sx_min) > 0:
            deltas["sx"] = float((sx_max - sx_min) / (2 * dsx))
        if dsy is not None and (sy_max - sy_min) > 0:
            deltas["sy"] = float((sy_max - sy_min) / (2 * dsy))
        if drx is not None and (rx_max - rx_min) > 0:
            deltas["rx"] = float((rx_max - rx_min) / (2 * drx))
        if dry is not None and (ry_max - ry_min) > 0:
            deltas["ry"] = float((ry_max - ry_min) / (2 * dry))
        self.scale = deltas
        stats = {
            "sx_min": sx_all.min(), "sx_max": sx_all.max(),
            "sy_min": sy_all.min(), "sy_max": sy_all.max(),
            "rx_min": rx_all.min(), "rx_max": rx_all.max(),
            "ry_min": ry_all.min(), "ry_max": ry_all.max(),
        }
        stats["Lx"] = 0.5 * max(stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"])
        stats["Ly"] = 0.5 * max(stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"])

        # 在归一化前应用 p_scale：将 stats 的 min/max 乘以 p_scale
        if self.use_p_scale and self.scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.scale.get(name)
                if s is not None:
                    stats[f"{name}_min"] *= s
                    stats[f"{name}_max"] *= s
            print(f"[DatasetH5_all] p_scale applied to coord_stats: {self.scale}")

        return stats

    def __getitem__(self, idx):
        if self.train:
            np.random.seed(idx)
            expand = args.expand * np.random.rand()
            neighbor_idx = np.sort(self._index_row(self.dataset_neighbors, idx))
            data_full = self.h5_data['data'][neighbor_idx]
            rx_full = self.h5_data['rx'][neighbor_idx]
            ry_full = self.h5_data['ry'][neighbor_idx]
            sx_full = self.h5_data['sx'][neighbor_idx]
            sy_full = self.h5_data['sy'][neighbor_idx]

            trace_num_all, time_num_all = data_full.shape
            trace_num = int(min(trace_num_all, max(self.trace_ps, int(self.trace_ps * (1 + expand)))))
            if trace_num < self.trace_ps:
                traces = np.sort(
                    np.random.choice(np.arange(trace_num_all), self.trace_ps, replace=True)
                )
            else:
                trace_id_0 = np.random.randint(0, trace_num_all - trace_num + 1)
                traces = np.random.choice(np.arange(trace_num), self.trace_ps, replace=False)
                traces = np.sort(traces)
                traces = trace_id_0 + traces

            diff = time_num_all - self.time_ps
            if diff > 0:
                ori = data_full[traces, diff:]
            else:
                padding = diff * -1
                ori = np.pad(data_full[traces, :], ((0, 0), (padding, 0)), 'constant', constant_values=0)

            missing_ratio = sample_missing_ratio()
            masked_patch, mask_patch = apply_mixed_mask(ori, missing_ratio, block_prob=0.0)

            obs = masked_patch[mask_patch > 0]
            obs = obs[np.isfinite(obs)]
            std_val = np.float32(np.std(obs))
            std_val = np.float32(max(std_val, 1e-2))
            self.std_val = std_val
            thres = np.percentile(np.abs(masked_patch), 99.5)
            if thres == 0:
                thres = 1e-6
            masked_patch = np.clip(masked_patch, -thres, thres)
            masked_patch = masked_patch / thres
            data_patch = np.clip(ori, -thres, thres)
            data_patch = data_patch / thres

            rx_patch = rx_full[traces]
            ry_patch = ry_full[traces]
            sx_patch = sx_full[traces]
            sy_patch = sy_full[traces]
            sx_patch, sy_patch, rx_patch, ry_patch = self._normalize_coords(sx_patch, sy_patch, rx_patch, ry_patch)

            rx_patch = rx_patch.astype(np.float32)
            ry_patch = ry_patch.astype(np.float32)
            sx_patch = sx_patch.astype(np.float32)
            sy_patch = sy_patch.astype(np.float32)

            time_idx_1d = np.arange(0, self.time_ps, dtype=np.int32)
            time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
            time_axis_2d = np.tile(time_axis_1d[None, :], (self.trace_ps, 1))

            return {
                'data': data_patch.astype(np.float32),
                'masked_patch': masked_patch.astype(np.float32),
                'rx_patch': rx_patch,
                'ry_patch': ry_patch,
                'sx_patch': sx_patch,
                'sy_patch': sy_patch,
                'time_axis_2d': time_axis_2d.astype(np.float32),
                'std_val': self.std_val,
                **amplitude_metadata(thres),
            }

        neighbor_idx = np.sort(self._index_row(self.dataset_neighbors, idx))
        data_full = self.h5_data_regular['data'][neighbor_idx]
        data_tgt = self.h5_data_tgt['data'][neighbor_idx]
        rx_full = self.h5_data_regular['rx'][neighbor_idx]
        ry_full = self.h5_data_regular['ry'][neighbor_idx]
        sx_full = self.h5_data_regular['sx'][neighbor_idx]
        sy_full = self.h5_data_regular['sy'][neighbor_idx]
        trace_num_all, time_num_all = data_full.shape
        patch_info = {
            'shot_line': self.h5_data_regular['shot_line'][neighbor_idx],
            'shot_stake': self.h5_data_regular['shot_stake'][neighbor_idx],
            'recv_line': self.h5_data_regular['recv_line'][neighbor_idx],
            'recv_stake': self.h5_data_regular['recv_stake'][neighbor_idx],
        }
        diff = time_num_all - self.time_ps
        if diff > 0:
            ori = data_full[:, diff:]
            tgt = data_tgt[:, diff:]
        else:
            padding = diff * -1
            ori = np.pad(data_full[:, :], ((0, 0), (padding, 0)), 'constant', constant_values=0)
            tgt = np.pad(data_tgt[:, :], ((0, 0), (padding, 0)), 'constant', constant_values=0)
        # Sort by trace_sort_keys (default: sx, sy, rx, ry)
        if self.trace_sort_keys:
            _cols = {"sx": sx_full, "sy": sy_full, "rx": rx_full, "ry": ry_full}
            order = np.lexsort([_cols[k] for k in reversed(self.trace_sort_keys)])
        else:
            order = np.arange(ori.shape[0])
        ori = ori[order]
        tgt = tgt[order]
        rx_full = rx_full[order]
        ry_full = ry_full[order]
        sx_full = sx_full[order]
        sy_full = sy_full[order]
        patch_info['shot_line'] = patch_info['shot_line'][order]
        patch_info['shot_stake'] = patch_info['shot_stake'][order]
        patch_info['recv_line'] = patch_info['recv_line'][order]
        patch_info['recv_stake'] = patch_info['recv_stake'][order]
        data_raw = ori.astype(np.float32, copy=True)
        masked_raw = tgt.astype(np.float32, copy=True)
        obs = masked_raw[np.isfinite(masked_raw)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else np.float32(0.0)
        std_val = np.float32(max(std_val, 1e-2))
        thres = np.percentile(np.abs(obs), 99.5) if obs.size > 0 else 1e-6
        thres = float(max(thres, 1e-6))
        ori = np.clip(ori, -thres, thres) / thres
        tgt = np.clip(tgt, -thres, thres) / thres
        rx_full, ry_full, sx_full, sy_full = self._normalize_coords(sx_full, sy_full, rx_full, ry_full)
        return {
            'data': ori.astype(np.float32),
            'masked_patch': tgt.astype(np.float32),
            'rx_patch': rx_full,
            'ry_patch': ry_full,
            'sx_patch': sx_full,
            'sy_patch': sy_full,
            'patch_info': patch_info,
            'std_val': std_val,
            'data_raw': data_raw,
            'masked_patch_raw': masked_raw,
            **amplitude_metadata(thres),
        }


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    try:
        from tqdm import trange as _trange
    except ImportError:
        _trange = range

    dataset = DatasetH5_all_queryctx(
        h5File='../h5/dongfang/raw5d_data1104.h5',
        h5File_regular='../h5/dongfang/reg5dbin_label1031.h5',
        h5File_tgt='../h5/dongfang/reg5dbin_label1031_binning.h5',
        dataset_neighbors='../h5/dongfang/patchV2/infer_query_context.npz',
        train=False,
    )
    sample = dataset[0]
    if sample is None:
        raise RuntimeError(
            "dataset[0] 为 None：请检查 neighbors npz 与 train/patch_mode 是否匹配 "
            "(例如 train_pool 需 train=True；推理 infer 需 infer_query_context npz)。"
        )

    data = np.asarray(sample["data"])
    masked_patch = np.asarray(sample["masked_patch"])
    rx_patch = np.asarray(sample["rx_patch"])
    ry_patch = np.asarray(sample["ry_patch"])
    sx_patch = np.asarray(sample["sx_patch"])
    sy_patch = np.asarray(sample["sy_patch"])

    # DatasetH5_all_queryctx：train_pool / infer_query_context 均返回 is_query 与 Q/K 计数
    has_query_context = "is_query" in sample
    if has_query_context:
        is_query = np.asarray(sample["is_query"], dtype=bool)
        q = int(sample["query_count"]) if "query_count" in sample else int(np.count_nonzero(is_query))
        k = int(sample["context_count"]) if "context_count" in sample else int(data.shape[0] - q)
        t1_left, t1_mid, t1_right = (
            "GT patch (scaled)",
            "Input (query 道置零)",
            "被置零的 query 能量 (GT − input)",
        )
        t2_left, t2_order = "Query/Context 几何", f"排列顺序 (Q={q}, K={k})"
    else:
        # 如 DatasetH5_all：无 query 概念；data 与 masked_patch 为配对通道（规则网格 vs 缺失/目标等）
        is_query = np.zeros((data.shape[0],), dtype=bool)
        q = k = 0
        t1_left, t1_mid, t1_right = "Patch A (data)", "Patch B (masked_patch)", "差分 (data − masked_patch)"
        t2_left, t2_order = "炮检点（归一化）", f"道序 (N={data.shape[0]})"

    patch_mode = getattr(dataset, "patch_mode", None)
    print(
        "sample:",
        f"patch_mode={patch_mode!r}," if patch_mode is not None else "",
        "data=",
        data.shape,
        "masked=",
        masked_patch.shape,
        "coords=",
        rx_patch.shape,
        "has_query_context=",
        has_query_context,
        "Q=",
        q,
        "K=",
        k,
        "std_val=",
        sample.get("std_val", "—"),
    )

    # 图1：与数据集一致 — query 模式为 GT / 遮挡输入 / 差分；否则为配对两路 + 差分
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    vmax = float(
        max(np.abs(data).max(), np.abs(masked_patch).max(), 1e-6)
    )
    diff = data - masked_patch
    vmax = max(vmax, float(np.abs(diff).max()), 1e-6)

    im0 = axes1[0].imshow(data.T, vmin=-vmax, vmax=vmax, aspect="auto", cmap="seismic")
    axes1[0].set_title(t1_left)
    axes1[0].set_xlabel("Trace")
    axes1[0].set_ylabel("Time sample")
    plt.colorbar(im0, ax=axes1[0], shrink=0.7)

    im1 = axes1[1].imshow(masked_patch.T, vmin=-vmax, vmax=vmax, aspect="auto", cmap="seismic")
    axes1[1].set_title(t1_mid)
    axes1[1].set_xlabel("Trace")
    axes1[1].set_ylabel("Time sample")
    plt.colorbar(im1, ax=axes1[1], shrink=0.7)

    im2 = axes1[2].imshow(diff.T, vmin=-vmax, vmax=vmax, aspect="auto", cmap="seismic")
    axes1[2].set_title(t1_right)
    axes1[2].set_xlabel("Trace")
    axes1[2].set_ylabel("Time sample")
    plt.colorbar(im2, ax=axes1[2], shrink=0.7)

    fig1.tight_layout()
    fig1.savefig("./test_data.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved ./test_data.png")

    # 图2：有 is_query 时区分 query/context；否则只画统一炮检点
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    if has_query_context and np.any(is_query):
        ctx = ~is_query
        axes2[0].scatter(rx_patch[ctx], ry_patch[ctx], c="C0", s=2, alpha=0.6, label="Context recv")
        axes2[0].scatter(rx_patch[is_query], ry_patch[is_query], c="C3", s=8, alpha=0.9, label="Query recv")
        axes2[0].scatter(sx_patch[ctx], sy_patch[ctx], c="C1", s=2, alpha=0.4, marker="s", label="Context src")
        axes2[0].scatter(sx_patch[is_query], sy_patch[is_query], c="C4", s=10, alpha=0.9, marker="s", label="Query src")
        axes2[1].scatter(np.arange(data.shape[0])[ctx], np.zeros(np.count_nonzero(ctx)), c="C0", s=4, alpha=0.7, label="Context")
        axes2[1].scatter(
            np.arange(data.shape[0])[is_query],
            np.zeros(np.count_nonzero(is_query)),
            c="C3",
            s=10,
            alpha=0.9,
            label="Query",
        )
        axes2[1].set_title(t2_order)
    else:
        axes2[0].scatter(rx_patch, ry_patch, c="C0", s=3, alpha=0.7, label="Recv")
        axes2[0].scatter(sx_patch, sy_patch, c="C1", s=3, alpha=0.5, marker="s", label="Src")
        n = data.shape[0]
        axes2[1].scatter(np.arange(n), np.zeros(n), c="C2", s=6, alpha=0.8, label="Traces")
        axes2[1].set_title(t2_order)

    axes2[0].set_xlabel("X (normalized)")
    axes2[0].set_ylabel("Y (normalized)")
    axes2[0].set_title(t2_left)
    axes2[0].legend(loc="best", fontsize=8)
    axes2[0].set_aspect("equal", adjustable="box")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_yticks([])
    axes2[1].set_xlabel("Patch order")
    axes2[1].set_ylabel("")
    axes2[1].legend(loc="best", fontsize=8)
    axes2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("./test_data_coords.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved ./test_data_coords.png")
