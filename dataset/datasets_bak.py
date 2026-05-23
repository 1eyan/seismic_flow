import datetime
from typing import Tuple
import torch
import numpy as np
import math
from h5py import File
from config import args

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



class DatasetH5_all(torch.utils.data.Dataset):

    def __init__(
        self,
        h5File=None,
        h5File_regular=None,
        dataset_neighbors=None,
    ):
        super().__init__()
        print('Loading dataset...')

        self.h5File = h5File 
        self.h5File_regular = h5File_regular 
        self.dataset_neighbors = np.load(dataset_neighbors, allow_pickle=True)['0']
        #self.dataset_neighbors = np.load(args.dataset_neighbors, allow_pickle=True)['0']
        self.time_ps = args.time_ps
        self.trace_ps = args.trace_ps
        self.train = args.train
        self._rng = np.random.default_rng(123)
        self.std_val = None

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
        print(self.h5_data_regular["data"].shape)
        print(self.h5_data["data"].shape)
        print("loading data")
        
        self.coord_stats = self.compute_coord_stats()
        print('coord_stats computed')

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
        return len(self.dataset_neighbors)
    
    def _normalize_coords(self, sx, sy, gx, gy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """归一化坐标到 [-1, 1]"""
        stats = self.coord_stats
        sx_n = 2 * (sx - stats['sx_min']) / (stats['sx_max'] - stats['sx_min']) - 1
        sy_n = 2 * (sy - stats['sy_min']) / (stats['sy_max'] - stats['sy_min']) - 1
        gx_n = 2 * (gx - stats['rx_min']) / (stats['rx_max'] - stats['rx_min']) - 1
        gy_n = 2 * (gy - stats['ry_min']) / (stats['ry_max'] - stats['ry_min']) - 1
        return sx_n, sy_n, gx_n, gy_n

    def compute_coord_stats(self,):
        
        sx_all = np.clip(self.h5_data_regular['sx'], np.percentile(self.h5_data_regular['sx'], 0.5), np.percentile(self.h5_data_regular['sx'], 99.5))
        sy_all = np.clip(self.h5_data_regular['sy'], np.percentile(self.h5_data_regular['sy'], 0.5), np.percentile(self.h5_data_regular['sy'], 99.5))
        rx_all = np.clip(self.h5_data_regular['rx'], np.percentile(self.h5_data_regular['rx'], 0.5), np.percentile(self.h5_data_regular['rx'], 99.5))
        ry_all = np.clip(self.h5_data_regular['ry'], np.percentile(self.h5_data_regular['ry'], 0.5), np.percentile(self.h5_data_regular['ry'], 99.5))
        
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
        
        return stats
    
    def __getitem__(self, idx):
        if self.train:
            np.random.seed(idx)
            expand = args.expand * np.random.rand()
            neighbor_idx = np.sort(self.dataset_neighbors[idx])
            data_full = self.h5_data['data'][neighbor_idx]
            rx_full = self.h5_data['rx'][neighbor_idx]
            ry_full = self.h5_data['ry'][neighbor_idx]
            sx_full = self.h5_data['sx'][neighbor_idx]
            sy_full = self.h5_data['sy'][neighbor_idx]

            trace_num_all, time_num_all = data_full.shape
            #print(f"trace_num_all: {trace_num_all}, time_num_all: {time_num_all}")
            trace_num = int(min(trace_num_all, self.trace_ps * (1 + expand)))
            trace_id_0 = np.random.randint(0, trace_num_all - trace_num + 1)
            traces = np.random.choice(np.arange(trace_num), self.trace_ps, replace=False)
            traces = np.sort(traces)
            traces = trace_id_0 + traces
            
            diff = time_num_all - self.time_ps
            if diff > 0:  #直接截断
               ori = data_full[traces, diff:]
            else:
                padding= diff * -1 #padding 0 到前面
                ori = np.pad(data_full[traces, :], ((0, 0), (padding, 0)), 'constant', constant_values=0)
            
            missing_ratio = sample_missing_ratio()
            masked_patch, mask_patch = apply_mixed_mask(ori, missing_ratio, block_prob=0.0)
            
            # ========== 归一化：使用 masked_patch 的标准差（与推理一致）==========
            obs = masked_patch[mask_patch == 0]  # 只用观测点
            obs = obs[np.isfinite(obs)]
            std_val  = np.float32(np.std(obs))
            std_val  = np.float32(max(std_val, 1e-2))
            self.std_val = std_val
            thres = np.percentile(np.abs(masked_patch), 99.5)
            if thres == 0:
                thres = 1e-6
            masked_patch =np.clip(masked_patch, -thres, thres)
            masked_patch = masked_patch / thres
            data_patch = np.clip(ori, -thres, thres)
            data_patch = data_patch / thres
            # ====================================================================

            rx_patch = rx_full[traces]
            ry_patch = ry_full[traces]
            sx_patch = sx_full[traces]
            sy_patch = sy_full[traces]
            
            '''rx_patch, ry_patch, sx_patch, sy_patch = _augment_coords(
                rx_patch, ry_patch, sx_patch, sy_patch,
                jitter=0.05, rot_scale=True
            )'''
            theta = np.random.rand() * np.pi * 2
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rx = rx_patch * cos_t - ry_patch * sin_t
            ry = rx_patch * sin_t + ry_patch * cos_t
            sx = sx_patch * cos_t - sy_patch * sin_t
            sy = sx_patch * sin_t + sy_patch * cos_t

            randomD = np.random.uniform(0.8, 1.2)
            rx = rx * randomD
            ry = ry * randomD
            sx = sx * randomD
            sy = sy * randomD

            sx_patch, sy_patch, rx_patch, ry_patch = self._normalize_coords(sx_patch, sy_patch, rx_patch, ry_patch)
            
            rx_patch = rx_patch.astype(np.float32)
            ry_patch = ry_patch.astype(np.float32)
            sx_patch = sx_patch.astype(np.float32)
            sy_patch = sy_patch.astype(np.float32)
            
            # 时间轴
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
            }
        else:
            raise ValueError("Invalid mode")


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dataset = DatasetH5_all(h5File='/NAS/czt/mount/seis_flow_data12V2/h5_/dongfang_real_raw.h5', h5File_regular='/NAS/czt/mount/seis_flow_data12V2/h5_/dongfang_mask_from_label.h5', dataset_neighbors='/NAS/czt/mount/seis_flow_data12V2/h5_/dongfang_real_raw_train_index_5d_line_by_order.pkl')
    print(dataset[0]['data'].shape)
    data = dataset[0]['data']
    masked_patch = dataset[0]['masked_patch']
    rx_patch = dataset[0]['rx_patch']*dataset.scale['rx']
    ry_patch = dataset[0]['ry_patch']*dataset.scale['ry']
    sx_patch = dataset[0]['sx_patch']*dataset.scale['sx']
    sy_patch = dataset[0]['sy_patch']*dataset.scale['sy']
    time_axis_2d = dataset[0]['time_axis_2d']
    std_val = dataset[0]['std_val']
    print(data.shape, masked_patch.shape, rx_patch.shape, ry_patch.shape, sx_patch.shape, sy_patch.shape, time_axis_2d.shape, std_val)
    # 图1：完整数据、mask 数据（含缺失）、两者作差
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    vmax = max(np.abs(data).max(), 1e-6)
    im0 = axes1[0].imshow(data.T, vmin=-vmax, vmax=vmax, aspect="auto", cmap="seismic")
    axes1[0].set_title("Full data")
    axes1[0].set_xlabel("Trace")
    axes1[0].set_ylabel("Time sample")
    plt.colorbar(im0, ax=axes1[0], shrink=0.7)

    im1 = axes1[1].imshow(masked_patch.T, vmin=-vmax, vmax=vmax, aspect="auto", cmap="seismic")
    axes1[1].set_title("Masked data (observed)")
    axes1[1].set_xlabel("Trace")
    axes1[1].set_ylabel("Time sample")
    plt.colorbar(im1, ax=axes1[1], shrink=0.7)

    diff = data - masked_patch
    im2 = axes1[2].imshow(diff.T, vmin=-vmax, vmax=vmax, aspect="auto", cmap="seismic")
    axes1[2].set_title("Difference (full − masked)")
    axes1[2].set_xlabel("Trace")
    axes1[2].set_ylabel("Time sample")
    plt.colorbar(im2, ax=axes1[2], shrink=0.7)

    fig1.tight_layout()
    fig1.savefig("./test_data.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved ./test_data.png")

    # 图2：patch 坐标散点图（检波点 + 炮点）
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    axes2[0].scatter(rx_patch, ry_patch, c="C0", s=1, alpha=0.8, label="Receiver (rx, ry)")
    axes2[0].scatter(sx_patch, sy_patch, c="C1", s=1, alpha=0.8, marker="s", label="Source (sx, sy)")
    axes2[0].set_xlabel("X (normalized)")
    axes2[0].set_ylabel("Y (normalized)")
    axes2[0].set_title("Patch: receivers and sources")
    axes2[0].legend(loc="best", fontsize=8)
    axes2[0].set_aspect("equal", adjustable="box")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].scatter(rx_patch, ry_patch, c="C0", s=1, alpha=0.8)
    axes2[1].set_xlabel("Receiver X")
    axes2[1].set_ylabel("Receiver Y")
    axes2[1].set_title("Receivers only (rx, ry)")
    axes2[1].set_aspect("equal", adjustable="box")
    axes2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("./test_data_coords.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved ./test_data_coords.png")