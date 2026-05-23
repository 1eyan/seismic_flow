from pathlib import Path
import argparse
import h5py as h5
import numpy as np
import segyio
import time
import dataset_config
import os
import sys
import struct
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segy_schema import DEFAULT_PROFILE_NAME, SegyProfile, get_segy_profile, parse_sort_keys, profile_names


def get_traces_idx(cfg):
    return np.load(os.path.join(os.path.dirname(info_h5), f"{info_h5.split('/')[-1].split('.')[0]}_info", f"kept_trace_indices_{cfg['domain']}_{cfg['keep_ratio']}.npy"))

DEFAULT_PROFILE = get_segy_profile(getattr(dataset_config, "segy_profile", DEFAULT_PROFILE_NAME))
BYTE_POS = dict(DEFAULT_PROFILE.byte_pos)
SORT_KEYS = list(parse_sort_keys(getattr(dataset_config, "sort_keys", None), DEFAULT_PROFILE))


def _resolve_profile(profile=None) -> SegyProfile:
    if profile is None:
        return DEFAULT_PROFILE
    if isinstance(profile, SegyProfile):
        return profile
    return get_segy_profile(str(profile))


# === 辅助函数 ===
def _read_bin_header_format_and_ns(f):
    f.seek(3200, 0)
    binhdr = f.read(400)
    ns_from_bin = struct.unpack('>H', binhdr[20:22])[0]
    fmt_code    = struct.unpack('>H', binhdr[24:26])[0]
    return fmt_code, ns_from_bin

def _bps_from_fmt(fmt: int) -> int:
    if fmt in (1, 2, 5): return 4
    if fmt == 3: return 2
    if fmt == 8: return 1
    return 4

def _scale_coords(values, scalars):
    v = np.asarray(values, dtype=np.float64)
    if scalars is None:
        return v
    s = np.asarray(scalars, dtype=np.int64)
    out = v.astype(np.float64, copy=True)
    pos = s > 0
    neg = s < 0
    out[pos] = out[pos] * s[pos]
    out[neg] = out[neg] / np.abs(s[neg])
    return out

# === sgy-->headers-->pandas dataframe ===
## 道头文件中如果自带炮线炮桩测线检波点
def read_headers_pure_python_fixed(path: Path, profile=None):
    profile = _resolve_profile(profile)
    byte_pos = profile.byte_pos
    out = {'trace': []}
    for key in byte_pos.keys():
        out[key] = []
    with open(path, 'rb') as f:
        fmt, ns_bin = _read_bin_header_format_and_ns(f)
        bps = _bps_from_fmt(fmt)
        f.seek(3600, 0)
        t = 0
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            def i32be(pos1b):
                j0 = pos1b - 1
                return struct.unpack('>i', hdr[j0:j0+4])[0]
            out['trace'].append(t)
            for key, pos in byte_pos.items():
                out[key].append(i32be(pos))
            ns = struct.unpack('>H', hdr[114:116])[0] or ns_bin
            f.seek(ns * bps, 1)
            t += 1
    return pd.DataFrame(out)

def read_headers_pure_self_computed(path: Path, profile=None):
    profile = _resolve_profile(profile)
    byte_pos = profile.byte_pos
    out = {'trace': [],
    'shot_x': [],
    'shot_y': [],
    'rec_x': [],
    'rec_y': [],
    }
    with open(path, 'rb') as f:
        fmt, ns_bin = _read_bin_header_format_and_ns(f)
        bps = _bps_from_fmt(fmt)
        f.seek(3600, 0)
        t = 0
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            def i32be(pos1b):
                j0 = pos1b - 1
                return struct.unpack('>i', hdr[j0:j0+4])[0]
            out['trace'].append(t)
            out['shot_x'].append(i32be(byte_pos['shot_x']))
            out['shot_y'].append(i32be(byte_pos['shot_y']))
            out['rec_x'].append(i32be(byte_pos['rec_x']))
            out['rec_y'].append(i32be(byte_pos['rec_y']))
            ns = struct.unpack('>H', hdr[114:116])[0] or ns_bin
            f.seek(ns * bps, 1)
            t += 1
    #print(len(out['trace']),len(out['shot_x']),len(out['shot_y']),len(out['rec_x']),len(out['rec_y']))
    return pd.DataFrame(out)

def organize_traces(input_segy, headers_df=None, sort_keys=None, mode='self_computed', profile=None):
    """
    按 headers dataframe 的排序键重排地震道。
    - headers_df 为 None: 自动从 SEG-Y 读取道头并按 sort_keys 排序
    - headers_df 不为 None: 使用传入 dataframe（必须包含 trace 列），再按 sort_keys 排序
    返回:
      {
        'headers': 排序后的 dataframe,
        'data':    排序后的地震道矩阵,
        'sx','sy','rx','ry','inline','crossline','delta','t0': 对齐后的属性
      }
    """
    profile = _resolve_profile(profile)
    sort_keys = list(parse_sort_keys(sort_keys, profile))

    with segyio.open(input_segy, ignore_geometry=True) as f:
        data = f.trace.raw[:]
        scalar = np.abs(f.attributes(segyio.TraceField.SourceGroupScalar)[:].astype(np.float32))
        delta = f.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[:].astype(np.float32) / 1000.0
        t0 = f.attributes(segyio.TraceField.DelayRecordingTime)[:].astype(np.float32) / 1000.0
    
    if headers_df is not None:
        headers = headers_df.copy()
    elif mode == 'fixed':
        headers = read_headers_pure_python_fixed(Path(input_segy), profile=profile)
    elif mode == 'self_computed':
        headers = read_headers_pure_self_computed(Path(input_segy), profile=profile)
    else:
        raise ValueError(f"mode 必须为 'fixed' 或 'self_computed'")
    if 'trace' not in headers.columns:
        raise ValueError("headers_df 必须包含 'trace' 列")
    if mode == 'self_computed':
        trace_for_scalar = headers['trace'].to_numpy(dtype=np.intp)
        scalar_for_headers = scalar[trace_for_scalar].copy()
        scalar_for_headers[scalar_for_headers == 0] = 1.0
        if 'shot_line' not in headers:
            headers['shot_line'] = pd.Series(
                np.rint(_scale_coords(headers['shot_y'].to_numpy(dtype=np.float32), scalar_for_headers)),
                dtype="Int64",
            )
        if 'shot_no' not in headers:
            headers['shot_no'] = pd.Series(
                np.rint(_scale_coords(headers['shot_x'].to_numpy(dtype=np.float32), scalar_for_headers)),
                dtype="Int64",
            )
        if 'recv_line' not in headers:
            headers['recv_line'] = pd.Series(
                np.rint(_scale_coords(headers['rec_y'].to_numpy(dtype=np.float32), scalar_for_headers)),
                dtype="Int64",
            )
        if 'recv_no' not in headers:
            headers['recv_no'] = pd.Series(
                np.rint(_scale_coords(headers['rec_x'].to_numpy(dtype=np.float32), scalar_for_headers)),
                dtype="Int64",
            )
        if 'shot_stake' not in headers:
            headers['shot_stake'] = headers['shot_no']
        if 'recv_stake' not in headers:
            headers['recv_stake'] = headers['recv_no']
    trace_idx_old = headers['trace'].to_numpy(dtype=np.intp)
    missing_sort = [k for k in sort_keys if k not in headers.columns]
    if missing_sort:
        raise ValueError(
            f"sort_keys contain fields not present in headers for profile={profile.name}, mode={mode}: {missing_sort}"
        )
    headers = headers.sort_values(by=sort_keys).reset_index(drop=True)
    trace_idx = headers['trace'].to_numpy(dtype=np.intp)
    if np.all(trace_idx == trace_idx_old):
        print('sort_headers is not needed')
    else:
        print('sort_headers is needed')
    n_traces = data.shape[0]
    if trace_idx.min() < 0 or trace_idx.max() >= n_traces:
        raise ValueError(f"trace 索引越界，合法范围是 [0, {n_traces - 1}]")

    scalar = scalar[trace_idx]
    scalar[scalar == 0] = 1.0
    if mode == 'fixed':
        out = {
            'data': data[trace_idx],
            'sx': headers['shot_x'].to_numpy(dtype=np.float32) / scalar,
            'sy': headers['shot_y'].to_numpy(dtype=np.float32) / scalar,
            'rx': headers['rec_x'].to_numpy(dtype=np.float32) / scalar,
            'ry': headers['rec_y'].to_numpy(dtype=np.float32) / scalar,
            'delta':delta[trace_idx],
            't0': t0[trace_idx],
            'shot_line': headers['shot_line'].to_numpy(dtype=np.int32),
            'shot_no': headers['shot_no'].to_numpy(dtype=np.int32),
            'recv_line': headers['recv_line'].to_numpy(dtype=np.int32),
            'recv_no': headers['recv_no'].to_numpy(dtype=np.int32),
            'shot_stake': headers['shot_stake'].to_numpy(dtype=np.int32),
            'recv_stake': headers['recv_stake'].to_numpy(dtype=np.int32),
            'cmp': headers['cmp'].to_numpy(dtype=np.int32),
            'cmp_line': headers['cmp_line'].to_numpy(dtype=np.int32),
            'offset': headers['offset'].to_numpy(dtype=np.int32),
            'trace_idx': headers['trace'].to_numpy(dtype=np.int32),
        }
    elif mode == 'self_computed':
        out = {
            'data': data[trace_idx],
            'sx': _scale_coords(headers['shot_x'].to_numpy(dtype=np.float32), scalar),
            'sy': _scale_coords(headers['shot_y'].to_numpy(dtype=np.float32), scalar),
            'rx': _scale_coords(headers['rec_x'].to_numpy(dtype=np.float32), scalar),
            'ry': _scale_coords(headers['rec_y'].to_numpy(dtype=np.float32), scalar),
            'sx_original': headers['shot_x'].to_numpy(dtype=np.float32),
            'sy_original': headers['shot_y'].to_numpy(dtype=np.float32),
            'rx_original': headers['rec_x'].to_numpy(dtype=np.float32),
            'ry_original': headers['rec_y'].to_numpy(dtype=np.float32),
            'delta':delta[trace_idx],
            't0': t0[trace_idx],
            'shot_line': pd.Series(np.rint(_scale_coords(headers['shot_y'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'shot_no': pd.Series(np.rint(_scale_coords(headers['shot_x'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'recv_line': pd.Series(np.rint(_scale_coords(headers['rec_y'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'recv_no': pd.Series(np.rint(_scale_coords(headers['rec_x'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'trace_idx': headers['trace'].to_numpy(dtype=np.int32),
        }
    else:
        raise ValueError(f"mode 必须为 'fixed' 或 'self_computed'")
    return out

def compute_ovt_fields(sx, sy, rx, ry,
                        mx_bin=None, my_bin=None,
                        hx_bin=None, hy_bin=None,
                        mx_origin=None, my_origin=None,
                        hx_origin=None, hy_origin=None):
    """
    从炮点、检波点坐标计算 OVT 域字段。

    Parameters
    ----------
    sx, sy, rx, ry : array-like
        炮点和检波点坐标（已缩放）。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        各维 bin size（米），None 则自动从数据范围估计。
    mx_origin, my_origin, hx_origin, hy_origin : float, optional
        各维起始参考点，None 则对齐到数据最小值。

    Returns
    -------
    dict with keys:
        mx, my, hx, hy            — continuous midpoint / half-offset
        imx, imy, ihx, ihy       — binned integer indices
        mx_center, my_center, hx_center, hy_center — bin centers
        mx_bin, my_bin, hx_bin, hy_bin — actual bin sizes used
        mx_origin, my_origin, hx_origin, hy_origin — actual origins used
        fold                     — 每条 trace 所在 OVT cell 的 fold
    """
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)

    mx = 0.5 * (sx + rx)
    my = 0.5 * (sy + ry)
    hx = 0.5 * (rx - sx)
    hy = 0.5 * (ry - sy)

    def _auto_bin(arr, n_div=100, min_val=1.0):
        r = arr.max() - arr.min()
        return max(r / n_div, min_val) if r > 0 else float(min_val)

    if mx_bin is None:
        mx_bin = _auto_bin(mx)
    if my_bin is None:
        my_bin = _auto_bin(my)
    if hx_bin is None:
        hx_bin = _auto_bin(hx, n_div=50, min_val=0.5)
    if hy_bin is None:
        hy_bin = _auto_bin(hy, n_div=50, min_val=0.5)

    if mx_origin is None:
        mx_origin = mx.min()
    if my_origin is None:
        my_origin = my.min()
    if hx_origin is None:
        hx_origin = hx.min()
    if hy_origin is None:
        hy_origin = hy.min()

    imx = np.floor((mx - mx_origin) / mx_bin).astype(np.int64)
    imy = np.floor((my - my_origin) / my_bin).astype(np.int64)
    ihx = np.floor((hx - hx_origin) / hx_bin).astype(np.int64)
    ihy = np.floor((hy - hy_origin) / hy_bin).astype(np.int64)

    mx_center = mx_origin + (imx.astype(np.float64) + 0.5) * mx_bin
    my_center = my_origin + (imy.astype(np.float64) + 0.5) * my_bin
    hx_center = hx_origin + (ihx.astype(np.float64) + 0.5) * hx_bin
    hy_center = hy_origin + (ihy.astype(np.float64) + 0.5) * hy_bin

    # 计算 fold：每个 OVT cell 有多少条 trace
    keys = np.column_stack((imx, imy, ihx, ihy))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    fold = np.zeros(len(imx), dtype=np.int32)
    for i, k in enumerate(unique_keys):
        mask = (keys[:, 0] == k[0]) & (keys[:, 1] == k[1]) & \
               (keys[:, 2] == k[2]) & (keys[:, 3] == k[3])
        cnt = mask.sum()
        fold[mask] = cnt

    return {
        'mx': mx.astype(np.float32),
        'my': my.astype(np.float32),
        'hx': hx.astype(np.float32),
        'hy': hy.astype(np.float32),
        'imx': imx.astype(np.int32),
        'imy': imy.astype(np.int32),
        'ihx': ihx.astype(np.int32),
        'ihy': ihy.astype(np.int32),
        'mx_center': mx_center.astype(np.float32),
        'my_center': my_center.astype(np.float32),
        'hx_center': hx_center.astype(np.float32),
        'hy_center': hy_center.astype(np.float32),
        'mx_bin': float(mx_bin),
        'my_bin': float(my_bin),
        'hx_bin': float(hx_bin),
        'hy_bin': float(hy_bin),
        'mx_origin': float(mx_origin),
        'my_origin': float(my_origin),
        'hx_origin': float(hx_origin),
        'hy_origin': float(hy_origin),
        'fold': fold,
    }


def add_ovt_to_h5(h5_file, group_name='1551',
                  mx_bin=None, my_bin=None,
                  hx_bin=None, hy_bin=None):
    """
    读取已有 H5（由 segy2h5 生成），计算 OVT 字段并写入同一 H5 文件。

    Parameters
    ----------
    h5_file : str
        H5 文件路径。
    group_name : str
        H5 中的 group 名。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        分箱 bin size，None 则自动估计。
    """
    with h5.File(h5_file, 'r+') as h5f:
        g = h5f[group_name]
        sx = g['sx'][:]
        sy = g['sy'][:]
        rx = g['rx'][:]
        ry = g['ry'][:]

        ovt = compute_ovt_fields(
            sx, sy, rx, ry,
            mx_bin=mx_bin, my_bin=my_bin,
            hx_bin=hx_bin, hy_bin=hy_bin,
        )

        for key, val in ovt.items():
            if key not in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                           'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
                g.create_dataset(key, data=val, compression='gzip')

        # bin 参数作为 group 属性存储
        for key in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                    'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
            g.attrs[key] = ovt[key]

    print(f"[add_ovt_to_h5] OVT fields written to {h5_file}/{group_name}")


def segy2h5(h5_file, input_segy, group_name='1551', headers_df=None, sort_keys=None, mode='self_computed',
            compute_ovt=False, mx_bin=None, my_bin=None, hx_bin=None, hy_bin=None, profile=None):
    """
    单个 SEG-Y 落盘到 H5，按 sort_keys 组织地震道。

    Parameters
    ----------
    h5_file : str
        输出 H5 路径。
    input_segy : str
        输入 SEG-Y 路径。
    group_name : str
        H5 group 名。
    headers_df : pd.DataFrame, optional
        预读取的道头。
    sort_keys : list
        排序键。
    mode : str
        'self_computed' 或 'fixed'。
    compute_ovt : bool
        若为 True，计算并写入 OVT 字段（mx, my, hx, hy, imx, imy, ihx, ihy, fold）。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        OVT 分箱 bin size。
    """
    profile = _resolve_profile(profile)
    effective_sort_keys = parse_sort_keys(sort_keys, profile)
    block = organize_traces(input_segy, headers_df=headers_df, sort_keys=effective_sort_keys, mode=mode, profile=profile)
    with h5.File(h5_file, 'w') as h5f:
        g = h5f.create_group(group_name)
        g.attrs['segy_profile'] = profile.name
        g.attrs['segy_mode'] = mode
        g.attrs['sort_keys'] = ','.join(effective_sort_keys)
        g.create_dataset('data', data=block['data'], compression='gzip')
        g.create_dataset('sx', data=block['sx'], compression='gzip')
        g.create_dataset('sy', data=block['sy'], compression='gzip')
        g.create_dataset('rx', data=block['rx'], compression='gzip')
        g.create_dataset('ry', data=block['ry'], compression='gzip')
        g.create_dataset('delta', data=block['delta'], compression='gzip')
        g.create_dataset('t0', data=block['t0'], compression='gzip')
        g.create_dataset('shot_line', data=block['shot_line'], compression='gzip')
        g.create_dataset('shot_no', data=block['shot_no'], compression='gzip')
        g.create_dataset('recv_line', data=block['recv_line'], compression='gzip')
        g.create_dataset('recv_no', data=block['recv_no'], compression='gzip')
        g.create_dataset('trace_idx', data=block['trace_idx'], compression='gzip')
        if mode == 'fixed':
            g.create_dataset('shot_stake', data=block['shot_stake'], compression='gzip')
            g.create_dataset('recv_stake', data=block['recv_stake'], compression='gzip')
            g.create_dataset('cmp', data=block['cmp'], compression='gzip')
            g.create_dataset('cmp_line', data=block['cmp_line'], compression='gzip')
            g.create_dataset('offset', data=block['offset'], compression='gzip')
        elif mode == 'self_computed':
            g.create_dataset('sx_original', data=block['sx_original'], compression='gzip')
            g.create_dataset('sy_original', data=block['sy_original'], compression='gzip')
            g.create_dataset('rx_original', data=block['rx_original'], compression='gzip')
            g.create_dataset('ry_original', data=block['ry_original'], compression='gzip')
        else:
            raise ValueError(f"mode 必须为 'fixed' 或 'self_computed'")

        if compute_ovt:
            sx = block['sx'] if isinstance(block['sx'], np.ndarray) else block['sx'].to_numpy()
            sy = block['sy'] if isinstance(block['sy'], np.ndarray) else block['sy'].to_numpy()
            rx = block['rx'] if isinstance(block['rx'], np.ndarray) else block['rx'].to_numpy()
            ry = block['ry'] if isinstance(block['ry'], np.ndarray) else block['ry'].to_numpy()
            ovt = compute_ovt_fields(
                sx, sy, rx, ry,
                mx_bin=mx_bin, my_bin=my_bin,
                hx_bin=hx_bin, hy_bin=hy_bin,
            )
            for key, val in ovt.items():
                if key not in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                               'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
                    g.create_dataset(key, data=val, compression='gzip')
            for key in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                        'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
                g.attrs[key] = ovt[key]
        h5f.close()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single SEG-Y -> H5 converter")
    parser.add_argument("--segy-profile", choices=profile_names(),
                        default=getattr(dataset_config, "segy_profile", DEFAULT_PROFILE_NAME))
    parser.add_argument("--mode", choices=["fixed", "self_computed"],
                        default=getattr(dataset_config, "segy_mode", "fixed"))
    parser.add_argument("--sort-keys", default=None,
                        help="Comma-separated sort keys; defaults to the active SEG-Y profile")
    parser.add_argument("--info-h5", default=dataset_config.info_h5)
    parser.add_argument("--input-segy", default=None)
    parser.add_argument("--group-name", default=None)
    cli_args = parser.parse_args()

    profile = get_segy_profile(cli_args.segy_profile)
    sort_keys = parse_sort_keys(
        cli_args.sort_keys if cli_args.sort_keys is not None else getattr(dataset_config, "sort_keys", None),
        profile,
    )
    mode = cli_args.mode
    info_h5 = cli_args.info_h5
    segyPairs = dataset_config.segyPairs
    os.makedirs(os.path.dirname(info_h5) or '.', exist_ok=True)

    first_key = next(iter(segyPairs.keys()))
    group_name = cli_args.group_name or first_key
    input_segy = cli_args.input_segy or segyPairs[first_key][0]
    headers_df = (
        read_headers_pure_python_fixed(Path(input_segy), profile=profile)
        if mode == 'fixed'
        else read_headers_pure_self_computed(Path(input_segy), profile=profile)
    )
    print(headers_df.head())
    print(f"segy_profile={profile.name} mode={mode} sort_keys={sort_keys}")
    s_time = time.time()
    segy2h5(
        h5_file=info_h5,
        input_segy=input_segy,
        headers_df=headers_df,
        group_name=group_name,
        mode=mode,
        sort_keys=sort_keys,
        profile=profile,
    )
    f_time = time.time()
    print(f"cost time: {f_time - s_time:.2f}")
