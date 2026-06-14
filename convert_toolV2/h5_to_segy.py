#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5-to-SEGY converter: reads trace data from an H5 file, copies trace headers
from a template SEG-Y file, matches traces by geometry keys defined in
segy_schema.py, and writes a new SEG-Y with template headers + H5 data.

Output is placed in the same directory as the template SEG-Y by default.

Usage:
    python h5_to_segy.py \
        --h5-file /data/shared/测试数据/h5/field1031_test_aligned.h5 \
        --template-segy /data/shared/测试数据/mask_from_label.sgy \
        --h5-group 1551 \
        --profile field1031 \
        --strict
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import argparse
import shutil
import struct
import numpy as np
import h5py
import segyio

from segy_schema import (
    SegyProfile,
    get_segy_profile,
    profile_names,
)

# ---------------------------------------------------------------------------
# Low-level helpers (from gen_infer.py)
# ---------------------------------------------------------------------------


def i32be(buf: bytes, pos_1b: int) -> int:
    """Read big-endian int32 from byte buffer at 1-based position."""
    return struct.unpack(">i", buf[pos_1b - 1: pos_1b + 3])[0]


def bytes_per_sample(fmt: int) -> int:
    """Map SEG-Y format code to bytes per sample."""
    if fmt in (1, 2, 5):
        return 4
    if fmt == 3:
        return 2
    return 1  # fmt 8


def scale_coord(v: int, scalar_raw: int) -> int:
    """Apply coordinate scalar to a raw header value."""
    if scalar_raw == 0:
        return int(round(v))
    if scalar_raw > 0:
        return int(round(float(v) * float(scalar_raw)))
    return int(round(float(v) / float(-scalar_raw)))


def fit_trace(trace: np.ndarray, ns: int) -> np.ndarray:
    """Crop or pad a trace to match target_ns samples."""
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    if trace.size > ns:
        return trace[trace.size - ns:]
    if trace.size < ns:
        return np.pad(trace, (ns - trace.size, 0)).astype(np.float32)
    return trace


# ---------------------------------------------------------------------------
# Step 1: Read H5 data and geometry keys
# ---------------------------------------------------------------------------

def read_h5_data(
    h5_path: str, group_name: str, profile: SegyProfile
) -> dict:
    """Read trace data and geometry keys from H5 file.

    Returns
    -------
    dict with keys:
        data      : np.ndarray (N, T) float32 — trace amplitudes
        keys      : np.ndarray (N, K) int64    — geometry key per trace
        ns        : int                        — number of time samples
        ntraces   : int                        — number of traces
        delta     : np.ndarray (N,) float32
        t0        : np.ndarray (N,) float32
    """
    print(f"[H5] Opening: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        # --- validate group exists ---
        if group_name not in f:
            available = list(f.keys())
            raise ValueError(
                f"H5 group '{group_name}' not found. Available: {available}"
            )
        g = f[group_name]

        # --- validate data exists ---
        if "data" not in g:
            raise ValueError(
                f"H5 group '{group_name}' missing required dataset 'data'. "
                f"Found: {list(g.keys())}"
            )
        data = np.asarray(g["data"], dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"H5 'data' must be 2D (ntraces, nsamples), got shape {data.shape}"
            )
        ntraces, ns = data.shape
        print(f"[H5] data shape: {ntraces} traces x {ns} samples")

        # --- build geometry key matrix ---
        key_cols = []
        for k in profile.key_columns:
            if k in g:
                col = np.asarray(g[k], dtype=np.int64).ravel()
            else:
                fallback = profile.h5_fallback.get(k)
                if fallback and fallback in g:
                    print(f"[H5] key '{k}' not found, using fallback '{fallback}'")
                    col = np.asarray(g[fallback], dtype=np.int64).ravel()
                else:
                    raise ValueError(
                        f"H5 group '{group_name}' missing key column '{k}' "
                        f"(and no fallback '{fallback}' found). "
                        f"Available datasets: {sorted(g.keys())}"
                    )
            if col.size != ntraces:
                raise ValueError(
                    f"H5 key column '{k}' has {col.size} entries, "
                    f"expected {ntraces} (matching data.shape[0])."
                )
            key_cols.append(col)
        keys = np.column_stack(key_cols).astype(np.int64)
        print(f"[H5] key columns: {list(profile.key_columns)}, keys shape: {keys.shape}")

        # --- read optional scalar fields ---
        delta = (
            np.asarray(g["delta"], dtype=np.float32).ravel()
            if "delta" in g
            else np.full(ntraces, np.nan, dtype=np.float32)
        )
        t0 = (
            np.asarray(g["t0"], dtype=np.float32).ravel()
            if "t0" in g
            else np.full(ntraces, np.nan, dtype=np.float32)
        )

        # --- validate no NaN/Inf in data ---
        if np.any(~np.isfinite(data)):
            n_bad = np.sum(~np.isfinite(data))
            raise ValueError(f"H5 'data' contains {n_bad} non-finite values (NaN/Inf).")

    return {
        "data": data,
        "keys": keys,
        "ns": ns,
        "ntraces": ntraces,
        "delta": delta,
        "t0": t0,
    }


# ---------------------------------------------------------------------------
# Step 2: Read SEG-Y template headers and data
# ---------------------------------------------------------------------------

def read_segy_headers_and_data(
    template_path: str, mode: str, profile: SegyProfile
) -> dict:
    """Parse SEG-Y trace headers and read trace data from template file.

    Returns
    -------
    dict with keys:
        headers  : list[dict]  — [{trace_idx, key, coords, ns}, ...]
        ns       : int         — number of samples (from binary header)
        ntraces  : int         — total trace count
        data     : np.ndarray  — (N, T) float32 trace data
        bps      : int         — bytes per sample
        fmt      : int         — SEG-Y format code
    """
    print(f"[SEGY] Reading template: {template_path}")
    byte_pos = profile.byte_pos
    key_columns = profile.key_columns

    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template SEG-Y not found: {template_path}")

    headers = []
    with open(template_path, "rb") as f:
        # --- binary reel header ---
        f.seek(3200)
        bin_header = f.read(400)
        if len(bin_header) < 400:
            raise RuntimeError(
                f"Template SEG-Y binary header too short: {len(bin_header)} bytes"
            )
        ns_bin = struct.unpack(">H", bin_header[20:22])[0]
        fmt_code = struct.unpack(">H", bin_header[24:26])[0]
        bps = bytes_per_sample(fmt_code)
        if ns_bin <= 0:
            raise ValueError(f"Invalid ns={ns_bin} in SEG-Y binary header (bytes 3220-3221)")

        # --- trace headers ---
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
                # self_computed: derive line/stake from scaled coordinates
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

    if trace_idx == 0:
        raise RuntimeError("No traces found in template SEG-Y.")

    # --- read trace data via segyio for verification ---
    with segyio.open(template_path, "r", strict=False, ignore_geometry=True) as f_segy:
        segy_data = f_segy.trace.raw[:].astype(np.float32)
        ntraces_segyio = f_segy.tracecount

    if ntraces_segyio != trace_idx:
        raise RuntimeError(
            f"SEG-Y trace count mismatch: raw parse={trace_idx}, segyio={ntraces_segyio}"
        )

    print(f"[SEGY] {trace_idx} traces, ns={ns_bin}, fmt={fmt_code}, bps={bps}")
    print(f"[SEGY] mode={mode}, key_columns={key_columns}")

    # --- validate ns consistency ---
    ns_values = set(h["ns"] for h in headers)
    if len(ns_values) > 1:
        print(f"[SEGY] WARNING: inconsistent per-trace ns: {ns_values}")

    return {
        "headers": headers,
        "ns": ns_bin,
        "ntraces": trace_idx,
        "data": segy_data,
        "bps": bps,
        "fmt": fmt_code,
    }


# ---------------------------------------------------------------------------
# Step 3: Build geometry-key lookup between H5 and SEG-Y
# ---------------------------------------------------------------------------

def build_geometry_lookup(
    h5_keys: np.ndarray, segy_headers: list, profile: SegyProfile
) -> dict:
    """Match H5 traces to SEG-Y traces by geometry key.

    Returns
    -------
    dict with keys:
        h5_to_segy          : {h5_idx: [segy_idx, ...]}
        segy_to_h5          : {segy_idx: [h5_idx, ...]}
        unmatched_h5        : [h5_idx, ...]
        unmatched_segy      : [segy_idx, ...]
        duplicate_keys_h5   : {key: count} for keys appearing > once in H5
        duplicate_keys_segy : {key: count} for keys appearing > once in SEG-Y
    """
    from collections import defaultdict

    # --- SEG-Y side: key -> [trace_idx, ...] ---
    segy_key_to_indices: dict = defaultdict(list)
    for h in segy_headers:
        segy_key_to_indices[h["key"]].append(int(h["trace_idx"]))

    # --- H5 side: key -> [trace_idx, ...] ---
    h5_key_to_indices: dict = defaultdict(list)
    for i, row in enumerate(h5_keys):
        key = tuple(int(v) for v in row)
        h5_key_to_indices[key].append(i)

    # --- detect duplicates ---
    duplicate_keys_h5 = {k: len(v) for k, v in h5_key_to_indices.items() if len(v) > 1}
    duplicate_keys_segy = {k: len(v) for k, v in segy_key_to_indices.items() if len(v) > 1}

    # --- match ---
    h5_to_segy: dict = {}
    segy_to_h5: dict = defaultdict(list)
    unmatched_h5: list = []
    unmatched_segy: list = []

    # H5 -> SEGY direction
    for h5_idx in range(len(h5_keys)):
        key = tuple(int(v) for v in h5_keys[h5_idx])
        segy_indices = segy_key_to_indices.get(key, [])
        if segy_indices:
            h5_to_segy[h5_idx] = list(segy_indices)
        else:
            unmatched_h5.append(h5_idx)

    # SEGY -> H5 direction
    for h in segy_headers:
        segy_idx = int(h["trace_idx"])
        key = h["key"]
        h5_indices = h5_key_to_indices.get(key, [])
        if h5_indices:
            segy_to_h5[segy_idx] = list(h5_indices)
        else:
            unmatched_segy.append(segy_idx)

    # --- summary ---
    n_matched_keys = len(h5_key_to_indices.keys() & segy_key_to_indices.keys())
    total_keys_h5 = len(h5_key_to_indices)
    total_keys_segy = len(segy_key_to_indices)

    print(f"[MATCH] Unique keys — H5: {total_keys_h5}, SEG-Y: {total_keys_segy}, "
          f"matched: {n_matched_keys}")
    print(f"[MATCH] Traces — H5: {len(h5_keys)}, SEG-Y: {len(segy_headers)}")
    print(f"[MATCH] Matched H5 traces: {len(h5_to_segy)}, unmatched: {len(unmatched_h5)}")
    print(f"[MATCH] Matched SEG-Y traces: {len(segy_to_h5)}, unmatched: {len(unmatched_segy)}")
    if duplicate_keys_h5:
        print(f"[MATCH] WARNING: {len(duplicate_keys_h5)} keys appear multiple times in H5 "
              f"(max multiplicity: {max(duplicate_keys_h5.values())})")
    if duplicate_keys_segy:
        print(f"[MATCH] INFO: {len(duplicate_keys_segy)} keys appear multiple times in SEG-Y "
              f"(max multiplicity: {max(duplicate_keys_segy.values())})")

    if n_matched_keys == 0:
        raise RuntimeError(
            "Zero geometry keys matched between H5 and SEG-Y. "
            "Check that --profile matches the data, or that key columns are correct."
        )

    return {
        "h5_to_segy": h5_to_segy,
        "segy_to_h5": dict(segy_to_h5),
        "unmatched_h5": unmatched_h5,
        "unmatched_segy": unmatched_segy,
        "duplicate_keys_h5": duplicate_keys_h5,
        "duplicate_keys_segy": duplicate_keys_segy,
    }


# ---------------------------------------------------------------------------
# Step 4: Assemble output data array in SEG-Y trace order
# ---------------------------------------------------------------------------

def assemble_output_data(
    h5_data: dict, segy_result: dict, lookup: dict, strict: bool = False
) -> np.ndarray:
    """Build (N_segy, ns) output array in template SEG-Y trace order.

    For 1:N mappings, H5 data is averaged across all H5 sources.
    For N:1 mappings, all matching SEG-Y traces get the same (averaged) data.
    All traces are fitted to the target ns via fit_trace().
    """
    n_segy = segy_result["ntraces"]
    target_ns = segy_result["ns"]
    h5_ns = h5_data["ns"]
    output = np.zeros((n_segy, target_ns), dtype=np.float32)
    h5_data_arr = h5_data["data"]
    segy_to_h5 = lookup["segy_to_h5"]

    ns_mismatch = h5_ns != target_ns
    if ns_mismatch:
        print(f"[ASSEMBLE] ns mismatch: H5={h5_ns}, SEG-Y={target_ns}. "
              f"Will fit traces (truncate from beginning or left-pad).")

    for segy_idx in range(n_segy):
        h5_indices = segy_to_h5.get(segy_idx, [])
        if not h5_indices:
            # unmatched — leave as zeros
            continue
        # average if multiple H5 sources for this key
        trace = np.mean(h5_data_arr[h5_indices], axis=0)
        if ns_mismatch:
            trace = fit_trace(trace, target_ns)
        output[segy_idx] = trace.astype(np.float32)

    # --- validate ---
    if np.any(~np.isfinite(output)):
        n_bad = np.sum(~np.isfinite(output))
        raise ValueError(f"Output data contains {n_bad} non-finite values after assembly.")

    n_matched = len(segy_to_h5)
    n_unmatched = n_segy - n_matched
    print(f"[ASSEMBLE] output shape: {output.shape}, matched={n_matched}, "
          f"unmatched (zero-filled)={n_unmatched}")

    return output


# ---------------------------------------------------------------------------
# Step 5: Write output SEG-Y (copy template headers, overwrite data)
# ---------------------------------------------------------------------------

def write_segy_output(
    template_path: str, output_path: str, data: np.ndarray
) -> None:
    """Copy template SEG-Y and overwrite trace sample data.

    Preserves all headers (3600-byte reel header + 240-byte trace headers)
    byte-for-byte from the template. Only trace sample data is overwritten.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[WRITE] Copying template: {template_path} -> {output_path}")
    shutil.copy2(template_path, output_path)

    print(f"[WRITE] Opening output for trace data overwrite...")
    with segyio.open(output_path, "r+", strict=False, ignore_geometry=True) as f:
        if f.tracecount != data.shape[0]:
            raise ValueError(
                f"Output SEG-Y tracecount={f.tracecount}, but data has {data.shape[0]} traces."
            )
        for i in range(f.tracecount):
            f.trace[i] = data[i].astype(np.float32)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[WRITE] Done. Output: {output_path} ({file_size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 6: Verify output SEG-Y against template SEG-Y by geometry key
# ---------------------------------------------------------------------------


def verify_against_template(
    output_path: str,
    segy_result: dict,
    profile: SegyProfile,
    header_mode: str,
    max_report: int = 50,
) -> dict:
    """Compare new SEG-Y against template SEG-Y trace-by-trace via geometry keys.

    Reads both SEG-Y files, matches traces by profile.key_columns, and compares
    trace data sample-by-sample. Reports exact-match count and details of
    differing traces sorted by key.

    Parameters
    ----------
    output_path : str
        Path to the newly written SEG-Y file.
    segy_result : dict
        Pre-loaded template SEG-Y data from read_segy_headers_and_data().
    profile : SegyProfile
        SEG-Y profile defining key_columns and byte_pos.
    header_mode : str
        "fixed" or "self_computed".
    max_report : int
        Maximum number of differing traces to print in detail.

    Returns
    -------
    dict with keys:
        n_total       : int — total traces in output
        n_exact_match : int — traces with all samples identical
        n_different   : int — traces with any sample difference
        n_key_mismatch: int — traces whose key doesn't match template
        max_diff      : float — maximum absolute sample difference
        max_reldiff   : float — maximum relative sample difference
        mean_diff     : float — mean absolute difference over all samples
        diff_details  : list[dict] — per-trace diff info for differing traces
                        (key, template_idx, output_idx, max_abs_diff, n_diff_samples)
    """
    byte_pos = profile.byte_pos
    key_columns = profile.key_columns

    print(f"[VERIFY] Comparing output SEG-Y against template SEG-Y")
    print(f"[VERIFY]   Output: {output_path}")

    # --- Reuse already-loaded template data from step 2 ---
    tpl_headers = segy_result["headers"]
    tpl_data = segy_result["data"]
    n_total = segy_result["ntraces"]
    ns = segy_result["ns"]

    # --- Read output SEG-Y headers and data ---
    out_headers = []
    with open(output_path, "rb") as f:
        f.seek(3200)
        bin_hdr = f.read(400)
        ns_out = struct.unpack(">H", bin_hdr[20:22])[0]
        fmt_out = struct.unpack(">H", bin_hdr[24:26])[0]
        bps_out = bytes_per_sample(fmt_out)
        f.seek(3600)
        tidx = 0
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            if header_mode == "fixed":
                key = tuple(i32be(hdr, byte_pos[k]) for k in key_columns)
            else:
                sx = i32be(hdr, byte_pos["shot_x"])
                sy = i32be(hdr, byte_pos["shot_y"])
                rx = i32be(hdr, byte_pos["rec_x"])
                ry = i32be(hdr, byte_pos["rec_y"])
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
            ns_trace = struct.unpack(">H", hdr[114:116])[0] or ns_out
            out_headers.append({
                "trace_idx": tidx,
                "key": tuple(int(x) for x in key),
                "ns": int(ns_trace),
            })
            f.seek(int(ns_trace) * bps_out, os.SEEK_CUR)
            tidx += 1

    with segyio.open(output_path, "r", strict=False, ignore_geometry=True) as f:
        out_data = f.trace.raw[:].astype(np.float32)

    if out_data.shape[0] != n_total:
        raise RuntimeError(
            f"Trace count mismatch: output {out_data.shape[0]}, template {n_total}"
        )

    # --- Build template key -> trace_idx lookup ---
    from collections import defaultdict

    tpl_key_to_idx: dict = {}
    tpl_multi: dict = defaultdict(list)
    for h in tpl_headers:
        key = h["key"]
        if key in tpl_key_to_idx:
            tpl_multi[key].append(h["trace_idx"])
        else:
            tpl_key_to_idx[key] = h["trace_idx"]
            if key in tpl_multi:
                tpl_multi[key].append(tpl_key_to_idx[key])
                del tpl_key_to_idx[key]

    # --- Compare trace by trace ---
    n_exact_match = 0
    n_different = 0
    n_key_mismatch = 0
    max_diff = 0.0
    max_reldiff = 0.0
    total_diff = 0.0
    diff_details = []

    for out_idx in range(n_total):
        out_key = out_headers[out_idx]["key"]
        tpl_idx = tpl_key_to_idx.get(out_key)

        if tpl_idx is None:
            # key not found in template — check multi-key
            found = False
            for k, indices in tpl_multi.items():
                if k == out_key:
                    tpl_idx = indices[0]
                    found = True
                    break
            if not found:
                n_key_mismatch += 1
                continue

        out_trace = out_data[out_idx].astype(np.float32)
        tpl_trace = tpl_data[tpl_idx].astype(np.float32)
        abs_diff = np.abs(out_trace - tpl_trace)

        trace_max = float(np.max(abs_diff))
        max_diff = max(max_diff, trace_max)
        safe_tpl = np.maximum(np.abs(tpl_trace), 1e-10)
        max_reldiff = max(max_reldiff, float(np.max(abs_diff / safe_tpl)))
        total_diff += float(np.sum(abs_diff))

        if trace_max == 0.0:
            n_exact_match += 1
        else:
            n_different += 1
            n_diff_samples = int(np.sum(abs_diff > 0))
            if len(diff_details) < max_report:
                diff_details.append({
                    "key": out_key,
                    "key_names": list(key_columns),
                    "template_idx": tpl_idx,
                    "output_idx": out_idx,
                    "max_abs_diff": trace_max,
                    "n_diff_samples": n_diff_samples,
                    "total_samples": ns,
                })

    mean_diff = total_diff / (n_total * ns) if n_total > 0 else 0.0

    # --- Print results ---
    print(f"[VERIFY] Total traces:              {n_total}")
    print(f"[VERIFY] Exact match (diff=0):       {n_exact_match}")
    print(f"[VERIFY] Different (diff>0):         {n_different}")
    print(f"[VERIFY] Key mismatch:              {n_key_mismatch}")
    print(f"[VERIFY] Max abs diff:              {max_diff:.6e}")
    print(f"[VERIFY] Max rel diff:              {max_reldiff:.6e}")
    print(f"[VERIFY] Mean abs diff:             {mean_diff:.6e}")

    # --- Print differing traces sorted by key ---
    if diff_details:
        # sort by key columns
        diff_details.sort(key=lambda d: d["key"])
        print(f"\n[VERIFY] Differing traces (first {min(len(diff_details), max_report)} "
              f"of {n_different}, sorted by key):")
        print(f"[VERIFY] {'Key':<30s} {'tpl_idx':>8s} {'out_idx':>8s} "
              f"{'max_diff':>12s} {'diff_samples':>14s}")
        print(f"[VERIFY] {'-'*30} {'-'*8} {'-'*8} {'-'*12} {'-'*14}")
        for d in diff_details[:max_report]:
            key_str = ",".join(str(v) for v in d["key"])
            print(f"[VERIFY] {key_str:<30s} {d['template_idx']:>8d} {d['output_idx']:>8d} "
                  f"{d['max_abs_diff']:>12.6e} {d['n_diff_samples']:>14d}")

    return {
        "n_total": n_total,
        "n_exact_match": n_exact_match,
        "n_different": n_different,
        "n_key_mismatch": n_key_mismatch,
        "max_diff": max_diff,
        "max_reldiff": max_reldiff,
        "mean_diff": mean_diff,
        "diff_details": diff_details,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H5-to-SEGY: copy template headers, fill with H5 trace data "
                    "matched by geometry keys from segy_schema."
    )
    parser.add_argument(
        "--h5-file", required=True,
        help="Path to input H5 file containing trace data and geometry keys.",
    )
    parser.add_argument(
        "--template-segy", required=True,
        help="Path to template SEG-Y file providing trace headers.",
    )
    parser.add_argument(
        "--h5-group", default="1551",
        help="H5 group name within the file (default: '1551').",
    )
    parser.add_argument(
        "--profile", default=None,
        help="SEG-Y profile name (sw06, field1031, segc3). "
             "Default: read from H5 group attr 'segy_profile', fallback to 'sw06'.",
    )
    parser.add_argument(
        "--header-mode", default=None, choices=["fixed", "self_computed"],
        help="Header parsing mode. Default: profile's default_header_mode.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output SEG-Y path. Default: <template_dir>/<stem>_from_h5.sgy",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat unmatched traces as fatal error.",
    )
    parser.add_argument(
        "--max-report", type=int, default=50,
        help="Maximum number of differing traces to print in detail (default: 50).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    # --- resolve profile ---
    if args.profile is not None:
        profile = get_segy_profile(args.profile)
    else:
        # try to read from H5 attribute
        try:
            with h5py.File(args.h5_file, "r") as f:
                profile_name = f[args.h5_group].attrs.get("segy_profile", "sw06")
        except Exception:
            profile_name = "sw06"
        profile = get_segy_profile(profile_name)

    header_mode = args.header_mode or profile.default_header_mode

    # --- resolve output path ---
    template_dir = os.path.dirname(os.path.abspath(args.template_segy))
    template_stem = Path(args.template_segy).stem
    output_path = args.output or os.path.join(template_dir, f"{template_stem}_from_h5.sgy")

    print("=" * 72)
    print("  H5-to-SEGY Conversion")
    print("=" * 72)
    print(f"  H5 file:       {args.h5_file}")
    print(f"  H5 group:      {args.h5_group}")
    print(f"  Template SEG-Y: {args.template_segy}")
    print(f"  Output:        {output_path}")
    print(f"  Profile:       {profile.name} ({profile.description})")
    print(f"  Key columns:   {profile.key_columns}")
    print(f"  Header mode:   {header_mode}")
    print(f"  Strict:        {args.strict}")
    print(f"  Max report:    {args.max_report}")
    print("=" * 72)

    # ---- Step 1: Read H5 ----
    print("\n>>> Step 1/6: Reading H5 data...")
    h5_result = read_h5_data(args.h5_file, args.h5_group, profile)

    # ---- Step 2: Read SEG-Y template ----
    print("\n>>> Step 2/6: Reading template SEG-Y headers and data...")
    segy_result = read_segy_headers_and_data(
        args.template_segy, header_mode, profile
    )

    # ---- Step 3: Build geometry lookup ----
    print("\n>>> Step 3/6: Building geometry-key lookup...")
    lookup = build_geometry_lookup(h5_result["keys"], segy_result["headers"], profile)

    if args.strict and (lookup["unmatched_h5"] or lookup["unmatched_segy"]):
        raise RuntimeError(
            f"Strict mode: {len(lookup['unmatched_h5'])} unmatched H5 traces, "
            f"{len(lookup['unmatched_segy'])} unmatched SEG-Y traces. Aborting."
        )

    # ---- Step 4: Assemble output data ----
    print("\n>>> Step 4/6: Assembling output data array...")
    output_data = assemble_output_data(
        h5_result, segy_result, lookup, strict=args.strict
    )

    # ---- Step 5: Write output SEG-Y ----
    print("\n>>> Step 5/6: Writing output SEG-Y...")
    write_segy_output(args.template_segy, output_path, output_data)

    # ---- Step 6: Verify ----
    print("\n>>> Step 6/6: Comparing new SEG-Y against template SEG-Y...")
    verify_result = verify_against_template(
        output_path, segy_result, profile, header_mode,
        max_report=args.max_report,
    )

    # ---- Final summary ----
    print("\n" + "=" * 72)
    print("  Conversion Complete")
    print("=" * 72)
    print(f"  Output:            {output_path}")
    print(f"  Total traces:      {verify_result['n_total']}")
    print(f"  Exact match:       {verify_result['n_exact_match']} (fully identical)")
    print(f"  Different:         {verify_result['n_different']}")
    print(f"  Key mismatch:      {verify_result['n_key_mismatch']}")
    print(f"  Max abs diff:      {verify_result['max_diff']:.6e}")
    print(f"  Max rel diff:      {verify_result['max_reldiff']:.6e}")
    print(f"  Mean abs diff:     {verify_result['mean_diff']:.6e}")
    print("=" * 72)

    return 0 if verify_result["n_key_mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
