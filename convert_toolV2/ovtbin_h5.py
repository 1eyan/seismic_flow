#!/usr/bin/env python3
"""Compute grid-driven Cartesian OVTBin fields and store them in H5.

This utility keeps OVT numbering stable by letting a regular/label H5 define
the grid. Sparse irregular input can be projected back onto that full geometry
before writing OVTBin fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segy_schema import DEFAULT_PROFILE_NAME, get_segy_profile, profile_names


ALGORITHM_VERSION = "ovtbin-h5-grid-cartesian-v1"
COORDINATE_TYPE = "cartesian"
OVT_FIELD_NAMES = (
    "offset_x",
    "offset_y",
    "offset",
    "azimuth",
    "offset_x_bin",
    "offset_y_bin",
    "offset_x_center",
    "offset_y_center",
    "ovt",
    "ovt_fold",
)
MIDPOINT_FIELD_NAMES = (
    "midpoint_x",
    "midpoint_y",
    "midpoint_x_bin",
    "midpoint_y_bin",
)
PROJECTION_FIELD_NAMES = ("trace_mask", "projection_fold")
CELL_KEY_FIELDS = (
    "midpoint_x_bin",
    "midpoint_y_bin",
    "offset_x_bin",
    "offset_y_bin",
)
GRID_SPEC_VERSION = 1


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required for H5 input/output.") from exc
    return h5py


def _normalize_wave_type(wave_type: str) -> str:
    wave = str(wave_type).upper()
    if wave not in {"PP", "PS"}:
        raise ValueError(f"wave_type must be 'PP' or 'PS', got {wave_type!r}")
    return wave


def _normalize_binning_mode(binning_mode: str) -> str:
    mode = str(binning_mode).lower()
    if mode not in {"expert", "beginner"}:
        raise ValueError(f"binning_mode must be 'expert' or 'beginner', got {binning_mode!r}")
    return mode


def _positive_float(name: str, value: Optional[float]) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def resolve_cartesian_bin_sizes(
    *,
    wave_type: str,
    binning_mode: str,
    offset_x_bin_size: Optional[float] = None,
    offset_y_bin_size: Optional[float] = None,
    source_line_interval: Optional[float] = None,
    receiver_line_interval: Optional[float] = None,
    gamma: float = 2.0,
) -> Tuple[float, float]:
    wave = _normalize_wave_type(wave_type)
    mode = _normalize_binning_mode(binning_mode)
    if mode == "expert":
        return (
            _positive_float("offset_x_bin_size", offset_x_bin_size),
            _positive_float("offset_y_bin_size", offset_y_bin_size),
        )

    source_interval = _positive_float("source_line_interval", source_line_interval)
    receiver_interval = _positive_float("receiver_line_interval", receiver_line_interval)
    if wave == "PP":
        return 2.0 * source_interval, 2.0 * receiver_interval

    gamma = _positive_float("gamma", gamma)
    factor = (1.0 + gamma) / gamma
    return factor * source_interval, factor * receiver_interval


def _validate_shift(name: str, value: float, bin_size: float) -> float:
    value = float(value)
    half = 0.5 * float(bin_size)
    if value < -half or value > half:
        raise ValueError(f"{name} must be within [-{half}, {half}], got {value}")
    return value


def _as_int32(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.size and (
        values.min() < np.iinfo(np.int32).min
        or values.max() > np.iinfo(np.int32).max
    ):
        raise OverflowError(f"{name} is outside int32 range")
    return values.astype(np.int32)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    arr = np.asarray(value)
    if arr.dtype == object:
        try:
            arr = arr.astype(np.int64)
        except (TypeError, ValueError):
            arr = arr.astype(str)
    return arr


def _dataset_kwargs(data: np.ndarray) -> Dict[str, object]:
    if data.shape == ():
        return {}
    return {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def _write_dataset(group, name: str, data, *, overwrite: bool) -> None:
    if name in group:
        if not overwrite:
            raise ValueError(
                f"Dataset {group.name}/{name} already exists. "
                "Use --overwrite-fields to replace OVT fields."
            )
        del group[name]
    arr = _to_numpy(data)
    group.create_dataset(name, data=arr, **_dataset_kwargs(arr))


def _copy_dataset(group, name: str, data) -> None:
    arr = _to_numpy(data)
    group.create_dataset(name, data=arr, **_dataset_kwargs(arr))


def _load_h5_coordinates(group) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [key for key in ("sx", "sy", "rx", "ry") if key not in group]
    if missing:
        raise ValueError(f"H5 group {group.name} is missing datasets: {missing}")
    return group["sx"][:], group["sy"][:], group["rx"][:], group["ry"][:]


def _parse_key_columns(value: Optional[str], profile_name: str) -> Tuple[str, ...]:
    if value:
        cols = tuple(part.strip() for part in value.split(",") if part.strip())
        if not cols:
            raise ValueError("--key-columns must not be empty")
        return cols
    return tuple(get_segy_profile(profile_name).key_columns)


def _parse_midpoint_key_columns(value: Optional[str]) -> Tuple[str, str]:
    cols = tuple(part.strip() for part in (value or "cmp_line,cmp").split(",") if part.strip())
    if len(cols) != 2:
        raise ValueError("--midpoint-key-columns must contain exactly two columns")
    return cols[0], cols[1]


def _list_from_spec(value, default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _key_matrix(group, key_columns: Sequence[str]) -> np.ndarray:
    missing = [key for key in key_columns if key not in group]
    if missing:
        raise ValueError(f"H5 group {group.name} is missing key datasets: {missing}")
    cols = [np.asarray(group[key][:], dtype=np.int64) for key in key_columns]
    lengths = {len(col) for col in cols}
    if len(lengths) != 1:
        raise ValueError(f"Key columns have inconsistent lengths in {group.name}: {lengths}")
    return np.stack(cols, axis=1)


def _build_unique_lookup(
    keys: np.ndarray,
    *,
    label: str,
    key_name: str = "geometry key",
) -> Dict[Tuple[int, ...], int]:
    lookup: Dict[Tuple[int, ...], int] = {}
    duplicate = None
    for idx, row in enumerate(keys):
        key = tuple(int(v) for v in row)
        if key in lookup:
            duplicate = key
            break
        lookup[key] = idx
    if duplicate is not None:
        raise ValueError(f"Duplicate {key_name} in {label}: {duplicate}")
    return lookup


def _build_multi_lookup(keys: np.ndarray) -> Dict[Tuple[int, ...], list[int]]:
    lookup: Dict[Tuple[int, ...], list[int]] = {}
    for idx, row in enumerate(keys):
        key = tuple(int(v) for v in row)
        lookup.setdefault(key, []).append(idx)
    return lookup


def _ovt_from_bins(
    offset_x_bin: np.ndarray,
    offset_y_bin: np.ndarray,
    *,
    ref_min_x: Optional[int] = None,
    ref_min_y: Optional[int] = None,
    ref_ny: Optional[int] = None,
    ref_max_x: Optional[int] = None,
    ref_max_y: Optional[int] = None,
    allow_outside_grid: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if offset_x_bin.size == 0:
        meta = {
            "ovt_min_offset_x_bin": 0 if ref_min_x is None else int(ref_min_x),
            "ovt_max_offset_x_bin": 0 if ref_max_x is None else int(ref_max_x),
            "ovt_min_offset_y_bin": 0 if ref_min_y is None else int(ref_min_y),
            "ovt_max_offset_y_bin": 0 if ref_max_y is None else int(ref_max_y),
            "ovt_grid_nx": 0,
            "ovt_grid_ny": 0 if ref_ny is None else int(ref_ny),
        }
        return np.asarray([], dtype=np.int32), meta

    min_x = int(offset_x_bin.min()) if ref_min_x is None else int(ref_min_x)
    max_x = int(offset_x_bin.max()) if ref_max_x is None else int(ref_max_x)
    min_y = int(offset_y_bin.min()) if ref_min_y is None else int(ref_min_y)
    max_y = int(offset_y_bin.max()) if ref_max_y is None else int(ref_max_y)
    n_y = max_y - min_y + 1 if ref_ny is None else int(ref_ny)
    if n_y <= 0:
        raise ValueError(f"Invalid OVT grid ny={n_y}")

    if allow_outside_grid:
        # Clip bin indices to reference grid boundaries so OVT numbers stay valid.
        clipped_x = np.clip(offset_x_bin, min_x, max_x)
        clipped_y = np.clip(offset_y_bin, min_y, max_y)
        n_clipped_x = int((offset_x_bin != clipped_x).sum())
        n_clipped_y = int((offset_y_bin != clipped_y).sum())
        if n_clipped_x or n_clipped_y:
            print(
                f"[WARNING] Clipping offset bins to reference grid: "
                f"{n_clipped_x} trace(s) on offset_x_bin, "
                f"{n_clipped_y} trace(s) on offset_y_bin. "
                f"Grid x=[{min_x}, {max_x}] y=[{min_y}, {max_y}]",
                file=sys.stderr,
            )
        offset_x_bin = clipped_x
        offset_y_bin = clipped_y
    else:
        if offset_x_bin.min() < min_x or offset_x_bin.max() > max_x:
            raise ValueError(
                "offset_x_bin outside reference grid: "
                f"data=[{int(offset_x_bin.min())}, {int(offset_x_bin.max())}] "
                f"grid=[{min_x}, {max_x}]"
            )
        if offset_y_bin.min() < min_y or offset_y_bin.max() > max_y:
            raise ValueError(
                "offset_y_bin outside reference grid: "
                f"data=[{int(offset_y_bin.min())}, {int(offset_y_bin.max())}] "
                f"grid=[{min_y}, {max_y}]"
            )

    ovt = (offset_x_bin.astype(np.int64) - min_x) * n_y
    ovt += offset_y_bin.astype(np.int64) - min_y + 1
    meta = {
        "ovt_min_offset_x_bin": min_x,
        "ovt_max_offset_x_bin": max_x,
        "ovt_min_offset_y_bin": min_y,
        "ovt_max_offset_y_bin": max_y,
        "ovt_grid_nx": max_x - min_x + 1,
        "ovt_grid_ny": n_y,
    }
    return _as_int32("ovt", ovt), meta


def compute_cartesian_ovt(
    sx,
    sy,
    rx,
    ry,
    *,
    wave_type: str = "PP",
    binning_mode: str = "expert",
    offset_x_bin_size: Optional[float] = None,
    offset_y_bin_size: Optional[float] = None,
    source_line_interval: Optional[float] = None,
    receiver_line_interval: Optional[float] = None,
    gamma: float = 2.0,
    offset_x_shift: Optional[float] = None,
    offset_y_shift: Optional[float] = None,
    swap: bool = False,
    grid_spec: Optional[Mapping[str, object]] = None,
    allow_outside_grid: bool = False,
) -> Dict[str, object]:
    if swap:
        raise NotImplementedError("PP swap is not implemented in this H5 OVTBin utility.")

    if grid_spec is not None:
        wave = str(grid_spec.get("wave_type", wave_type)).upper()
        mode = str(grid_spec.get("binning_mode", binning_mode)).lower()
        bin_x = float(grid_spec["offset_x_bin_size"])
        bin_y = float(grid_spec["offset_y_bin_size"])
        shift_x = float(grid_spec["offset_x_shift"])
        shift_y = float(grid_spec["offset_y_shift"])
        gamma = float(grid_spec.get("gamma", gamma))
    else:
        wave = _normalize_wave_type(wave_type)
        mode = _normalize_binning_mode(binning_mode)
        bin_x, bin_y = resolve_cartesian_bin_sizes(
            wave_type=wave,
            binning_mode=mode,
            offset_x_bin_size=offset_x_bin_size,
            offset_y_bin_size=offset_y_bin_size,
            source_line_interval=source_line_interval,
            receiver_line_interval=receiver_line_interval,
            gamma=gamma,
        )
        shift_x = -0.5 * bin_x if offset_x_shift is None else _validate_shift("offset_x_shift", offset_x_shift, bin_x)
        shift_y = -0.5 * bin_y if offset_y_shift is None else _validate_shift("offset_y_shift", offset_y_shift, bin_y)

    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)
    shapes = {arr.shape for arr in (sx, sy, rx, ry)}
    if len(shapes) != 1:
        raise ValueError(f"sx/sy/rx/ry must have the same shape, got {shapes}")

    offset_x = rx - sx
    offset_y = ry - sy
    offset = np.hypot(offset_x, offset_y)
    azimuth = (np.degrees(np.arctan2(offset_y, offset_x)) + 360.0) % 360.0
    azimuth = np.where(offset == 0.0, 0.0, azimuth)
    midpoint_x = 0.5 * (sx + rx)
    midpoint_y = 0.5 * (sy + ry)

    offset_x_bin = np.floor((offset_x - shift_x) / bin_x).astype(np.int64)
    offset_y_bin = np.floor((offset_y - shift_y) / bin_y).astype(np.int64)
    offset_x_center = shift_x + (offset_x_bin.astype(np.float64) + 0.5) * bin_x
    offset_y_center = shift_y + (offset_y_bin.astype(np.float64) + 0.5) * bin_y

    if grid_spec is None:
        ovt, ovt_meta = _ovt_from_bins(offset_x_bin, offset_y_bin)
    else:
        ovt, ovt_meta = _ovt_from_bins(
            offset_x_bin,
            offset_y_bin,
            ref_min_x=int(grid_spec["ovt_min_offset_x_bin"]),
            ref_min_y=int(grid_spec["ovt_min_offset_y_bin"]),
            ref_ny=int(grid_spec["ovt_grid_ny"]),
            ref_max_x=int(grid_spec["ovt_max_offset_x_bin"]),
            ref_max_y=int(grid_spec["ovt_max_offset_y_bin"]),
            allow_outside_grid=allow_outside_grid,
        )

    if ovt.size:
        _, inverse, counts = np.unique(ovt, return_inverse=True, return_counts=True)
        ovt_fold = counts[inverse].astype(np.int32)
    else:
        ovt_fold = np.asarray([], dtype=np.int32)

    attrs: Dict[str, object] = {
        "algorithm_version": ALGORITHM_VERSION,
        "coordinate_type": COORDINATE_TYPE,
        "wave_type": wave,
        "binning_mode": mode,
        "offset_x_bin_size": float(bin_x),
        "offset_y_bin_size": float(bin_y),
        "offset_x_shift": float(shift_x),
        "offset_y_shift": float(shift_y),
        "gamma": float(gamma),
        "swap": bool(swap),
        "ovt_numbering": "x_major_reference_cartesian_grid"
        if grid_spec is not None
        else "x_major_observed_cartesian_grid",
    }
    attrs.update(ovt_meta)

    return {
        "offset_x": offset_x.astype(np.float32),
        "offset_y": offset_y.astype(np.float32),
        "offset": offset.astype(np.float32),
        "azimuth": azimuth.astype(np.float32),
        "offset_x_bin": _as_int32("offset_x_bin", offset_x_bin),
        "offset_y_bin": _as_int32("offset_y_bin", offset_y_bin),
        "offset_x_center": offset_x_center.astype(np.float32),
        "offset_y_center": offset_y_center.astype(np.float32),
        "midpoint_x": midpoint_x.astype(np.float32),
        "midpoint_y": midpoint_y.astype(np.float32),
        "ovt": ovt,
        "ovt_fold": ovt_fold,
        "attrs": attrs,
    }


def _midpoint_bin_matrix(
    group,
    *,
    midpoint_bin_mode: str,
    midpoint_key_columns: Sequence[str],
) -> np.ndarray:
    mode = str(midpoint_bin_mode).lower()
    if mode == "cmp":
        if len(midpoint_key_columns) != 2:
            raise ValueError("midpoint_key_columns must contain exactly two columns")
        return _key_matrix(group, midpoint_key_columns)
    if mode == "coordinate":
        raise NotImplementedError(
            "--midpoint-bin-mode coordinate is not implemented yet. "
            "Use --midpoint-bin-mode cmp with cmp_line,cmp."
        )
    raise ValueError(f"Unsupported midpoint_bin_mode: {midpoint_bin_mode!r}")


def compute_midpoint_fields(
    group,
    *,
    midpoint_bin_mode: str,
    midpoint_key_columns: Sequence[str],
) -> Dict[str, np.ndarray]:
    sx, sy, rx, ry = _load_h5_coordinates(group)
    midpoint_bins = _midpoint_bin_matrix(
        group,
        midpoint_bin_mode=midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    return {
        "midpoint_x": (0.5 * (sx.astype(np.float64) + rx.astype(np.float64))).astype(np.float32),
        "midpoint_y": (0.5 * (sy.astype(np.float64) + ry.astype(np.float64))).astype(np.float32),
        "midpoint_x_bin": _as_int32("midpoint_x_bin", midpoint_bins[:, 0]),
        "midpoint_y_bin": _as_int32("midpoint_y_bin", midpoint_bins[:, 1]),
    }


def write_midpoint_to_group(
    group,
    *,
    midpoint_bin_mode: str,
    midpoint_key_columns: Sequence[str],
    overwrite: bool,
) -> Dict[str, np.ndarray]:
    midpoint = compute_midpoint_fields(
        group,
        midpoint_bin_mode=midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    for name in MIDPOINT_FIELD_NAMES:
        _write_dataset(group, name, midpoint[name], overwrite=overwrite)
    group.attrs["midpoint_bin_mode"] = str(midpoint_bin_mode)
    group.attrs["midpoint_key_columns"] = ",".join(midpoint_key_columns)
    group.attrs["cell_key"] = ",".join(CELL_KEY_FIELDS)
    return midpoint


def _cell_key_matrix(
    group,
    ovt: Mapping[str, object],
    *,
    midpoint_bin_mode: str,
    midpoint_key_columns: Sequence[str],
) -> np.ndarray:
    midpoint_bins = _midpoint_bin_matrix(
        group,
        midpoint_bin_mode=midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    offset_x_bin = np.asarray(ovt["offset_x_bin"], dtype=np.int64)
    offset_y_bin = np.asarray(ovt["offset_y_bin"], dtype=np.int64)
    n = len(offset_x_bin)
    if midpoint_bins.shape[0] != n or len(offset_y_bin) != n:
        raise ValueError(
            "Cell key component lengths differ: "
            f"midpoint={midpoint_bins.shape[0]}, "
            f"offset_x_bin={len(offset_x_bin)}, offset_y_bin={len(offset_y_bin)}"
        )
    return np.column_stack(
        (
            midpoint_bins[:, 0],
            midpoint_bins[:, 1],
            offset_x_bin,
            offset_y_bin,
        )
    ).astype(np.int64, copy=False)


def _validate_unique_cell_keys(
    group,
    ovt: Mapping[str, object],
    *,
    midpoint_bin_mode: str,
    midpoint_key_columns: Sequence[str],
) -> None:
    keys = _cell_key_matrix(
        group,
        ovt,
        midpoint_bin_mode=midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    _build_unique_lookup(keys, label=f"H5 group {group.name}", key_name="cell key")


def build_grid_spec(
    ovt: Mapping[str, object],
    *,
    key_columns: Sequence[str],
    group_name: str,
    midpoint_bin_mode: str = "cmp",
    midpoint_key_columns: Sequence[str] = ("cmp_line", "cmp"),
    projection_mode: str = "cell",
) -> Dict[str, object]:
    attrs = dict(ovt["attrs"])
    return {
        "grid_spec_version": GRID_SPEC_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "coordinate_type": COORDINATE_TYPE,
        "wave_type": attrs["wave_type"],
        "binning_mode": attrs["binning_mode"],
        "offset_x_bin_size": float(attrs["offset_x_bin_size"]),
        "offset_y_bin_size": float(attrs["offset_y_bin_size"]),
        "offset_x_shift": float(attrs["offset_x_shift"]),
        "offset_y_shift": float(attrs["offset_y_shift"]),
        "gamma": float(attrs.get("gamma", 2.0)),
        "swap": bool(attrs.get("swap", False)),
        "ovt_min_offset_x_bin": int(attrs["ovt_min_offset_x_bin"]),
        "ovt_max_offset_x_bin": int(attrs["ovt_max_offset_x_bin"]),
        "ovt_min_offset_y_bin": int(attrs["ovt_min_offset_y_bin"]),
        "ovt_max_offset_y_bin": int(attrs["ovt_max_offset_y_bin"]),
        "ovt_grid_nx": int(attrs["ovt_grid_nx"]),
        "ovt_grid_ny": int(attrs["ovt_grid_ny"]),
        "key_columns": list(key_columns),
        "projection_mode": str(projection_mode),
        "midpoint_bin_mode": str(midpoint_bin_mode),
        "midpoint_key_columns": list(midpoint_key_columns),
        "cell_key": list(CELL_KEY_FIELDS),
        "group_name": str(group_name),
    }


def load_grid_spec(path: str | os.PathLike[str]) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    required = (
        "offset_x_bin_size",
        "offset_y_bin_size",
        "offset_x_shift",
        "offset_y_shift",
        "ovt_min_offset_x_bin",
        "ovt_max_offset_x_bin",
        "ovt_min_offset_y_bin",
        "ovt_max_offset_y_bin",
        "ovt_grid_ny",
        "key_columns",
        "group_name",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Grid spec {path} is missing fields: {missing}")
    spec.setdefault("projection_mode", "cell")
    spec.setdefault("midpoint_bin_mode", "cmp")
    spec.setdefault("midpoint_key_columns", ["cmp_line", "cmp"])
    spec.setdefault("cell_key", list(CELL_KEY_FIELDS))
    return spec


def _grid_spec_from_group_attrs(
    group,
    key_columns: Sequence[str],
    group_name: str,
    *,
    midpoint_bin_mode: str = "cmp",
    midpoint_key_columns: Sequence[str] = ("cmp_line", "cmp"),
) -> Optional[Dict[str, object]]:
    attrs = group.attrs
    required = (
        "offset_x_bin_size",
        "offset_y_bin_size",
        "offset_x_shift",
        "offset_y_shift",
        "ovt_min_offset_x_bin",
        "ovt_max_offset_x_bin",
        "ovt_min_offset_y_bin",
        "ovt_max_offset_y_bin",
        "ovt_grid_nx",
        "ovt_grid_ny",
    )
    if not all(key in attrs for key in required):
        return None
    return {
        "grid_spec_version": GRID_SPEC_VERSION,
        "algorithm_version": str(attrs.get("algorithm_version", ALGORITHM_VERSION)),
        "coordinate_type": str(attrs.get("coordinate_type", COORDINATE_TYPE)),
        "wave_type": str(attrs.get("wave_type", "PP")),
        "binning_mode": str(attrs.get("binning_mode", "expert")),
        "offset_x_bin_size": float(attrs["offset_x_bin_size"]),
        "offset_y_bin_size": float(attrs["offset_y_bin_size"]),
        "offset_x_shift": float(attrs["offset_x_shift"]),
        "offset_y_shift": float(attrs["offset_y_shift"]),
        "gamma": float(attrs.get("gamma", 2.0)),
        "swap": bool(attrs.get("swap", False)),
        "ovt_min_offset_x_bin": int(attrs["ovt_min_offset_x_bin"]),
        "ovt_max_offset_x_bin": int(attrs["ovt_max_offset_x_bin"]),
        "ovt_min_offset_y_bin": int(attrs["ovt_min_offset_y_bin"]),
        "ovt_max_offset_y_bin": int(attrs["ovt_max_offset_y_bin"]),
        "ovt_grid_nx": int(attrs["ovt_grid_nx"]),
        "ovt_grid_ny": int(attrs["ovt_grid_ny"]),
        "key_columns": list(key_columns),
        "projection_mode": str(attrs.get("projection_mode", "cell")),
        "midpoint_bin_mode": str(attrs.get("midpoint_bin_mode", midpoint_bin_mode)),
        "midpoint_key_columns": list(
            _list_from_spec(attrs.get("midpoint_key_columns"), midpoint_key_columns)
        ),
        "cell_key": list(_list_from_spec(attrs.get("cell_key"), CELL_KEY_FIELDS)),
        "group_name": str(group_name),
    }


def write_grid_spec(path: str | os.PathLike[str], spec: Mapping[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(spec), fh, indent=2, sort_keys=True)


def write_ovt_to_group(group, ovt: Mapping[str, object], *, overwrite: bool) -> None:
    for name in OVT_FIELD_NAMES:
        _write_dataset(group, name, ovt[name], overwrite=overwrite)
    for key, value in dict(ovt["attrs"]).items():
        group.attrs[key] = value


def write_qc_outputs(
    ovt: Mapping[str, object],
    qc_dir: str | os.PathLike[str],
    *,
    midpoint_bin_size: Optional[float] = None,
) -> None:
    qc_path = Path(qc_dir)
    qc_path.mkdir(parents=True, exist_ok=True)

    attrs = dict(ovt["attrs"])
    ovt_ids = np.asarray(ovt["ovt"])
    folds = np.asarray(ovt["ovt_fold"])
    n_traces = int(ovt_ids.size)
    unique_ovt, counts = np.unique(ovt_ids, return_counts=True)

    # ---- basic summary ----
    summary = {
        "trace_count": n_traces,
        "unique_ovt_count": int(unique_ovt.size),
        "min_fold": int(folds.min()) if folds.size else 0,
        "max_fold": int(folds.max()) if folds.size else 0,
        "mean_fold": float(folds.mean()) if folds.size else 0.0,
        "attrs": attrs,
    }
    with open(qc_path / "ovtbin_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # ---- per-OVT fold histogram ----
    with open(qc_path / "ovt_fold_hist.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ovt", "fold"])
        for ovt_id, count in zip(unique_ovt, counts):
            writer.writerow([int(ovt_id), int(count)])

    # ---- per-OVT-per-CMP fold & duplicate rate ----
    if n_traces > 0 and "midpoint_x" in ovt and "midpoint_y" in ovt:
        mpx = np.asarray(ovt["midpoint_x"])
        mpy = np.asarray(ovt["midpoint_y"])
        _write_per_ovt_cmp_qc(
            ovt_ids, mpx, mpy, n_traces, attrs, qc_path,
            midpoint_bin_size=midpoint_bin_size,
        )

    # ---- plots ----
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib is not available; skipped QC plots.", file=sys.stderr)
        return

    # scatter plot
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sc = ax.scatter(
        np.asarray(ovt["offset_x"]),
        np.asarray(ovt["offset_y"]),
        c=ovt_ids,
        s=4,
        cmap="tab20",
        linewidths=0,
    )
    ax.set_xlabel("offset_x")
    ax.set_ylabel("offset_y")
    ax.set_title("Cartesian OVT bins")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    fig.colorbar(sc, ax=ax, label="ovt")
    fig.tight_layout()
    fig.savefig(qc_path / "offset_xy_ovt_scatter.png")
    plt.close(fig)

    # heatmap: OVT grid population
    if "offset_x_bin" in ovt and "offset_y_bin" in ovt:
        _write_ovt_grid_heatmap(
            np.asarray(ovt["offset_x_bin"]),
            np.asarray(ovt["offset_y_bin"]),
            ovt_ids,
            attrs,
            qc_path,
        )


def _estimate_spatial_bin(x: np.ndarray, y: np.ndarray, fallback: float = 50.0) -> float:
    """Estimate spatial bin size from coordinate data.

    Uses the median spacing between unique sorted coordinate values.
    Falls back to *fallback* (metres) for tiny datasets.
    """
    bins: list[float] = []
    for coord in (x, y):
        if coord.size < 2:
            continue
        uniq = np.unique(coord)
        if uniq.size < 2:
            continue
        diffs = np.diff(uniq)
        diffs = diffs[diffs > 0]
        if diffs.size:
            bins.append(float(np.median(diffs)))
    if not bins:
        return fallback
    # Round to a sensible number
    raw = float(np.median(bins))
    # Snap to common bin sizes: 12.5, 25, 50, 100, 200, 400
    candidates = [12.5, 25.0, 50.0, 100.0, 200.0, 400.0]
    best = min(candidates, key=lambda c: abs(c - raw))
    return best if abs(best - raw) / max(raw, 1e-9) < 0.5 else raw


def _write_per_ovt_cmp_qc(
    ovt_ids: np.ndarray,
    midpoint_x: np.ndarray,
    midpoint_y: np.ndarray,
    n_traces: int,
    attrs: dict,
    qc_path: Path,
    *,
    midpoint_bin_size: Optional[float] = None,
) -> None:
    """Write per-OVT-per-CMP fold statistics and duplicate rate."""
    # Determine midpoint bin size: use user-provided, else auto-estimate from data spacing, else 50m
    if midpoint_bin_size is None:
        midpoint_bin_size = _estimate_spatial_bin(midpoint_x, midpoint_y)

    # Compute midpoint bins
    mpx_bin = np.floor(midpoint_x / midpoint_bin_size).astype(np.int64)
    mpy_bin = np.floor(midpoint_y / midpoint_bin_size).astype(np.int64)

    # Per-OVT-per-CMP fold aggregation
    unique_ovts = np.unique(ovt_ids)
    per_ovt_cmp_folds: List[float] = []       # mean fold per CMP within each OVT
    per_ovt_cmp_counts: List[int] = []         # unique CMP count per OVT
    per_ovt_duplicate_rates: List[float] = []  # duplicate rate per OVT

    for ovt_id in unique_ovts:
        mask = ovt_ids == ovt_id
        ovt_n = int(mask.sum())
        # Pack (mpx_bin, mpy_bin) into a single int64 key for fast unique
        keys = (mpx_bin[mask].astype(np.int64) << 32) | (mpy_bin[mask].astype(np.int64) & 0xFFFFFFFF)
        unique_keys, key_counts = np.unique(keys, return_counts=True)
        n_unique = int(unique_keys.size)
        per_ovt_cmp_counts.append(n_unique)
        if n_unique > 0:
            per_ovt_cmp_folds.append(float(ovt_n) / float(n_unique))
            per_ovt_duplicate_rates.append(1.0 - float(n_unique) / float(ovt_n))
        else:
            per_ovt_cmp_folds.append(0.0)
            per_ovt_duplicate_rates.append(0.0)

    # Global unique CMP bins
    n_unique_all = int(np.unique(np.stack([mpx_bin, mpy_bin], axis=1), axis=0).shape[0])
    # Unique (OVT, CMP) pairs — stack three columns for correct uniqueness
    stacked = np.stack([ovt_ids.astype(np.int64), mpx_bin, mpy_bin], axis=1)
    n_unique_ovt_cmp = int(np.unique(stacked, axis=0).shape[0])

    ovt_cmp_dup = {
        "midpoint_bin_size": float(midpoint_bin_size),
        "total_traces": n_traces,
        "unique_cmp_bins_global": n_unique_all,
        "unique_ovt_cmp_pairs": n_unique_ovt_cmp,
        "ovt_cmp_duplicate_traces": n_traces - n_unique_ovt_cmp,
        "ovt_cmp_duplicate_rate": round(1.0 - n_unique_ovt_cmp / max(n_traces, 1), 6),
        "per_ovt_cmp_fold": {
            "mean": round(float(np.mean(per_ovt_cmp_folds)), 2),
            "median": round(float(np.median(per_ovt_cmp_folds)), 2),
            "p99": round(float(np.percentile(per_ovt_cmp_folds, 99)), 2),
            "min": round(float(np.min(per_ovt_cmp_folds)), 2),
            "max": round(float(np.max(per_ovt_cmp_folds)), 2),
            "fraction_lt_10": round(
                float(np.mean(np.array(per_ovt_cmp_folds) < 10)), 4
            ),
            "fraction_gt_50": round(
                float(np.mean(np.array(per_ovt_cmp_folds) > 50)), 4
            ),
        },
        "per_ovt_duplicate_rate": {
            "mean": round(float(np.mean(per_ovt_duplicate_rates)), 6),
            "median": round(float(np.median(per_ovt_duplicate_rates)), 6),
            "max": round(float(np.max(per_ovt_duplicate_rates)), 6),
        },
    }
    with open(qc_path / "ovt_cmp_duplicate.json", "w", encoding="utf-8") as fh:
        json.dump(ovt_cmp_dup, fh, indent=2, sort_keys=True)

    # Per-OVT-per-CMP fold CSV
    with open(qc_path / "per_ovt_cmp_fold.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ovt", "n_traces", "unique_cmp", "mean_cmp_fold", "duplicate_rate"])
        for i, ovt_id in enumerate(unique_ovts):
            writer.writerow([
                int(ovt_id),
                int((ovt_ids == ovt_id).sum()),
                per_ovt_cmp_counts[i],
                round(per_ovt_cmp_folds[i], 4),
                round(per_ovt_duplicate_rates[i], 6),
            ])

    # Print summary to stdout
    print(f"[QC] midpoint bin size: {midpoint_bin_size:.1f}m")
    print(f"[QC] unique CMP bins (global): {n_unique_all}")
    print(f"[QC] unique OVT-CMP pairs: {n_unique_ovt_cmp}")
    dup_pct = ovt_cmp_dup["ovt_cmp_duplicate_rate"] * 100
    print(f"[QC] OVT-CMP duplicate rate: {dup_pct:.2f}% ({n_traces - n_unique_ovt_cmp} / {n_traces} traces)")
    print(f"[QC] per-OVT-per-CMP fold: mean={ovt_cmp_dup['per_ovt_cmp_fold']['mean']:.1f}, "
          f"median={ovt_cmp_dup['per_ovt_cmp_fold']['median']:.1f}, "
          f"p99={ovt_cmp_dup['per_ovt_cmp_fold']['p99']:.1f}")
    print(f"[QC]   fold < 10: {ovt_cmp_dup['per_ovt_cmp_fold']['fraction_lt_10']*100:.2f}% OVTs, "
          f"fold > 50: {ovt_cmp_dup['per_ovt_cmp_fold']['fraction_gt_50']*100:.2f}% OVTs")


def _write_ovt_grid_heatmap(
    offset_x_bin: np.ndarray,
    offset_y_bin: np.ndarray,
    ovt_ids: np.ndarray,
    attrs: dict,
    qc_path: Path,
) -> None:
    """Write a 2D heatmap of OVT grid population."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    x_min = int(offset_x_bin.min())
    x_max = int(offset_x_bin.max())
    y_min = int(offset_y_bin.min())
    y_max = int(offset_y_bin.max())
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    # Build 2D histogram: count traces per (offset_x_bin, offset_y_bin)
    grid = np.zeros((ny, nx), dtype=np.int64)
    for xb, yb in zip(offset_x_bin, offset_y_bin):
        grid[yb - y_min, xb - x_min] += 1

    fig, ax = plt.subplots(figsize=(max(8, nx * 0.12), max(7, ny * 0.12)), dpi=150)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5],
    )
    ax.set_xlabel("offset_x_bin")
    ax.set_ylabel("offset_y_bin")
    ax.set_title(f"OVT grid population (nx={nx}, ny={ny}, cells={nx * ny})")
    fig.colorbar(im, ax=ax, label="trace count")
    fig.tight_layout()
    fig.savefig(qc_path / "ovt_grid_heatmap.png")
    plt.close(fig)

    # Also write a CSV of the grid for downstream use
    with open(qc_path / "ovt_grid_heatmap.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["offset_x_bin", "offset_y_bin", "trace_count"])
        for iy in range(ny):
            for ix in range(nx):
                count = int(grid[iy, ix])
                if count > 0:
                    writer.writerow([x_min + ix, y_min + iy, count])


def _log_grid_spec(spec: Mapping[str, object]) -> None:
    grid_nx = int(spec["ovt_grid_nx"])
    grid_ny = int(spec["ovt_grid_ny"])
    n_cells = grid_nx * grid_ny
    print(
        f"[GRID] shape=({grid_nx}, {grid_ny})=>{n_cells} cells, "
        f"bin=({spec['offset_x_bin_size']}, {spec['offset_y_bin_size']}), "
        f"shift=({spec['offset_x_shift']}, {spec['offset_y_shift']}), "
        f"offset_x_bin=[{spec['ovt_min_offset_x_bin']}, {spec['ovt_max_offset_x_bin']}], "
        f"offset_y_bin=[{spec['ovt_min_offset_y_bin']}, {spec['ovt_max_offset_y_bin']}]",
        file=sys.stderr,
    )


def _log_fold_stats(fold_array: np.ndarray, label: str = "") -> None:
    if fold_array.size == 0:
        print(f"[FOLD] {label}no traces", file=sys.stderr)
        return
    prefix = f"[FOLD] {label}" if label else "[FOLD]"
    print(
        f"{prefix}n_traces={fold_array.size}, min={int(fold_array.min())}, "
        f"max={int(fold_array.max())}, "
        f"mean={float(fold_array.mean()):.2f}, "
        f"median={float(np.median(fold_array)):.1f}, "
        f"std={float(fold_array.std()):.2f}",
        file=sys.stderr,
    )


def _log_projection_stats(stats: Mapping[str, int], *, mode: str = "cell") -> None:
    grid = int(stats["grid_traces"])
    inp = int(stats["input_traces"])

    if mode == "cell":
        matched_cells = int(stats.get("matched_cells", 0))
        matched_traces = int(stats.get("matched_input_traces", 0))
        unmatched = int(stats.get("unmatched_input_cells", 0))
        coverage = matched_cells / grid if grid > 0 else 0.0
        input_util = matched_traces / inp if inp > 0 else 0.0
        print(
            f"[PROJECTION] mode=cell, grid_cells={grid}, input_traces={inp}, "
            f"matched_cells={matched_cells} ({coverage:.1%}), "
            f"unmatched_input_cells={unmatched}, "
            f"input_utilization={input_util:.1%}"
            + (
                f", merged_duplicates={int(stats['merged_duplicate_traces'])}"
                if int(stats.get("merged_duplicate_traces", 0)) > 0
                else ""
            ),
            file=sys.stderr,
        )
    else:
        matched = int(stats.get("matched_input_traces", 0))
        coverage = matched / grid if grid > 0 else 0.0
        print(
            f"[PROJECTION] mode=trace-key, grid_traces={grid}, input_traces={inp}, "
            f"matched={matched} ({coverage:.1%})",
            file=sys.stderr,
        )

    observed = int(stats.get("observed_traces", 0))
    missing = int(stats.get("missing_traces", 0))
    if grid > 0:
        print(
            f"[PROJECTION] observed_traces={observed} ({observed / grid:.1%}), "
            f"missing_traces={missing} ({missing / grid:.1%})",
            file=sys.stderr,
        )


def _remove_existing_output(path: Path, *, overwrite_output: bool) -> None:
    if not path.exists():
        return
    if not overwrite_output:
        raise FileExistsError(
            f"Output file already exists: {path}. Use --overwrite-output to replace it."
        )
    path.unlink()


def _load_input_block(args: argparse.Namespace) -> Mapping[str, object]:
    if args.input_type == "h5":
        raise RuntimeError("_load_input_block is only used for SEG-Y input")
    from convert_tool.Segy2H5 import organize_traces

    profile = get_segy_profile(args.segy_profile)
    return organize_traces(
        args.input,
        headers_df=None,
        sort_keys=None,
        mode=args.segy_mode,
        profile=profile,
    )


def _create_group_from_block(h5f, group_name: str, block: Mapping[str, object]):
    group = h5f.create_group(group_name)
    for key, value in block.items():
        _copy_dataset(group, key, value)
    return group


def _copy_h5_to_output(input_path: Path, output_path: Path, *, overwrite_output: bool) -> None:
    same_file = input_path.resolve() == output_path.resolve()
    if same_file:
        return
    _remove_existing_output(output_path, overwrite_output=overwrite_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def _load_grid_group(args: argparse.Namespace, h5py):
    if args.grid_source is None:
        raise ValueError("--align-to-grid requires --grid-source")
    if args.grid_source_type != "h5":
        raise NotImplementedError("Only --grid-source-type h5 is supported in v1")
    grid_path = Path(args.grid_source)
    grid_group_name = args.grid_group_name or args.group_name
    h5f = h5py.File(grid_path, "r")
    if grid_group_name not in h5f:
        h5f.close()
        raise ValueError(f"H5 group {grid_group_name!r} not found in {grid_path}")
    return h5f, h5f[grid_group_name]


def _resolve_grid_spec(
    args: argparse.Namespace,
    group,
    key_columns: Sequence[str],
    midpoint_key_columns: Sequence[str],
) -> Optional[Dict[str, object]]:
    if args.grid_spec_in:
        return load_grid_spec(args.grid_spec_in)
    if args.grid_source is None:
        return None
    group_name = args.grid_group_name or args.group_name
    spec_from_attrs = _grid_spec_from_group_attrs(
        group,
        key_columns,
        group_name,
        midpoint_bin_mode=args.midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    if spec_from_attrs is not None:
        return spec_from_attrs
    sx, sy, rx, ry = _load_h5_coordinates(group)
    ovt = compute_cartesian_ovt(
        sx,
        sy,
        rx,
        ry,
        wave_type=args.wave_type,
        binning_mode=args.binning_mode,
        offset_x_bin_size=args.offset_x_bin_size,
        offset_y_bin_size=args.offset_y_bin_size,
        source_line_interval=args.source_line_interval,
        receiver_line_interval=args.receiver_line_interval,
        gamma=args.gamma,
        offset_x_shift=args.offset_x_shift,
        offset_y_shift=args.offset_y_shift,
        swap=args.swap,
    )
    return build_grid_spec(
        ovt,
        key_columns=key_columns,
        group_name=group_name,
        midpoint_bin_mode=args.midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
        projection_mode=args.projection_mode,
    )


def _project_input_to_grid_trace_key(
    *,
    input_group,
    grid_group,
    output_group,
    key_columns: Sequence[str],
    missing_eps: float,
) -> Dict[str, int]:
    grid_keys = _key_matrix(grid_group, key_columns)
    input_keys = _key_matrix(input_group, key_columns)
    grid_lookup = _build_unique_lookup(grid_keys, label=f"grid group {grid_group.name}")
    input_lookup = _build_unique_lookup(input_keys, label=f"input group {input_group.name}")

    if "data" not in input_group:
        raise ValueError(f"Input group {input_group.name} is missing dataset: data")
    if "data" not in grid_group:
        raise ValueError(f"Grid group {grid_group.name} is missing dataset: data")

    grid_data_shape = grid_group["data"].shape
    input_data = input_group["data"][:]
    output_data = np.zeros(grid_data_shape, dtype=input_data.dtype)
    trace_mask = np.zeros(grid_data_shape[0], dtype=np.float32)

    unmatched_input = []
    matched = 0
    observed = 0
    for key, input_idx in input_lookup.items():
        grid_idx = grid_lookup.get(key)
        if grid_idx is None:
            unmatched_input.append(key)
            continue
        output_data[grid_idx] = input_data[input_idx]
        matched += 1
        if np.any(np.abs(input_data[input_idx]) > missing_eps):
            trace_mask[grid_idx] = 1.0
            observed += 1

    if unmatched_input:
        preview = unmatched_input[:5]
        raise ValueError(
            f"{len(unmatched_input)} input geometry keys were not found in grid. "
            f"First keys: {preview}"
        )

    for key in grid_group.keys():
        if key in OVT_FIELD_NAMES or key == "trace_mask":
            continue
        if key == "data":
            _copy_dataset(output_group, key, output_data)
        else:
            _copy_dataset(output_group, key, grid_group[key][:])
    _copy_dataset(output_group, "trace_mask", trace_mask)
    return {
        "grid_traces": int(grid_data_shape[0]),
        "input_traces": int(input_data.shape[0]),
        "matched_input_traces": int(matched),
        "observed_traces": int(observed),
        "missing_traces": int(grid_data_shape[0] - observed),
    }


def _trace_from_duplicate_policy(
    input_data: np.ndarray,
    indices: Sequence[int],
    *,
    policy: str,
):
    if len(indices) == 1:
        return input_data[int(indices[0])], 0
    if policy == "error":
        raise ValueError(f"Duplicate input cell key maps to {len(indices)} traces")
    if policy == "first":
        return input_data[int(indices[0])], len(indices) - 1
    if policy == "mean":
        return np.mean(input_data[np.asarray(indices, dtype=np.int64)], axis=0), len(indices) - 1
    raise ValueError(f"Unsupported input duplicate policy: {policy!r}")


def _copy_projected_grid_datasets(
    *,
    grid_group,
    output_group,
    output_data: np.ndarray,
    trace_mask: np.ndarray,
    projection_fold: np.ndarray,
) -> None:
    skip = set(OVT_FIELD_NAMES) | set(MIDPOINT_FIELD_NAMES) | set(PROJECTION_FIELD_NAMES)
    for key in grid_group.keys():
        if key in skip:
            continue
        if key == "data":
            _copy_dataset(output_group, key, output_data)
        else:
            _copy_dataset(output_group, key, grid_group[key][:])
    _copy_dataset(output_group, "trace_mask", trace_mask)
    _copy_dataset(output_group, "projection_fold", projection_fold)


def _project_input_to_grid_cell(
    *,
    input_group,
    grid_group,
    output_group,
    grid_spec: Mapping[str, object],
    midpoint_bin_mode: str,
    midpoint_key_columns: Sequence[str],
    input_duplicate_policy: str,
    outside_grid_policy: str,
) -> Dict[str, int]:
    if "data" not in input_group:
        raise ValueError(f"Input group {input_group.name} is missing dataset: data")
    if "data" not in grid_group:
        raise ValueError(f"Grid group {grid_group.name} is missing dataset: data")

    grid_sx, grid_sy, grid_rx, grid_ry = _load_h5_coordinates(grid_group)
    input_sx, input_sy, input_rx, input_ry = _load_h5_coordinates(input_group)
    grid_ovt = compute_cartesian_ovt(grid_sx, grid_sy, grid_rx, grid_ry, grid_spec=grid_spec)
    _log_fold_stats(np.asarray(grid_ovt["ovt_fold"]), label="grid ")
    input_ovt = compute_cartesian_ovt(
        input_sx,
        input_sy,
        input_rx,
        input_ry,
        grid_spec=grid_spec,
        allow_outside_grid=True,
    )
    _log_fold_stats(np.asarray(input_ovt["ovt_fold"]), label="input ")
    print(
        f"[PROJECT] grid_traces={grid_sx.size}, input_traces={input_sx.size}",
        file=sys.stderr,
    )

    grid_keys = _cell_key_matrix(
        grid_group,
        grid_ovt,
        midpoint_bin_mode=midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    input_keys = _cell_key_matrix(
        input_group,
        input_ovt,
        midpoint_bin_mode=midpoint_bin_mode,
        midpoint_key_columns=midpoint_key_columns,
    )
    grid_lookup = _build_unique_lookup(
        grid_keys,
        label=f"grid group {grid_group.name}",
        key_name="cell key",
    )
    input_lookup = _build_multi_lookup(input_keys)
    print(
        f"[PROJECT] grid_cell_keys={len(grid_lookup)}, "
        f"input_cell_keys={len(input_lookup)}, "
        f"input_duplicate_policy={input_duplicate_policy}",
        file=sys.stderr,
    )

    grid_data_shape = grid_group["data"].shape
    input_data = input_group["data"][:]
    output_data = np.zeros(grid_data_shape, dtype=input_data.dtype)
    trace_mask = np.zeros(grid_data_shape[0], dtype=np.float32)
    projection_fold = np.zeros(grid_data_shape[0], dtype=np.int32)

    unmatched_input = []
    matched_cells = 0
    matched_input_traces = 0
    merged_duplicates = 0
    for key, input_indices in input_lookup.items():
        grid_idx = grid_lookup.get(key)
        if grid_idx is None:
            unmatched_input.append(key)
            continue
        trace_data, merged = _trace_from_duplicate_policy(
            input_data,
            input_indices,
            policy=input_duplicate_policy,
        )
        output_data[grid_idx] = trace_data
        trace_mask[grid_idx] = 1.0
        projection_fold[grid_idx] = len(input_indices)
        matched_cells += 1
        matched_input_traces += len(input_indices)
        merged_duplicates += merged

    if unmatched_input and outside_grid_policy == "error":
        preview = unmatched_input[:5]
        raise ValueError(
            f"{len(unmatched_input)} input cell keys were not found in grid. "
            f"First keys: {preview}"
        )
    if unmatched_input:
        preview = unmatched_input[:5]
        print(
            f"[WARNING] {len(unmatched_input)} input cell keys were not found in grid "
            f"and were skipped. First keys: {preview}",
            file=sys.stderr,
        )

    _copy_projected_grid_datasets(
        grid_group=grid_group,
        output_group=output_group,
        output_data=output_data,
        trace_mask=trace_mask,
        projection_fold=projection_fold,
    )
    return {
        "grid_traces": int(grid_data_shape[0]),
        "input_traces": int(input_data.shape[0]),
        "matched_cells": int(matched_cells),
        "matched_input_traces": int(matched_input_traces),
        "unmatched_input_cells": int(len(unmatched_input)),
        "merged_duplicate_traces": int(merged_duplicates),
        "observed_traces": int(np.count_nonzero(trace_mask > 0.5)),
        "missing_traces": int(grid_data_shape[0] - np.count_nonzero(trace_mask > 0.5)),
    }


def _midpoint_config_from_spec(
    args: argparse.Namespace,
    spec: Optional[Mapping[str, object]],
) -> Tuple[str, Tuple[str, ...]]:
    default_cols = _parse_midpoint_key_columns(args.midpoint_key_columns)
    if spec is None:
        return args.midpoint_bin_mode, default_cols
    mode = str(spec.get("midpoint_bin_mode", args.midpoint_bin_mode))
    cols = _list_from_spec(spec.get("midpoint_key_columns"), default_cols)
    if len(cols) != 2:
        raise ValueError(f"Grid spec midpoint_key_columns must contain two columns, got {cols}")
    return mode, cols


def process_h5(args: argparse.Namespace) -> None:
    h5py = _require_h5py()
    output_path = Path(args.output)
    key_columns = _parse_key_columns(args.key_columns, args.segy_profile)
    midpoint_key_columns = _parse_midpoint_key_columns(args.midpoint_key_columns)

    print(
        f"[START] input={args.input}, output={output_path}, group={args.group_name}, "
        f"align_to_grid={args.align_to_grid}, projection_mode={args.projection_mode}, "
        f"binning_mode={args.binning_mode}",
        file=sys.stderr,
    )

    if args.align_to_grid:
        _remove_existing_output(output_path, overwrite_output=args.overwrite_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_h5, grid_group = _load_grid_group(args, h5py)
        try:
            spec = (
                load_grid_spec(args.grid_spec_in)
                if args.grid_spec_in
                else _resolve_grid_spec(args, grid_group, key_columns, midpoint_key_columns)
            )
            if spec is not None:
                _log_grid_spec(spec)
            midpoint_bin_mode, spec_midpoint_key_columns = _midpoint_config_from_spec(args, spec)
            with h5py.File(args.input, "r") as in_h5, h5py.File(output_path, "w") as out_h5:
                if args.group_name not in in_h5:
                    raise ValueError(f"H5 group {args.group_name!r} not found in {args.input}")
                out_group = out_h5.create_group(args.group_name)
                if args.projection_mode == "cell":
                    stats = _project_input_to_grid_cell(
                        input_group=in_h5[args.group_name],
                        grid_group=grid_group,
                        output_group=out_group,
                        grid_spec=spec,
                        midpoint_bin_mode=midpoint_bin_mode,
                        midpoint_key_columns=spec_midpoint_key_columns,
                        input_duplicate_policy=args.input_duplicate_policy,
                        outside_grid_policy=args.outside_grid_policy,
                    )
                else:
                    stats = _project_input_to_grid_trace_key(
                        input_group=in_h5[args.group_name],
                        grid_group=grid_group,
                        output_group=out_group,
                        key_columns=key_columns,
                        missing_eps=args.missing_eps,
                    )
                _log_projection_stats(stats, mode=args.projection_mode)
                sx, sy, rx, ry = _load_h5_coordinates(out_group)
                ovt = compute_cartesian_ovt(
                    sx, sy, rx, ry, grid_spec=spec,
                    allow_outside_grid=args.allow_outside_grid,
                )
                _log_fold_stats(np.asarray(ovt["ovt_fold"]), label="grid output ")
                write_ovt_to_group(out_group, ovt, overwrite=False)
                if args.projection_mode == "cell":
                    write_midpoint_to_group(
                        out_group,
                        midpoint_bin_mode=midpoint_bin_mode,
                        midpoint_key_columns=spec_midpoint_key_columns,
                        overwrite=False,
                    )
                for key, value in stats.items():
                    out_group.attrs[f"align_{key}"] = value
                out_group.attrs["projection_mode"] = args.projection_mode
                out_group.attrs["input_duplicate_policy"] = args.input_duplicate_policy
                out_group.attrs["outside_grid_policy"] = args.outside_grid_policy
                if args.projection_mode == "trace-key":
                    out_group.attrs["align_key_columns"] = ",".join(key_columns)
                else:
                    out_group.attrs["align_cell_key"] = ",".join(CELL_KEY_FIELDS)
        finally:
            grid_h5.close()
    else:
        input_path = Path(args.input)
        _copy_h5_to_output(input_path, output_path, overwrite_output=args.overwrite_output)
        with h5py.File(output_path, "a") as h5f:
            if args.group_name not in h5f:
                raise ValueError(f"H5 group {args.group_name!r} not found in {output_path}")
            group = h5f[args.group_name]
            if args.grid_spec_in:
                spec = load_grid_spec(args.grid_spec_in)
            elif args.grid_source:
                grid_h5, grid_group = _load_grid_group(args, h5py)
                try:
                    spec = _resolve_grid_spec(args, grid_group, key_columns, midpoint_key_columns)
                finally:
                    grid_h5.close()
            else:
                spec = None
            midpoint_bin_mode, spec_midpoint_key_columns = _midpoint_config_from_spec(args, spec)
            if spec is not None:
                _log_grid_spec(spec)
            sx, sy, rx, ry = _load_h5_coordinates(group)
            ovt = compute_cartesian_ovt(
                sx,
                sy,
                rx,
                ry,
                wave_type=args.wave_type,
                binning_mode=args.binning_mode,
                offset_x_bin_size=args.offset_x_bin_size,
                offset_y_bin_size=args.offset_y_bin_size,
                source_line_interval=args.source_line_interval,
                receiver_line_interval=args.receiver_line_interval,
                gamma=args.gamma,
                offset_x_shift=args.offset_x_shift,
                offset_y_shift=args.offset_y_shift,
                swap=args.swap,
                grid_spec=spec,
                allow_outside_grid=args.allow_outside_grid,
            )
            _log_fold_stats(np.asarray(ovt["ovt_fold"]), label=f"group {args.group_name} ")
            write_ovt_to_group(group, ovt, overwrite=args.overwrite_fields)
            if args.projection_mode == "cell":
                write_midpoint_to_group(
                    group,
                    midpoint_bin_mode=midpoint_bin_mode,
                    midpoint_key_columns=spec_midpoint_key_columns,
                    overwrite=args.overwrite_fields,
                )
                n_traces = len(np.asarray(ovt["ovt_fold"]))
                try:
                    _validate_unique_cell_keys(
                        group,
                        ovt,
                        midpoint_bin_mode=midpoint_bin_mode,
                        midpoint_key_columns=spec_midpoint_key_columns,
                    )
                    print(f"[CELL] cell keys are unique (n_traces={n_traces})", file=sys.stderr)
                except ValueError:
                    cell_keys = _cell_key_matrix(
                        group,
                        ovt,
                        midpoint_bin_mode=midpoint_bin_mode,
                        midpoint_key_columns=spec_midpoint_key_columns,
                    )
                    n_unique = len(_build_multi_lookup(cell_keys))
                    raise ValueError(
                        f"[CELL] Duplicate cell keys detected: "
                        f"{n_traces} traces but only {n_unique} unique cell keys "
                        f"({n_traces - n_unique} duplicates). "
                        f"Use --projection-mode cell with --align-to-grid to project onto "
                        f"a regular grid, or ensure input traces have unique "
                        f"(midpoint_x_bin, midpoint_y_bin, offset_x_bin, offset_y_bin) tuples."
                    )

    if args.grid_spec_out:
        if "ovt" not in locals():
            raise RuntimeError("Internal error: OVT fields were not computed")
        midpoint_bin_mode, spec_midpoint_key_columns = _midpoint_config_from_spec(args, spec if "spec" in locals() else None)
        spec_out = build_grid_spec(
            ovt,
            key_columns=key_columns,
            group_name=args.group_name,
            midpoint_bin_mode=midpoint_bin_mode,
            midpoint_key_columns=spec_midpoint_key_columns,
            projection_mode=args.projection_mode,
        )
        write_grid_spec(args.grid_spec_out, spec_out)
    if args.qc_dir:
        write_qc_outputs(ovt, args.qc_dir)
    print(f"[DONE] OVT fields written to {output_path}/{args.group_name}", file=sys.stderr)


def process_segy(args: argparse.Namespace) -> None:
    h5py = _require_h5py()
    output_path = Path(args.output)
    key_columns = _parse_key_columns(args.key_columns, args.segy_profile)
    midpoint_key_columns = _parse_midpoint_key_columns(args.midpoint_key_columns)

    print(
        f"[START] input={args.input}, output={output_path}, group={args.group_name}, "
        f"binning_mode={args.binning_mode}, projection_mode={args.projection_mode}",
        file=sys.stderr,
    )
    _remove_existing_output(output_path, overwrite_output=args.overwrite_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    block = _load_input_block(args)
    with h5py.File(output_path, "w") as h5f:
        group = _create_group_from_block(h5f, args.group_name, block)
        spec = None
        if args.grid_spec_in:
            spec = load_grid_spec(args.grid_spec_in)
        elif args.grid_source:
            grid_h5, grid_group = _load_grid_group(args, h5py)
            try:
                spec = _resolve_grid_spec(args, grid_group, key_columns, midpoint_key_columns)
            finally:
                grid_h5.close()
        midpoint_bin_mode, spec_midpoint_key_columns = _midpoint_config_from_spec(args, spec)
        sx, sy, rx, ry = _load_h5_coordinates(group)
        ovt = compute_cartesian_ovt(
            sx,
            sy,
            rx,
            ry,
            wave_type=args.wave_type,
            binning_mode=args.binning_mode,
            offset_x_bin_size=args.offset_x_bin_size,
            offset_y_bin_size=args.offset_y_bin_size,
            source_line_interval=args.source_line_interval,
            receiver_line_interval=args.receiver_line_interval,
            gamma=args.gamma,
            offset_x_shift=args.offset_x_shift,
            offset_y_shift=args.offset_y_shift,
            swap=args.swap,
            grid_spec=spec,
            allow_outside_grid=args.allow_outside_grid,
        )
        _log_fold_stats(np.asarray(ovt["ovt_fold"]), label=f"group {args.group_name} ")
        write_ovt_to_group(group, ovt, overwrite=False)
        if args.projection_mode == "cell":
            write_midpoint_to_group(
                group,
                midpoint_bin_mode=midpoint_bin_mode,
                midpoint_key_columns=spec_midpoint_key_columns,
                overwrite=False,
            )
            n_traces = len(np.asarray(ovt["ovt_fold"]))
            try:
                _validate_unique_cell_keys(
                    group,
                    ovt,
                    midpoint_bin_mode=midpoint_bin_mode,
                    midpoint_key_columns=spec_midpoint_key_columns,
                )
                print(f"[CELL] cell keys are unique (n_traces={n_traces})", file=sys.stderr)
            except ValueError:
                cell_keys = _cell_key_matrix(
                    group,
                    ovt,
                    midpoint_bin_mode=midpoint_bin_mode,
                    midpoint_key_columns=spec_midpoint_key_columns,
                )
                n_unique = len(_build_multi_lookup(cell_keys))
                raise ValueError(
                    f"[CELL] Duplicate cell keys detected: "
                    f"{n_traces} traces but only {n_unique} unique cell keys "
                    f"({n_traces - n_unique} duplicates). "
                    f"Use --projection-mode cell with --align-to-grid to project onto "
                    f"a regular grid, or ensure input traces have unique "
                    f"(midpoint_x_bin, midpoint_y_bin, offset_x_bin, offset_y_bin) tuples."
                )

    if args.grid_spec_out:
        midpoint_bin_mode, spec_midpoint_key_columns = _midpoint_config_from_spec(args, spec if "spec" in locals() else None)
        spec_out = build_grid_spec(
            ovt,
            key_columns=key_columns,
            group_name=args.group_name,
            midpoint_bin_mode=midpoint_bin_mode,
            midpoint_key_columns=spec_midpoint_key_columns,
            projection_mode=args.projection_mode,
        )
        write_grid_spec(args.grid_spec_out, spec_out)
    if args.qc_dir:
        write_qc_outputs(ovt, args.qc_dir)
    print(f"[DONE] SEG-Y converted and OVT fields written to {output_path}/{args.group_name}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute grid-driven Cartesian OVTBin fields and store them in H5."
    )
    parser.add_argument("--input", required=True, help="Input SEG-Y or H5 file")
    parser.add_argument("--input-type", required=True, choices=["segy", "h5"])
    parser.add_argument("--output", required=True, help="Output H5 file")
    parser.add_argument("--group-name", default="1551", help="Input/output H5 group name")
    parser.add_argument("--segy-profile", choices=profile_names(), default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--key-columns", default=None, help="Comma-separated geometry key columns")
    parser.add_argument(
        "--segy-mode",
        default="fixed",
        choices=["fixed", "self_computed"],
        help="SEG-Y header reading mode, only used with --input-type segy",
    )

    parser.add_argument("--grid-source", default=None, help="Regular/label H5 defining full geometry")
    parser.add_argument("--grid-source-type", default="h5", choices=["h5"])
    parser.add_argument("--grid-group-name", default=None, help="Grid-source H5 group name")
    parser.add_argument("--grid-spec-out", default=None, help="Write computed OVT grid spec JSON")
    parser.add_argument("--grid-spec-in", default=None, help="Read existing OVT grid spec JSON")
    parser.add_argument("--align-to-grid", action="store_true", help="Project sparse input to grid-source geometry")
    parser.add_argument(
        "--projection-mode",
        default="cell",
        choices=["cell", "trace-key"],
        help="Grid projection mode: 4D midpoint/OVT cell or legacy trace-key matching",
    )
    parser.add_argument(
        "--midpoint-bin-mode",
        default="cmp",
        choices=["cmp", "coordinate"],
        help="How midpoint bins are defined for cell projection",
    )
    parser.add_argument(
        "--midpoint-key-columns",
        default="cmp_line,cmp",
        help="Comma-separated midpoint cell columns used when --midpoint-bin-mode cmp",
    )
    parser.add_argument(
        "--input-duplicate-policy",
        default="mean",
        choices=["mean", "error", "first"],
        help="How to combine multiple input traces in the same 4D cell",
    )
    parser.add_argument(
        "--outside-grid-policy",
        default="skip",
        choices=["skip", "error"],
        help="How to handle input 4D cells not present in the regular grid",
    )
    parser.add_argument(
        "--allow-outside-grid", action="store_true",
        help="Clip offset bins to reference grid boundaries instead of raising an error",
    )
    parser.add_argument("--missing-fill", default="zero", choices=["zero"])
    parser.add_argument("--missing-eps", type=float, default=1e-10)

    parser.add_argument("--wave-type", default="PP", choices=["PP", "PS", "pp", "ps"])
    parser.add_argument("--binning-mode", default="expert", choices=["expert", "beginner"])
    parser.add_argument("--offset-x-bin-size", type=float, default=None)
    parser.add_argument("--offset-y-bin-size", type=float, default=None)
    parser.add_argument("--source-line-interval", type=float, default=None)
    parser.add_argument("--receiver-line-interval", type=float, default=None)
    parser.add_argument("--gamma", "--effective-gamma", dest="gamma", type=float, default=2.0)
    parser.add_argument("--offset-x-shift", type=float, default=None)
    parser.add_argument("--offset-y-shift", type=float, default=None)
    parser.add_argument("--swap", action="store_true")

    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--overwrite-fields", action="store_true")
    parser.add_argument("--qc-dir", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_type == "h5":
        process_h5(args)
    else:
        if args.align_to_grid:
            raise NotImplementedError("--align-to-grid currently supports --input-type h5 only")
        process_segy(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
