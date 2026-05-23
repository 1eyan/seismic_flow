#!/usr/bin/env python3
"""
Parallel batch SEG-Y to H5 converter.
Reads segyPairs from dataset_config.py, distributes files across N workers,
each writing to its own temp H5. After all workers complete, merges temp
H5s into the final H5 via native h5py group copy (no re-compression).

Usage:
    # Edit dataset_config.py to list all SEGY files, then:
    python batch_segy2h5.py                                    # 4 workers, gzip=1
    python batch_segy2h5.py --num-workers 6 --gzip-level 1     # fast writes
    python batch_segy2h5.py --gzip-level 4                     # smaller files
    python batch_segy2h5.py --compute-ovt --keep-temp          # with OVT debugging
"""

import argparse
import gc
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

import h5py as h5
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_config
from Segy2H5 import organize_traces, compute_ovt_fields
from segy_schema import DEFAULT_PROFILE_NAME, get_segy_profile, parse_sort_keys, profile_names


DATASET_KEYS_FIXED = [
    'data', 'sx', 'sy', 'rx', 'ry',
    'delta', 't0',
    'shot_line', 'shot_no', 'recv_line', 'recv_no',
    'shot_stake', 'recv_stake', 'cmp', 'cmp_line', 'offset',
    'trace_idx',
]

OVT_ATTRS = (
    'mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
    'mx_origin', 'my_origin', 'hx_origin', 'hy_origin',
)


COMPRESSION_KWARGS = {
    'compression': 'gzip',
    'compression_opts': 1,
    'shuffle': True,
}
DATA_CHUNK_NTRACE = 128
DATA_CHUNK_NSAMPLE = 256


def _make_ds(g, key, data, spatial_chunk=None):
    if key == 'data' and spatial_chunk is not None:
        return g.create_dataset(key, data=data, chunks=spatial_chunk, **COMPRESSION_KWARGS)
    return g.create_dataset(key, data=data, **COMPRESSION_KWARGS)


def write_group(h5_path, group_name, block, compute_ovt=False,
                mx_bin=None, my_bin=None, hx_bin=None, hy_bin=None,
                segy_profile=None, segy_mode=None, sort_keys=None):
    """Append a group to an existing H5 file (creates if missing)."""
    with h5.File(h5_path, 'a') as h5f:
        if group_name in h5f:
            raise ValueError(f"Group '{group_name}' already exists in {h5_path}")
        g = h5f.create_group(group_name)
        if segy_profile is not None:
            g.attrs['segy_profile'] = segy_profile
        if segy_mode is not None:
            g.attrs['segy_mode'] = segy_mode
        if sort_keys is not None:
            g.attrs['sort_keys'] = ','.join(sort_keys)

        data_arr = block.get('data')
        data_chunk = None
        if data_arr is not None and data_arr.ndim == 2:
            ntr, nts = data_arr.shape
            data_chunk = (min(DATA_CHUNK_NTRACE, ntr), min(DATA_CHUNK_NSAMPLE, nts))

        for key in DATASET_KEYS_FIXED:
            if key in block:
                _make_ds(g, key, block[key], spatial_chunk=data_chunk)

        if compute_ovt:
            sx = (block['sx'].to_numpy() if hasattr(block['sx'], 'to_numpy')
                  else np.asarray(block['sx']))
            sy = (block['sy'].to_numpy() if hasattr(block['sy'], 'to_numpy')
                  else np.asarray(block['sy']))
            rx = (block['rx'].to_numpy() if hasattr(block['rx'], 'to_numpy')
                  else np.asarray(block['rx']))
            ry = (block['ry'].to_numpy() if hasattr(block['ry'], 'to_numpy')
                  else np.asarray(block['ry']))
            ovt = compute_ovt_fields(
                sx, sy, rx, ry,
                mx_bin=mx_bin, my_bin=my_bin,
                hx_bin=hx_bin, hy_bin=hy_bin,
            )
            for key, val in ovt.items():
                if key not in OVT_ATTRS:
                    g.create_dataset(key, data=val, **COMPRESSION_KWARGS)
            for key in OVT_ATTRS:
                g.attrs[key] = ovt[key]


def _worker(chunk, temp_h5, worker_id, compute_ovt,
            segy_profile, segy_mode, sort_keys,
            mx_bin, my_bin, hx_bin, hy_bin, error_queue):
    """Process a chunk of (group_name, [input_segy, ...]) entries -> temp H5."""
    start_t = time.time()
    n = len(chunk)
    try:
        for i, (group_name, entry) in enumerate(chunk.items(), 1):
            input_segy = entry[0]
            t0 = time.time()
            print(f"  [Worker {worker_id}] ({i}/{n}) {group_name}: {input_segy}", flush=True)

            block = organize_traces(input_segy, headers_df=None,
                                    sort_keys=sort_keys, mode=segy_mode,
                                    profile=segy_profile)

            write_group(temp_h5, group_name, block,
                        compute_ovt=compute_ovt,
                        mx_bin=mx_bin, my_bin=my_bin,
                        hx_bin=hx_bin, hy_bin=hy_bin,
                        segy_profile=segy_profile,
                        segy_mode=segy_mode,
                        sort_keys=sort_keys)

            del block
            gc.collect()

            dt = time.time() - t0
            print(f"  [Worker {worker_id}] ({i}/{n}) done in {dt:.1f}s", flush=True)

        elapsed = time.time() - start_t
        print(f"[Worker {worker_id}] All {n} groups done in {elapsed:.1f}s -> {temp_h5}",
              flush=True)

    except Exception:
        error_queue.put((worker_id, traceback.format_exc()))
        raise


def merge_temp_h5s(temp_paths, final_path):
    """Copy all groups from temp H5 files into a single master H5.

    Uses h5py's native group copy - O(1) per dataset, preserves compression.
    """
    seen = set()
    os.makedirs(os.path.dirname(final_path) or '.', exist_ok=True)

    with h5.File(final_path, 'w') as f_dst:
        pass

    for tp in temp_paths:
        with h5.File(tp, 'r') as f_src:
            for name in f_src.keys():
                if name in seen:
                    raise RuntimeError(
                        f"Duplicate group '{name}' in {tp} - already exists in master"
                    )
                seen.add(name)
                f_src.copy(name, final_path)
                print(f"  Merged group: {name}")

    print(f"Merged {len(seen)} groups into {final_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch SEG-Y -> H5 converter (parallel)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--compute-ovt', action='store_true',
                        help='Compute and store OVT fields')
    parser.add_argument('--mx-bin', type=float, default=None)
    parser.add_argument('--my-bin', type=float, default=None)
    parser.add_argument('--hx-bin', type=float, default=None)
    parser.add_argument('--hy-bin', type=float, default=None)
    parser.add_argument('--keep-temp', action='store_true',
                        help='Do not delete temp H5 files after merge')
    parser.add_argument('--segy-profile', choices=profile_names(),
                        default=getattr(dataset_config, 'segy_profile', DEFAULT_PROFILE_NAME),
                        help='SEG-Y header/key profile')
    parser.add_argument('--mode', choices=['fixed', 'self_computed'],
                        default=getattr(dataset_config, 'segy_mode', 'fixed'),
                        help='Header reading mode')
    parser.add_argument('--sort-keys', default=None,
                        help='Comma-separated sort keys; defaults to the active SEG-Y profile')
    parser.add_argument('--gzip-level', type=int, default=1, choices=range(1, 10),
                        help='gzip compression level 1-9 (default: 1, fastest)')
    parser.add_argument('--chunk-ntrace', type=int, default=128,
                        help='Chunk size along trace axis for data dataset (default: 128)')
    parser.add_argument('--chunk-nsample', type=int, default=256,
                        help='Chunk size along sample axis for data dataset (default: 256)')
    args = parser.parse_args()

    COMPRESSION_KWARGS['compression_opts'] = args.gzip_level

    profile = get_segy_profile(args.segy_profile)
    sort_keys = parse_sort_keys(
        args.sort_keys if args.sort_keys is not None else getattr(dataset_config, 'sort_keys', None),
        profile,
    )
    info_h5 = dataset_config.info_h5
    segyPairs = dataset_config.segyPairs
    num_workers = min(args.num_workers, len(segyPairs))

    print(f"Output H5: {info_h5}")
    print(f"Total groups: {len(segyPairs)}")
    print(f"Workers:     {num_workers}")
    print(f"Profile:     {profile.name}")
    print(f"Mode:        {args.mode}")
    print(f"Sort keys:   {sort_keys}")
    print(f"OVT:         {args.compute_ovt}")
    print(f"gzip level:  {args.gzip_level}")
    print(f"data chunk:  ({args.chunk_ntrace}, {args.chunk_nsample})")
    print(f"shuffle:     {COMPRESSION_KWARGS['shuffle']}")
    print("-" * 60)

    global DATA_CHUNK_NTRACE, DATA_CHUNK_NSAMPLE
    DATA_CHUNK_NTRACE = args.chunk_ntrace
    DATA_CHUNK_NSAMPLE = args.chunk_nsample

    keys = list(segyPairs.keys())
    chunks = []
    chunk_size = max(1, len(keys) // num_workers)
    remainder = len(keys) % num_workers
    idx = 0
    for w in range(num_workers):
        n = chunk_size + (1 if w < remainder else 0)
        chunk = {k: segyPairs[k] for k in keys[idx:idx + n]}
        chunks.append(chunk)
        idx += n

    print("Phase 1: Parallel conversion...")
    temp_dir = os.path.dirname(info_h5) or '.'
    base_name = os.path.splitext(os.path.basename(info_h5))[0]
    temp_paths = [os.path.join(temp_dir, f"{base_name}_tmp_{w}.h5") for w in range(num_workers)]

    for tp in temp_paths:
        if os.path.exists(tp):
            os.remove(tp)

    error_queue = Queue()
    processes = []
    for w in range(num_workers):
        p = Process(
            target=_worker,
            args=(chunks[w], temp_paths[w], w, args.compute_ovt,
                  profile.name, args.mode, sort_keys,
                  args.mx_bin, args.my_bin, args.hx_bin, args.hy_bin,
                  error_queue),
            name=f"worker-{w}",
        )
        processes.append(p)
        p.start()
        print(f"Started worker {w}: {len(chunks[w])} groups -> {temp_paths[w]}")

    for p in processes:
        p.join()

    errors = []
    while not error_queue.empty():
        worker_id, tb = error_queue.get()
        errors.append((worker_id, tb))

    if errors:
        print(f"\nERROR: {len(errors)} worker(s) failed:")
        for worker_id, tb in errors:
            print(f"  Worker {worker_id}:")
            print(f"  {tb.strip()}")
        for tp in temp_paths:
            if os.path.exists(tp):
                os.remove(tp)
        sys.exit(1)

    print("\nPhase 2: Merging temp H5s...")
    merge_temp_h5s(temp_paths, info_h5)

    if not args.keep_temp:
        print("\nCleaning up temp files...")
        for tp in temp_paths:
            if os.path.exists(tp):
                os.remove(tp)
                print(f"  Removed: {tp}")

    print(f"\nDone. Output: {info_h5}")


if __name__ == '__main__':
    main()
