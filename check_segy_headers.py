#!/usr/bin/env python3
"""Quick SEG-Y trace header inspection and comparison using seisio.

Single file:  python3 check_segy_headers.py a.sgy
Compare two:  python3 check_segy_headers.py a.sgy b.sgy
"""

import argparse
import sys
import logging
import numpy as np
from pathlib import Path
from seisio.segy import Reader

logging.getLogger("seisio").setLevel(logging.WARNING)

DEFAULT_FIELDS = [
    "tracl", "tracr", "fldr", "ep", "cdp", "iline", "xline",
    "sx", "sy", "gx", "gy", "cdpx", "cdpy",
    "offset", "scalco", "scalel", "ns", "dt", "delrt",
]

BINHEAD_INTERESTING = [
    "job", "line", "reel", "nt", "auxnt", "dt", "dtorig", "ns", "nsorig",
    "format", "fold", "tsort", "vsc", "msys", "ntxthead", "maxtrhead",
    "ntfile", "byteoff", "ntrailer", "segymaj", "segymin", "fixed", "timbas",
]

HEADER_BATCH_SIZE = 50000
TRACE_BATCH_SIZE = 10000


def _load_reader(path):
    try:
        return Reader(path)
    except Exception as e:
        print(f"Error opening {path}: {e}")
        sys.exit(1)


def _bh_get(reader, key):
    bh = reader.binhead
    if key not in bh.dtype.names:
        return None
    val = bh[0][key]
    return val.item() if hasattr(val, "item") else val


def _resolve_fields(reader, field_str):
    if field_str:
        fields = [f.strip() for f in field_str.split(",") if f.strip()]
    else:
        fields = list(DEFAULT_FIELDS)
    dtype_names = set(reader.read_headers(0)[0].dtype.names)
    resolved = [f for f in fields if f in dtype_names]
    missing = [f for f in fields if f not in dtype_names]
    if missing:
        print(f"Note: fields not found: {missing}")
    return resolved


def _format_row(idx, header, fields, widths):
    row = f"{idx:>{widths['_idx']}}  "
    row += "  ".join(f"{header[f]!s:>{widths[f]}}" for f in fields)
    return row


def _build_indices(n, num_traces, all_traces, tail_only):
    if all_traces:
        return list(range(n))
    head_n = min(num_traces, n)
    if tail_only:
        return list(range(max(0, n - num_traces), n))
    if n <= head_n * 2:
        return list(range(n))
    return list(range(head_n)) + list(range(n - head_n, n))


def _compute_widths(fields, headers, indices, n):
    widths = {"_idx": max(6, len(str(n)))}
    for f in fields:
        widths[f] = max(len(f), 8)
        for i in range(len(indices)):
            widths[f] = max(widths[f], len(str(headers[i][f])))
    return widths


def _print_table_header(fields, widths):
    line = f"{'#':>{widths['_idx']}}  "
    line += "  ".join(f"{f:>{widths[f]}}" for f in fields)
    print("    " + line)
    print("    " + "-" * len(line))


# ════════════════════════════════════════════════════════════════════════════
# Single file inspection
# ════════════════════════════════════════════════════════════════════════════

def print_file_info(reader):
    t_ms = reader.vaxis[-1] * 1000 if hasattr(reader, "vaxis") else reader.nsamples * reader.vsi / 1e6
    print(f"File: {reader.file}")
    print(f"  Size: {reader.fsize:,} bytes | Traces: {reader.ntraces:,} | "
          f"Samples: {reader.nsamples} | dt: {reader.vsi} us")
    print(f"  Duration: {t_ms:.1f} ms | "
          f"Format: {_bh_get(reader, 'format')} | "
          f"Endian: {reader.endianess} | "
          f"Header ext1: {getattr(reader, 'thext1', False)}")
    if _bh_get(reader, "ntfile"):
        print(f"  ntfile (declared traces): {_bh_get(reader, 'ntfile')}")


def print_binhead(reader):
    nonzero = []
    for k in BINHEAD_INTERESTING:
        v = _bh_get(reader, k)
        if v is not None and v != 0:
            nonzero.append((k, v))
    if nonzero:
        print("  Binary header (nonzero):")
        for k, v in nonzero:
            print(f"    {k}: {v}")


def print_text_header(reader):
    try:
        txt = reader.get_txthead()
        if txt:
            lines = txt.strip().split("\n")
            print("  Text header snippet:")
            for line in lines[:8]:
                print(f"    {line.rstrip()}")
            if len(lines) > 8:
                print(f"    ... ({len(lines)} lines total)")
    except Exception:
        pass


def print_trace_header_table(reader, args):
    n = reader.ntraces
    indices = _build_indices(n, args.num_traces, args.all_traces, args.tail_only)
    fields = _resolve_fields(reader, args.fields)
    if not indices or not fields:
        return
    print(f"  Trace headers ({len(indices)} of {n}):")
    headers = reader.read_headers(*indices)
    widths = _compute_widths(fields, headers, indices, n)
    _print_table_header(fields, widths)
    for i, idx in enumerate(indices):
        print(_format_row(idx, headers[i], fields, widths))
    print()


def print_field_statistics(reader, args):
    n = reader.ntraces
    fields = _resolve_fields(reader, args.fields)
    if not fields:
        return
    print("  Field statistics (all traces):")
    # Use batch reading for efficiency
    stats = {f: {"min": np.inf, "max": -np.inf, "unique": set(), "sum": 0.0} for f in fields}
    for batch in reader.batches_of_headers(HEADER_BATCH_SIZE):
        for f in fields:
            vals = batch[f]
            vmin = np.min(vals)
            vmax = np.max(vals)
            if vmin < stats[f]["min"]:
                stats[f]["min"] = vmin.item() if hasattr(vmin, "item") else vmin
            if vmax > stats[f]["max"]:
                stats[f]["max"] = vmax.item() if hasattr(vmax, "item") else vmax
            # Unique: sample for large counts (keep track of first 1000 uniques)
            if len(stats[f]["unique"]) < 1000:
                for v in np.unique(vals):
                    stats[f]["unique"].add(v.item() if hasattr(v, "item") else v)
            stats[f]["sum"] += float(np.sum(vals))
    for f in fields:
        s = stats[f]
        n_uniq = len(s["unique"])
        u_str = f"{n_uniq}" if n_uniq < 1000 else f"{n_uniq}+"
        print(f"    {f:>12s}: min={s['min']:>12}, max={s['max']:>12}, "
              f"unique={u_str:>6}, "
              f"mean={s['sum'] / n:>14.1f}")


def check_empty_traces(reader, label=""):
    n = reader.ntraces
    threshold = 1e-10
    empty = 0
    near_zero = 0
    ns = reader.nsamples
    for batch in reader.batches_of_headers(TRACE_BATCH_SIZE):
        bs = len(batch)
        # For each trace in the batch, read its trace data
        # Read all traces for this batch range in one call
        start = batch[0]["tracl"]  # Not reliable; use index
        ...
    # Batch-based empty check is complex; use trace-by-trace with batches
    for start in range(0, n, TRACE_BATCH_SIZE):
        nt = min(TRACE_BATCH_SIZE, n - start)
        try:
            batch_traces = reader.read_batch_of_traces(start, nt)
            for i in range(nt):
                trc = batch_traces[i]
                if np.max(np.abs(trc)) < threshold:
                    empty += 1
                if np.sum(np.abs(trc)) < threshold * ns:
                    near_zero += 1
        except Exception:
            break
    prefix = f"Empty traces ({label}): " if label else "Empty traces: "
    print(f"  {prefix}{empty} / {n}")
    print(f"  Near-zero traces: {near_zero} / {n}")


def inspect_single(reader, args):
    print("=" * 78)
    print("SEG-Y Header Inspection")
    print("=" * 78)
    print_file_info(reader)
    print_binhead(reader)
    print_text_header(reader)
    print()
    print_trace_header_table(reader, args)
    print_field_statistics(reader, args)
    if args.check_empty:
        print("-" * 78)
        check_empty_traces(reader)
    print("=" * 78)


# ════════════════════════════════════════════════════════════════════════════
# Two file comparison
# ════════════════════════════════════════════════════════════════════════════

def _batch_field_stats(reader, fields):
    """Collect min/max/unique/sum for each field using batch reading."""
    stats = {f: {"min": np.inf, "max": -np.inf, "unique": set(), "sum": 0.0} for f in fields}
    count = 0
    for batch in reader.batches_of_headers(HEADER_BATCH_SIZE):
        bs = len(batch)
        count += bs
        for f in fields:
            vals = batch[f]
            vmin = np.min(vals)
            vmax = np.max(vals)
            if vmin < stats[f]["min"]:
                stats[f]["min"] = vmin.item() if hasattr(vmin, "item") else vmin
            if vmax > stats[f]["max"]:
                stats[f]["max"] = vmax.item() if hasattr(vmax, "item") else vmax
            if len(stats[f]["unique"]) < 1000:
                for v in np.unique(vals):
                    stats[f]["unique"].add(v.item() if hasattr(v, "item") else v)
            stats[f]["sum"] += float(np.sum(vals))
    return stats, count


def _first_header(reader):
    for h in reader.batches_of_headers(1):
        return h[0]
    return None


def compare_two(reader_a, reader_b, args):
    print("=" * 78)
    print("SEG-Y Header Comparison")
    print("=" * 78)

    ra, rb = reader_a, reader_b

    # ── Basic info ──
    print(f"  {'':>38s}  {'A':>15s}  {'B':>15s}  {'diff':>10s}")
    print("  " + "-" * 76)
    ta_ms = ra.vaxis[-1] * 1000
    tb_ms = rb.vaxis[-1] * 1000
    rows = [
        ("File", Path(ra.file).name, Path(rb.file).name, ""),
        ("Size (bytes)", f"{ra.fsize:,}", f"{rb.fsize:,}",
         f"{rb.fsize - ra.fsize:+,}"),
        ("Traces", str(ra.ntraces), str(rb.ntraces),
         f"{rb.ntraces - ra.ntraces:+,}"),
        ("Samples", str(ra.nsamples), str(rb.nsamples),
         f"{rb.nsamples - ra.nsamples:+,}"),
        ("dt (us)", str(ra.vsi), str(rb.vsi),
         f"{rb.vsi - ra.vsi:+,}"),
        ("Duration (ms)", f"{ta_ms:.1f}", f"{tb_ms:.1f}",
         f"{tb_ms - ta_ms:+.1f}"),
        ("Format", str(_bh_get(ra, "format")), str(_bh_get(rb, "format")), ""),
        ("Endian", ra.endianess, rb.endianess, ""),
    ]
    for label, a, b, delta in rows:
        marker = " *" if (a != b and delta == "") else ""
        print(f"  {label:>38s}: {a:>15s}  {b:>15s}  {delta:>10s}{marker}")
    print()

    # ── Binary header diff ──
    all_bh = sorted(set(ra.binhead.dtype.names) | set(rb.binhead.dtype.names))
    diffs = []
    for k in all_bh:
        va = ra.binhead[0][k].item() if hasattr(ra.binhead[0][k], "item") else ra.binhead[0][k]
        vb = rb.binhead[0][k].item() if hasattr(rb.binhead[0][k], "item") else rb.binhead[0][k]
        if va != vb:
            diffs.append((k, va, vb))
    if diffs:
        print(f"  Binary header differences ({len(diffs)}):")
        print(f"    {'Field':>15s}: {'A':>15s}  {'B':>15s}")
        for k, va, vb in diffs:
            print(f"    {k:>15s}: {str(va):>15s}  {str(vb):>15s}")
    else:
        print("  Binary header: identical")
    print()

    # ── Trace header diff: first N side-by-side ──
    fields = _resolve_fields(ra, args.fields)
    if fields:
        na, nb = ra.ntraces, rb.ntraces
        nshow = args.num_traces
        widths = {"_idx": max(6, len(str(max(na, nb) - 1)))}
        for f in fields:
            widths[f] = max(len(f), 8)

        print(f"  Trace headers (first {nshow} traces):")
        print(f"    {'#':>{widths['_idx']}} "
              + "  ".join(f"{f:>{widths[f]}}" for f in fields)
              + f"    {'#':>{widths['_idx']}} "
              + "  ".join(f"{f:>{widths[f]}}" for f in fields)
              + f"  diff?")
        print("    " + "-" * 80)

        a_headers = ra.read_headers(*list(range(min(nshow, na)))) if na > 0 else []
        b_headers = rb.read_headers(*list(range(min(nshow, nb)))) if nb > 0 else []
        max_n = min(nshow, max(na, nb))
        for i in range(max_n):
            row_a, row_b = ("--", "--")
            differs = False
            if i < na:
                row_a = _format_row(i, a_headers[i], fields, widths)
            else:
                row_a = f"{'--':>{widths['_idx']}}  " + "  ".join(
                    "--".rjust(widths[f]) for f in fields)
            if i < nb:
                row_b = _format_row(i, b_headers[i], fields, widths)
            else:
                row_b = f"{'--':>{widths['_idx']}}  " + "  ".join(
                    "--".rjust(widths[f]) for f in fields)
            if i < na and i < nb:
                for f in fields:
                    if a_headers[i][f] != b_headers[i][f]:
                        differs = True
                        break
            marker = " *" if differs else ""
            print(f"    {row_a}    {row_b}{marker}")
        print()

    # ── Field statistics comparison (batch reading) ──
    print("  Field statistics comparison (all traces):")
    print(f"    {'Field':>14s}  {'min A':>14s}  {'min B':>14s}  "
          f"{'max A':>14s}  {'max B':>14s}  "
          f"{'uniq A':>8s}  {'uniq B':>8s}")
    print("    " + "-" * 80)
    sa, ca = _batch_field_stats(ra, fields)
    sb, cb = _batch_field_stats(rb, fields)
    for f in fields:
        min_a, min_b = sa[f]["min"], sb[f]["min"]
        max_a, max_b = sa[f]["max"], sb[f]["max"]
        ua, ub = len(sa[f]["unique"]), len(sb[f]["unique"])
        ua_s = f"{ua}" if ua < 1000 else f"{ua}+"
        ub_s = f"{ub}" if ub < 1000 else f"{ub}+"
        diff_marker = ""
        if min_a != min_b or max_a != max_b or ua != ub:
            diff_marker = " *"
        print(f"    {f:>14s}: {min_a:>14}  {min_b:>14}  "
              f"{max_a:>14}  {max_b:>14}  "
              f"{ua_s:>8s}  {ub_s:>8s}{diff_marker}")
    print()

    # ── Empty trace check ──
    if args.check_empty:
        print("-" * 78)
        check_empty_traces(ra, "A")
        check_empty_traces(rb, "B")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Quickly inspect or compare SEG-Y file trace headers using seisio"
    )
    parser.add_argument("segy_a", help="Path to first SEG-Y file")
    parser.add_argument("segy_b", nargs="?", default=None,
                        help="Path to second SEG-Y file (optional, enables diff mode)")
    parser.add_argument("-n", "--num-traces", type=int, default=5,
                        help="Number of traces to display from head/tail (default: 5)")
    parser.add_argument("--tail-only", action="store_true",
                        help="Show tail traces only (single-file mode)")
    parser.add_argument("--all-traces", action="store_true",
                        help="Show all traces (caution: large output)")
    parser.add_argument("--check-empty", action="store_true",
                        help="Check for empty (all-zero) traces")
    parser.add_argument("--fields", default=None,
                        help="Comma-separated header fields, e.g. sx,sy,gx,gy,iline,xline")
    args = parser.parse_args()

    if args.segy_b:
        compare_two(_load_reader(args.segy_a), _load_reader(args.segy_b), args)
    else:
        inspect_single(_load_reader(args.segy_a), args)


if __name__ == "__main__":
    main()
