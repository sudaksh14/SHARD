"""
Energy per inference from benchmark JSON files (paper Table 3).

Energy per run:   E_run [mJ] = power_mW / 1000 [W] * latency_ms [ms]
(i.e. the jtop power snapshot of a run multiplied by that run's latency), reported as mean +- std (population std,
`numpy.std` default) over the runs of each level and over all runs of the file ("overall").

    python -m utils.jetson_harness.energy_stats jetson_runs/run1/shard_onnx.json
    python -m utils.jetson_harness.energy_stats --table jetson_runs/run1/*.json     # one row per file
    python -m utils.jetson_harness.energy_stats --json out.json jetson_runs/run1/shard_onnx.json

Runs whose power sample is null (no jtop on the machine that produced the file) are skipped, and a file without any
power sample yields no energy figures at all; nothing is imputed.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def compute_energy_stats(path) -> dict:
    """{label: {avg_energy_mJ, std_energy_mJ, num_runs}} plus an 'overall' entry (mJ per inference)."""
    results, pooled = {}, []
    for label, m in load(path).items():
        lat_ms = np.array([np.nan if v is None else v for v in m["latency_ms"]], dtype=float)
        p_mw = np.array([np.nan if v is None else v for v in m["power_mW"]], dtype=float)
        e_mj = (p_mw / 1000.0) * lat_ms          # W * ms = mJ
        e_mj = e_mj[~np.isnan(e_mj)]
        if e_mj.size == 0:
            results[label] = {"avg_energy_mJ": None, "std_energy_mJ": None, "num_runs": 0}
            continue
        pooled.extend(e_mj.tolist())
        results[label] = {"avg_energy_mJ": float(e_mj.mean()), "std_energy_mJ": float(e_mj.std()), "num_runs": int(e_mj.size)}
    if pooled:
        arr = np.array(pooled)
        results["overall"] = {"avg_energy_mJ": float(arr.mean()), "std_energy_mJ": float(arr.std()), "num_runs": int(arr.size)}
    return results


def fmt(stat):
    if stat["avg_energy_mJ"] is None:
        return "n/a (no power samples)"
    return f"{stat['avg_energy_mJ']:.1f} ± {stat['std_energy_mJ']:.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--table", action="store_true", help="compact one-row-per-file table")
    ap.add_argument("--json", default=None, help="also write all statistics to this JSON file")
    args = ap.parse_args(argv)

    out = {}
    for path in args.files:
        out[path] = compute_energy_stats(path)
    if args.table:
        for path, res in out.items():
            cells = [fmt(v) for k, v in res.items() if k != "overall"]
            overall = fmt(res["overall"]) if "overall" in res else "n/a"
            print(f"{path}\n   levels: {' | '.join(cells)}\n   overall: {overall}")
    else:
        for path, res in out.items():
            print(f"==== {path} ====")
            for label, s in res.items():
                print(f"{label}:  runs={s['num_runs']}  energy [mJ] = {fmt(s)}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
