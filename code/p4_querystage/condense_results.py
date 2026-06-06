"""
Condense raw P4 result JSONs (with full 2610 records) into lightweight
versions suitable for committing to the repo.

Strategy: uniform sampling of records.
  - Aggregated metrics are preserved as-is (they were computed on the full
    record set by run_m3_ablation / run_m3_test BEFORE this script runs).
  - Records are downsampled to `sample_n` evenly-spaced records (default 200)
    and stored under the SAME key `results` as the original JSON, so notebook
    code that reads `data["results"]` works unchanged.
  - All chunk text is preserved for sampled records → bias diagnostic and
    case study work as expected.

Usage:
    python condense_results.py \\
        --src /scratch/results/p4 \\
        --dst /scratch/open-project-m3-mnlp212/results/p4 \\
        --sample 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def condense_one(src_path: str, dst_path: str, sample_n: int):
    """Condense a single JSON file."""
    with open(src_path) as f:
        data = json.load(f)

    # Copy everything EXCEPT the heavy "results" field; we'll add a sampled
    # version of it below.
    out = {k: v for k, v in data.items() if k != "results"}

    if "results" in data and data["results"]:
        records = data["results"]
        n_full = len(records)

        if n_full > sample_n:
            # Uniform sampling: keep every `step`-th record so the sample
            # spans the full evaluation slice (rather than only the head).
            step = n_full // sample_n
            out["results"] = [records[i] for i in range(0, n_full, step)][:sample_n]
        else:
            # Already small enough — keep as-is
            out["results"] = records

        # Track the original size so the notebook can mention it in the
        # methodology paragraph (e.g. "sampled from N=2610 records").
        out["results_total_records"] = n_full

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out, f, indent=2)

    src_mb = os.path.getsize(src_path) / 1024 / 1024
    dst_mb = os.path.getsize(dst_path) / 1024 / 1024
    return src_mb, dst_mb


def main(src_root: str, dst_root: str, sample_n: int):
    """Walk src_root, condense every JSON to dst_root keeping the same structure."""
    pattern = os.path.join(src_root, "**", "*.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No JSON files found under {src_root}")
        return

    total_src, total_dst = 0.0, 0.0
    for src in paths:
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_root, rel)
        src_mb, dst_mb = condense_one(src, dst, sample_n)
        total_src += src_mb
        total_dst += dst_mb
        print(f"  {rel:50s}  {src_mb:6.1f} MB → {dst_mb:6.2f} MB")
    print(f"  {'TOTAL':50s}  {total_src:6.1f} MB → {total_dst:6.2f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src",    required=True, help="Source dir (e.g., /scratch/results/p4)")
    p.add_argument("--dst",    required=True, help="Destination dir (committable, e.g., <repo>/results/p4)")
    p.add_argument("--sample", type=int, default=200,
                   help="Number of records to keep per JSON (default 200)")
    args = p.parse_args()
    main(args.src, args.dst, args.sample)