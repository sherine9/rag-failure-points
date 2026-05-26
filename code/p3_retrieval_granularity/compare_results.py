"""
Load saved result JSONs and print a side-by-side comparison table.

Looks for results in /scratch/results/ by default.
Missing files are shown as (pending) so partial runs are still readable.

Run from anywhere:
    python compare_results.py
    python compare_results.py --results_dir /path/to/results
"""
import json
import os
import argparse

# (label, filename, n_wiki used) — order matches the experimental narrative
_ENTRIES = [
    ("baseline (fixed-200k)", "baseline.json",         200_000),
    ("fixed",                 "p3_fixed.json",          200_000),
    ("semantic",              "p3_semantic.json",       200_000),
    ("small_to_big",          "p3_small_to_big.json",  200_000),
    ("proposition",           "p3_proposition.json",     1_000),
]

_HDR = f"{'Chunker':<22} {'n_wiki':>8}  {'EM':>7} {'ROUGE-L':>8} {'Token F1':>9} {'Hit Rate':>9} {'n':>5}"
_SEP = "-" * 72


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["metrics"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare P3 experiment results")
    parser.add_argument("--results_dir", default="/scratch/results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print("P3 RETRIEVAL GRANULARITY — RESULTS")
    print(f"{'='*72}")
    print(_HDR)
    print(_SEP)

    for label, fname, n_wiki in _ENTRIES:
        m = load_metrics(os.path.join(args.results_dir, fname))
        if m is None:
            print(f"{label:<22} {str(n_wiki):>8}  {'(pending)':>44}")
        else:
            print(
                f"{label:<22} {str(n_wiki):>8}  "
                f"{m['exact_match']:>7.4f} {m['rouge_l']:>8.4f} "
                f"{m['token_f1']:>9.4f} {m['retrieval_hit_rate']:>9.4f} {m['n_samples']:>5}"
            )

    print(f"{'='*72}\n")
