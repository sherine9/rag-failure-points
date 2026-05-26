"""
Print P1 ablation table from saved result JSONs.

Usage:
    python compare_results.py
    python compare_results.py --results_dir /scratch/results
"""

import json
import os
import argparse

_ENTRIES = [
    ("no retrieval", "p1_no_retrieval.json"),
    ("always retrieve", "p1_always_retrieve.json"),
    ("selective", "p1_selective.json"),
]

_HDR = (
    f"{'Mode':<18} {'EM':>7} {'ROUGE-L':>8} {'Token F1':>9} "
    f"{'Hit Rate':>9} {'% Retr.':>8} {'n':>5}"
)
_SEP = "-" * 72


def load_payload(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare P1 ablation results")
    parser.add_argument("--results_dir", default="/scratch/results")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print("P1 ADAPTIVE RETRIEVAL — ABLATION")
    print(f"{'='*72}")
    print(_HDR)
    print(_SEP)

    for label, fname in _ENTRIES:
        data = load_payload(os.path.join(args.results_dir, fname))
        if data is None:
            print(f"{label:<18} {'(pending)':>44}")
            continue
        m = data["metrics"]
        pct = data.get("pct_retrieved", 0.0)
        hit = m.get("retrieval_hit_rate", 0.0)
        if label == "no retrieval":
            hit_str = "    n/a"
            pct_str = "  0.0%"
        else:
            hit_str = f"{hit:>9.4f}"
            pct_str = f"{pct * 100:>6.1f}%"
        print(
            f"{label:<18} {m['exact_match']:>7.4f} {m['rouge_l']:>8.4f} "
            f"{m['token_f1']:>9.4f} {hit_str:>9} {pct_str:>8} {m['n_samples']:>5}"
        )

    ablation = load_payload(os.path.join(args.results_dir, "p1_ablation.json"))
    if ablation:
        print(_SEP)
        print("Failure label counts (selective):")
        selective = ablation.get("modes", {}).get("selective", {})
        counts = selective.get("failure_label_counts", {})
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {label:<22} {count}")

    print(f"{'='*72}\n")
