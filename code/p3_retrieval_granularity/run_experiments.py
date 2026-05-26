"""
Run all P3 retrieval granularity experiments and print a comparison table.

Configs:
  fixed, semantic, small_to_big : 200k Wikipedia articles (same as baseline)
  proposition                   :  1k articles (LLM decomposition bottleneck)

Run from code/p3_retrieval_granularity/:
    python run_experiments.py
    python run_experiments.py --chunkers fixed semantic    # subset only
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_p3 import run_pipeline

EXPERIMENTS = [
    {"chunker_name": "fixed",        "n_wiki": 200_000},
    {"chunker_name": "semantic",     "n_wiki": 200_000},
    {"chunker_name": "small_to_big", "n_wiki": 200_000},
    {"chunker_name": "proposition",  "n_wiki": 1_000},   # LLM bottleneck
]

# Shared baseline results (code/README.md) included for reference
_BASELINE_METRICS = {
    "exact_match":        0.0500,
    "rouge_l":            0.1350,
    "token_f1":           0.1308,
    "retrieval_hit_rate": 0.3500,
    "n_samples":          100,
}

_HDR = f"{'Chunker':<22} {'n_wiki':>8}  {'EM':>7} {'ROUGE-L':>8} {'Token F1':>9} {'Hit Rate':>9} {'n':>5}"
_SEP = "-" * 72


def _row(name, n_wiki, m):
    return (
        f"{name:<22} {str(n_wiki):>8}  "
        f"{m['exact_match']:>7.4f} {m['rouge_l']:>8.4f} "
        f"{m['token_f1']:>9.4f} {m['retrieval_hit_rate']:>9.4f} {m['n_samples']:>5}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P3 experiments")
    parser.add_argument(
        "--chunkers", nargs="+",
        choices=["fixed", "semantic", "small_to_big", "proposition"],
        help="Run only these chunkers (default: all four)",
    )
    args = parser.parse_args()

    selected = set(args.chunkers) if args.chunkers else None
    exps = [e for e in EXPERIMENTS if selected is None or e["chunker_name"] in selected]

    all_metrics = {}
    for exp in exps:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['chunker_name']}  (n_wiki={exp['n_wiki']})")
        print(f"{'='*60}")
        _, metrics = run_pipeline(**exp)
        all_metrics[exp["chunker_name"]] = (exp["n_wiki"], metrics)

    # Final comparison table
    print(f"\n{'='*72}")
    print("P3 RETRIEVAL GRANULARITY — COMPARISON")
    print(f"{'='*72}")
    print(_HDR)
    print(_SEP)
    print(_row("baseline (fixed-200k)", 200_000, _BASELINE_METRICS))
    for name, (n_wiki, m) in all_metrics.items():
        print(_row(name, n_wiki, m))
    print(f"{'='*72}\n")
