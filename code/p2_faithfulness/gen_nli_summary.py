"""
Generate p2_nli_summary.json
-----------------------------
Runs MiniCheck-DeBERTa NLI scoring over:
  - p2_reranked.json          → 3a_reranking  (nli_mean_unlocked: nanmean over all 100 samples)
  - p2_grounding_sweep/*.json → 3b_*          (nli_mean + n_valid: nanmean + count of non-NaN)

Output: RESULTS_DIR/p2_nli_summary.json
"""

import json
import os
import sys
import numpy as np

# Make p2_faithfulness importable
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from custom_nli import load_nli_model, evaluate_faithfulness_nli

RESULTS_DIR = "/scratch/open-project-m3-mnlp212/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "p2_nli_summary.json")

SWEEP_KEYS = {
    "P1_soft":          "3b_P1_soft",
    "P2_strict_short":  "3b_P2_strict_short",
    "P3_strict_verbose": "3b_P3_strict_verbose",
}


def nli_mean_and_n(results, scorer):
    metrics = evaluate_faithfulness_nli(results, model=scorer)
    scores = np.array(metrics["per_sample_scores"], dtype=float)
    return float(np.nanmean(scores)), int(np.sum(~np.isnan(scores)))


def main():
    scorer = load_nli_model()
    summary = {}

    # ── 3a: re-ranking ──────────────────────────────────────────────────────
    reranked_path = os.path.join(RESULTS_DIR, "p2_reranked.json")
    print(f"\nScoring 3a re-ranking: {reranked_path}")
    with open(reranked_path) as f:
        reranked_results = json.load(f)["results"]

    mean_3a, _ = nli_mean_and_n(reranked_results, scorer)
    summary["3a_reranking"] = {"nli_mean_unlocked": round(mean_3a, 4)}
    print(f"  3a nli_mean_unlocked = {mean_3a:.4f}")

    # ── 3b: grounding prompt sweep ──────────────────────────────────────────
    sweep_dir = os.path.join(RESULTS_DIR, "p2_grounding_sweep")
    for file_key, summary_key in SWEEP_KEYS.items():
        sweep_path = os.path.join(sweep_dir, f"{file_key}.json")
        print(f"\nScoring {summary_key}: {sweep_path}")
        with open(sweep_path) as f:
            sweep_results = json.load(f)["results"]

        mean_3b, n_valid = nli_mean_and_n(sweep_results, scorer)
        summary[summary_key] = {
            "nli_mean": round(mean_3b, 4),
            "n_valid":  n_valid,
        }
        print(f"  {summary_key}: nli_mean={mean_3b:.4f}, n_valid={n_valid}")

    # ── Save ────────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
