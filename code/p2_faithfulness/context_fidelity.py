"""
Context-Fidelity Output Filter — P2 Mitigation 3c (Leïla Benjelloun)
=====================================================================
Architecturally distinct from 3b (which changes the input prompt):
3c is a POST-GENERATION filter applied to the model's output.

Pipeline: P1 grounding prompt (3b) → generate → NLI filter (3c)
  If MiniCheck scores the generated answer below the faithfulness
  threshold, replace it with "I don't know" before returning.

This directly targets the 5 confabulation cases identified by the Wallat
probe: even when the model generates a confident parametric answer,
the filter catches it as ungrounded and forces a refusal.

No new Mistral-7B generation is needed — the filter is applied to the
already-generated P1 results using the MiniCheck scorer already loaded.

Diagnostic value: compare P1-alone vs P1+filter on:
  (a) Overall NLI faithfulness mean
  (b) Confabulation rate (answers that are specific facts, not refusals)
  (c) Specifically the 5 Wallat-probe confabulation cases
"""

import json
import os
import math


REFUSAL_PATTERNS = [
    'context does not provide', 'unknown', 'not mentioned',
    'no information', 'cannot find', 'not provided', 'no passage',
    "i don't know",
]


def is_refusal(text: str) -> bool:
    return any(p in text.lower() for p in REFUSAL_PATTERNS)


def apply_nli_filter(
    results: list,
    nli_scores: list,
    threshold: float = 0.5,
    replacement: str = "I don't know",
) -> list:
    """
    Post-process generated answers: replace any answer with NLI score
    below threshold with `replacement`, unless it's already a refusal.

    Args:
        results:    list of result dicts (with 'prediction', 'chunks', etc.)
        nli_scores: per-sample NLI scores from evaluate_faithfulness_nli()
        threshold:  faithfulness threshold (default 0.5, same as elsewhere)
        replacement: string to substitute for filtered answers

    Returns:
        new results list with low-faith answers replaced
    """
    filtered = []
    n_replaced = 0

    for r, score in zip(results, nli_scores):
        failed  = isinstance(score, float) and (math.isnan(score) or score < threshold)
        already_refusal = is_refusal(r["prediction"])

        if failed and not already_refusal:
            new_pred = replacement
            n_replaced += 1
        else:
            new_pred = r["prediction"]

        filtered.append({
            **r,
            "prediction":           new_pred,
            "prediction_pre_filter": r["prediction"],
            "nli_score_pre_filter":  score,
            "filter_applied":        (new_pred != r["prediction"]),
        })

    print(f"NLI filter applied: {n_replaced} answers replaced "
          f"(threshold={threshold})")
    return filtered


def save_fidelity_results(
    results: list,
    path: str = "/scratch/results/p2_fidelity.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Saved {len(results)} results to {path}")
