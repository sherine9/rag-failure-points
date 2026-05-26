"""
NLI-based Faithfulness Evaluation — P2 Track (Leïla Benjelloun)
===============================================================
Uses MiniCheck-DeBERTa-v3-Large via its dedicated pip package.

Why the dedicated package matters:
  Loading `lytang/MiniCheck-Deberta-v3-Large` with vanilla AutoModelForSequenceClassification
  gives constant near-1 scores for all inputs — the model requires MiniCheck's own
  inference pipeline (sentence-level document splitting, special tokenization, etc.).
  The minicheck package is the only correct way to use this model.

MiniCheck pipeline (internal):
  - Splits the document into sentences using NLTK sent_tokenize
  - Scores each claim against each document sentence
  - Returns max P(supported) across all sentences as the document-level support score
  This natively handles long contexts without truncation issues.

Our pipeline per sample:
  1. Split the answer into sentence-level claims
  2. Concatenate all retrieved chunks into one context string
  3. For each claim: call scorer.score(docs=[context], claims=[claim])
  4. Faithfulness = fraction of claims where P(supported) >= threshold

Install:
  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
  python -c "import nltk; nltk.download('punkt_tab')"
"""

import json
import re
import os
import numpy as np
from tqdm import tqdm
from minicheck.minicheck import MiniCheck


NLI_MODEL = "deberta-v3-large"          # MiniCheck model name (not HF repo ID)
MINICHECK_CACHE = "/scratch/hf_cache/minicheck"
ENTAILMENT_THRESHOLD = 0.5


# ── 1. Model loading ──────────────────────────────────────────────────────────

def load_nli_model(model_name: str = NLI_MODEL, cache_dir: str = MINICHECK_CACHE):
    """
    Load MiniCheck scorer.  Must call before evaluate_faithfulness_nli().

    Args:
        model_name: one of 'deberta-v3-large', 'roberta-large',
                    'flan-t5-large', 'Bespoke-MiniCheck-7B'
        cache_dir:  where model weights are cached

    Returns:
        MiniCheck scorer instance (pass as `model` to evaluate_faithfulness_nli)
    """
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Loading MiniCheck scorer ({model_name})...")
    scorer = MiniCheck(model_name=model_name, cache_dir=cache_dir)
    print("MiniCheck scorer ready.")
    return scorer


# ── 2. Claim splitting ─────────────────────────────────────────────────────────

def split_into_claims(answer: str) -> list[str]:
    """
    Split an answer string into sentence-level atomic claims.
    Filters very short fragments (< 10 chars) that can't form a verifiable claim.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── 3. Per-sample faithfulness ─────────────────────────────────────────────────

def compute_nli_faithfulness(
    chunks: list[str],
    answer: str,
    scorer: MiniCheck,
    threshold: float = ENTAILMENT_THRESHOLD,
) -> dict:
    """
    Compute MiniCheck-based faithfulness for a single (chunks, answer) pair.

    Concatenates all retrieved chunks into one context string.
    MiniCheck internally splits the context into sentences and returns
    max P(supported) across sentences — no truncation issues.

    Returns:
        dict with keys: score, n_claims, claims, claim_scores
    """
    claims = split_into_claims(answer)
    if not claims:
        # Answer is too short to contain a verifiable claim (e.g. "Unknown", "Judges").
        # MiniCheck was trained on 12–19 word claims; scoring a bare noun phrase is
        # undefined. Return NaN so these samples are excluded from the mean, matching
        # how RAGAS handles its 3 LLM-parsing failures.
        return {"score": float("nan"), "n_claims": 0, "claims": [], "claim_scores": []}

    context = " ".join(chunks)

    # scorer.score() takes parallel lists: claim[i] checked against docs[i]
    _, raw_probs, _, _ = scorer.score(
        docs=[context] * len(claims),
        claims=claims,
    )

    claim_scores = list(raw_probs)
    supported = sum(1 for s in claim_scores if s >= threshold)
    faithfulness = supported / len(claims)

    return {
        "score": faithfulness,
        "n_claims": len(claims),
        "claims": claims,
        "claim_scores": [round(float(s), 4) for s in claim_scores],
    }


# ── 4. Batch evaluation ────────────────────────────────────────────────────────

def evaluate_faithfulness_nli(
    results: list[dict],
    model: MiniCheck = None,
    tokenizer=None,       # unused — kept for API parity with custom_ragas
    device: str = None,   # unused — MiniCheck handles device internally
    threshold: float = ENTAILMENT_THRESHOLD,
    model_name: str = NLI_MODEL,
) -> dict:
    """
    Compute MiniCheck-based faithfulness over the full result list.

    Args:
        results:    list of pipeline dicts (keys: question, prediction, chunks, ...)
        model:      MiniCheck scorer (loads deberta-v3-large if None)
        threshold:  minimum P(supported) to count a claim as faithful
        model_name: used only if model is None

    Returns:
        metrics dict with mean/std/min/max and per-sample lists
    """
    if model is None:
        model = load_nli_model(model_name)

    print(f"Running MiniCheck NLI faithfulness on {len(results)} samples (threshold={threshold})...")
    per_sample_scores = []
    per_sample_details = []

    for r in tqdm(results, desc="NLI Faithfulness"):
        detail = compute_nli_faithfulness(r["chunks"], r["prediction"], model, threshold)
        per_sample_scores.append(detail["score"])
        per_sample_details.append(detail)

    scores = np.array(per_sample_scores, dtype=float)
    n_valid = int(np.sum(~np.isnan(scores)))
    n_nan   = int(np.sum( np.isnan(scores)))
    output = {
        "faithfulness_mean":   round(float(np.nanmean(scores)), 4),
        "faithfulness_std":    round(float(np.nanstd(scores)),  4),
        "faithfulness_min":    round(float(np.nanmin(scores)),  4),
        "faithfulness_max":    round(float(np.nanmax(scores)),  4),
        "per_sample_scores":   per_sample_scores,
        "per_sample_details":  per_sample_details,
        "n_samples":           len(results),
        "n_valid":             n_valid,
        "n_nan":               n_nan,
        "model":               f"MiniCheck/{model_name}",
        "threshold":           threshold,
    }

    print(f"\n{'='*45}")
    print(f"  NLI Faithfulness Results  (MiniCheck-{model_name})")
    print(f"{'='*45}")
    print(f"  Mean:  {output['faithfulness_mean']:.4f}")
    print(f"  Std:   {output['faithfulness_std']:.4f}")
    print(f"  Min:   {output['faithfulness_min']:.4f}")
    print(f"  Max:   {output['faithfulness_max']:.4f}")
    print(f"  N:     {n_valid} scored  ({n_nan} skipped — no verifiable claims)")
    print(f"{'='*45}\n")

    return output


# ── 5. Result annotation helpers ───────────────────────────────────────────────

def annotate_results_with_faithfulness_nli(
    results: list[dict],
    nli_metrics: dict,
) -> list[dict]:
    """Attach NLI faithfulness scores and claim-level details to result dicts."""
    for r, score, detail in zip(
        results,
        nli_metrics["per_sample_scores"],
        nli_metrics["per_sample_details"],
    ):
        r["faithfulness_nli_score"] = score
        r["faithfulness_nli_claims"] = detail.get("claims", [])
        r["faithfulness_nli_claim_scores"] = detail.get("claim_scores", [])
    return results


def get_unfaithful_examples_nli(
    results: list[dict],
    threshold: float = 0.5,
    n: int = 10,
) -> list[dict]:
    """Return the n worst-scoring examples by NLI faithfulness."""
    unfaithful = [
        r for r in results
        if r.get("faithfulness_nli_score", 1.0) < threshold
    ]
    unfaithful.sort(key=lambda x: x["faithfulness_nli_score"])
    return unfaithful[:n]


# ── 6. Save ───────────────────────────────────────────────────────────────────

def save_faithfulness_results_nli(
    metrics: dict,
    results: list[dict],
    path: str = "/scratch/results/faithfulness_nli.json",
) -> None:
    """Save NLI faithfulness results (metrics + annotated results) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"Saved to {path}")


# ── 7. CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_PATH = "/scratch/results/baseline.json"
    OUTPUT_PATH  = "/scratch/results/faithfulness_nli.json"

    print(f"Loading results from {RESULTS_PATH}...")
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    results = data["results"]

    scorer = load_nli_model()
    nli_metrics = evaluate_faithfulness_nli(results, model=scorer)
    results = annotate_results_with_faithfulness_nli(results, nli_metrics)
    save_faithfulness_results_nli(nli_metrics, results, path=OUTPUT_PATH)

    bad = get_unfaithful_examples_nli(results, threshold=0.5, n=5)
    print(f"\nTop {len(bad)} most unfaithful examples (NLI):")
    for ex in bad:
        print(f"\n  Q:        {ex['question']}")
        print(f"  Pred:     {ex['prediction']}")
        print(f"  NLI score:{ex['faithfulness_nli_score']:.2f}")
        print(f"  Claims:   {ex['faithfulness_nli_claims']}")
        print(f"  Scores:   {ex['faithfulness_nli_claim_scores']}")
