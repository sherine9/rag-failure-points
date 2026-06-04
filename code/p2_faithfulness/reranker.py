"""
Cross-Encoder Re-Ranking — P2 Mitigation 3a (Leïla Benjelloun)
===============================================================
Re-ranks retrieved chunks by relevance before passing them to the generator,
addressing the dominant failure mode identified by the Wallat probe (81%
retrieval failure).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO passage ranking — a supervised relevance model that
  directly predicts whether a passage answers a query. Single output logit:
  higher = more relevant. This is the standard choice for RAG re-ranking.

  Earlier attempt used cross-encoder/nli-deberta-v3-small (NLI entailment
  score as a relevance proxy). This performed worse than baseline because
  NLI entailment != answer relevance: a chunk can entail question keywords
  without containing the answer, leading to worse chunk selection.
"""

import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CACHE   = "/scratch/hf_cache"
DEFAULT_TOP_K   = 5


# ── 1. Model loading ──────────────────────────────────────────────────────────

def load_reranker(model_name: str = RERANKER_MODEL, cache_dir: str = DEFAULT_CACHE):
    """Load the cross-encoder re-ranker. Returns (model, tokenizer)."""
    print(f"Loading re-ranker ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("Re-ranker ready.")
    return model, tokenizer


# ── 2. Scoring ────────────────────────────────────────────────────────────────

def score_chunks(
    question: str,
    chunks: list,
    model,
    tokenizer,
    batch_size: int = 16,
) -> list:
    """
    Score each chunk against the question using the MS MARCO relevance logit.
    Higher score = chunk more likely to answer the question.
    Returns a list of float scores, one per chunk.
    """
    device = next(model.parameters()).device
    scores = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        inputs = tokenizer(
            [question] * len(batch_chunks),
            batch_chunks,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        # MS MARCO outputs [batch, 1] — view(-1) flattens to 1D regardless of shape
        scores.extend([float(x) for x in logits.view(-1).cpu()])

    return scores


def rerank_chunks(
    question: str,
    chunks: list,
    model,
    tokenizer,
    top_k: int = DEFAULT_TOP_K,
) -> tuple:
    """
    Re-rank chunks by relevance and return the top-k.
    Returns (top_k_chunks, all_scores) for analysis.
    """
    scores = score_chunks(question, chunks, model, tokenizer)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in ranked[:top_k]]
    return top_chunks, scores


# ── 3. Full pipeline (re-rank + regenerate) ────────────────────────────────────

def run_reranking_pipeline(
    results: list,
    reranker_model,
    reranker_tokenizer,
    generator,
    top_k: int = DEFAULT_TOP_K,
) -> list:
    """
    For each sample: re-rank retrieved chunks, regenerate the answer.

    Args:
        results:              baseline pipeline results (with 'chunks', 'question')
        reranker_model:       loaded cross-encoder model
        reranker_tokenizer:   loaded cross-encoder tokenizer
        generator:            Generator instance (Mistral-7B)
        top_k:                number of top chunks to keep after re-ranking

    Returns:
        new results list with 'chunks' replaced by re-ranked top-k chunks
        and 'prediction' regenerated from those chunks
    """
    new_results = []
    for r in tqdm(results, desc="Re-ranking + regenerating"):
        top_chunks, scores = rerank_chunks(
            r['question'], r['chunks'], reranker_model, reranker_tokenizer, top_k
        )
        new_pred = generator.generate(r['question'], top_chunks)
        new_results.append({
            **r,
            'chunks':            top_chunks,
            'prediction':        new_pred,
            'rerank_scores':     [round(s, 4) for s in scores],
            'original_chunks':   r['chunks'],
            'original_prediction': r['prediction'],
        })
    return new_results


# ── 4. Save ───────────────────────────────────────────────────────────────────

def save_reranked_results(
    results: list,
    path: str = "/scratch/results/p2_reranked.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"Saved {len(results)} results to {path}")
