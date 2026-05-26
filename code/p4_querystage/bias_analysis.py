"""
Bias diagnostics inspired by Goyal et al. 2026, "Masking or Mitigating?
Deconstructing the Impact of Query Rewriting on Retriever Biases in RAG".

The paper measures the Spearman correlation between FAISS retrieval scores
and surface-level features of retrieved documents (length, lexical overlap,
answer position, query-term repetition). A retriever that ranks purely on
semantic relevance should show near-zero correlation; a strong correlation
indicates the retriever is exploiting a surface feature.

We can't reproduce ColDeR's treatment-control pairs without their controlled
corpus, but we CAN apply the same correlation-based diagnostic on our NQ Open
runs to ask: *as we move from baseline → HyDE → Query2doc → Rewrite, does
the retriever's reliance on surface features shrink (decorrelation) or just
get noisier (variance-based reduction)?* — the central distinction Goyal et al.
draw between "real" and "apparent" bias mitigation.

Usage
-----
    from bias_analysis import (
        feature_score_correlations,
        retrieved_chunk_stats,
    )

    rho = feature_score_correlations(records)
    stats = retrieved_chunk_stats(records)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

import numpy as np


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ── Surface features (one per chunk) ──────────────────────────────────────────
def _length_tokens(chunk: str) -> float:
    return float(len(_tokenize(chunk)))


def _jaccard(query: str, chunk: str) -> float:
    q = set(_tokenize(query))
    c = set(_tokenize(chunk))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def _answer_position(chunk: str, gold_answers: List[str]) -> float:
    """Normalised position of first gold-answer occurrence, or NaN if absent."""
    chunk_low = chunk.lower()
    for ans in gold_answers:
        if not ans:
            continue
        idx = chunk_low.find(ans.lower())
        if idx >= 0:
            return idx / max(len(chunk_low), 1)
    return float("nan")


def _query_term_repetition(query: str, chunk: str) -> float:
    """Mean term-frequency of query terms inside the chunk."""
    q_terms = set(_tokenize(query))
    if not q_terms:
        return 0.0
    tf = Counter(_tokenize(chunk))
    return float(np.mean([tf.get(t, 0) for t in q_terms]))


# ── Correlation diagnostic (paper-style) ──────────────────────────────────────
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ with NaN handling; returns NaN when there is no signal."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x, y = x[mask], y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    # Spearman = Pearson on ranks
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def feature_score_correlations(records: List[Dict]) -> Dict[str, float]:
    """
    Spearman ρ between top-1 retrieval score and the four surface features
    measured in Goyal et al. 2026, computed across all (query, top-1 chunk)
    pairs in `records`.

    The "score" we use is `max_retrieval_score` saved by pipeline_p4 — the
    similarity of the BEST chunk for each query. This is the score the
    retriever actually optimises against during top-k selection.

    Interpretation:
        |ρ| close to 0 = retrieval score is independent of the surface feature
                         (the desired behaviour).
        Large positive ρ = retriever rewards documents that exhibit this feature
                           independent of relevance.

    Returns a dict {feature_name -> ρ}.
    """
    scores, lens, jaccs, pos, reps = [], [], [], [], []
    for r in records:
        if not r.get("chunks") or r.get("max_retrieval_score") is None:
            continue
        chunk = r["chunks"][0]
        scores.append(r["max_retrieval_score"])
        lens.append(_length_tokens(chunk))
        jaccs.append(_jaccard(r["question"], chunk))
        pos.append(_answer_position(chunk, r.get("gold_answers", [])))
        reps.append(_query_term_repetition(r["question"], chunk))

    scores = np.asarray(scores)
    return {
        "brevity_rho":    _spearman(scores, -np.asarray(lens)),   # short = bias-favourable
        "literal_rho":    _spearman(scores,  np.asarray(jaccs)),
        "position_rho":   _spearman(scores, -np.asarray(pos)),    # early = bias-favourable
        "repetition_rho": _spearman(scores,  np.asarray(reps)),
        "n":              int(len(scores)),
    }


# ── Aggregate retrieval-side stats ────────────────────────────────────────────
def retrieved_chunk_stats(records: List[Dict]) -> Dict[str, float]:
    """
    Summary statistics over the retrieved chunks. Useful as a sanity check:
    e.g. HyDE pseudo-docs tend to be long and topical, so we expect retrieved
    chunks to be longer/higher-overlap than the baseline's.
    """
    n_chunks, lens, jaccs, scores = 0, [], [], []
    for r in records:
        for c in r.get("chunks", []) or []:
            n_chunks += 1
            lens.append(_length_tokens(c))
            jaccs.append(_jaccard(r["question"], c))
        if r.get("max_retrieval_score") is not None:
            scores.append(r["max_retrieval_score"])
    return {
        "n_chunks":               n_chunks,
        "mean_chunk_len_tokens":  float(np.mean(lens)) if lens else 0.0,
        "mean_query_chunk_jaccard": float(np.mean(jaccs)) if jaccs else 0.0,
        "mean_top1_score":        float(np.mean(scores)) if scores else 0.0,
        "std_top1_score":         float(np.std(scores))  if scores else 0.0,
    }
