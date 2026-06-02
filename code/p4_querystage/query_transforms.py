"""
Query-stage transformations evaluated in P4.

All transforms take a batch of original questions and return (i) the rewritten
text actually fed to the retriever and (ii) the per-question retrieval input
(either a text string or a pre-computed embedding). The pipeline then performs
retrieval + answer generation uniformly across methods.

Methods
    transform_hyde       — Gao et al. 2022. Pseudo-doc REPLACES the query.
    transform_query2doc  — Wang et al. 2023. Pseudo-doc EXPANDS the query.
    transform_rewrite    — Ma et al. 2023. LLM rewrites the query as one
                            search-engine-friendly string.
    transform_hyde_avg   — HyDE with N>1 samples (Eq. 6 of the HyDE paper):
                            average the embeddings of N hypothetical docs,
                            optionally including the original query embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from prompts import (
    build_hyde_prompt,
    build_query2doc_prompt,
    build_rewrite_prompt,
    build_rewrite_multi_prompt,           # M3 NEW: multi-query rewrite prompt
    compose_query2doc,
)
# M3 NEW: also import retrieve_by_text — multi-query rewrite does its own
# retrieval inside the transform (one search per generated query, then merge),
# so this transform cannot defer to the pipeline's retrieve() step the way
# HyDE/Q2D/single-rewrite do.
from retrieval_utils import encode_texts, retrieve_by_text
from vllm_runner import (
    HYDE_SAMPLING,
    QUERY2DOC_SAMPLING,
    REWRITE_SAMPLING,
    REWRITE_MULTI_SAMPLING,
    VLLMRunner,
)


@dataclass
class QueryTransformResult:
    """
    Output of a query-stage transform.

    Exactly one of {retrieval_texts, retrieval_embeddings, precomputed_chunks}
    is populated; the pipeline dispatches retrieval accordingly:
        - retrieval_texts        → retrieve_by_text
        - retrieval_embeddings   → retrieve_by_embedding
        - precomputed_chunks     → skip retrieval (transform did it itself)

    Fields:
        intermediates: per-question artefact for analysis / case studies
                       (the pseudo-doc, the rewritten query, etc.)
        retrieval_texts: text to feed the retriever's encoder (or None)
        retrieval_embeddings: pre-encoded embeddings (or None)
        precomputed_chunks: M3 NEW — chunks already retrieved inside the
                       transform (used by multi-query rewrite, which runs
                       one FAISS search per generated query and merges)
    """
    intermediates: List[str]
    retrieval_texts: Optional[List[str]] = None
    retrieval_embeddings: Optional[np.ndarray] = None
    precomputed_chunks: Optional[List[List[str]]] = None    # M3 NEW


# ── HyDE (Gao et al. 2022) ────────────────────────────────────────────────────
def transform_hyde(
    llm: VLLMRunner,
    questions: List[str],
    temperature: float = HYDE_SAMPLING.temperature,
) -> QueryTransformResult:
    """
    Single-sample HyDE: generate one pseudo-document per question, use its
    embedding as the retrieval query.
    """
    sampling = HYDE_SAMPLING
    if temperature != sampling.temperature:
        sampling = type(sampling)(**{**sampling.__dict__, "temperature": temperature})
    prompts = [build_hyde_prompt(q) for q in questions]
    pseudo_docs = llm.generate(prompts, sampling)
    return QueryTransformResult(intermediates=pseudo_docs, retrieval_texts=pseudo_docs)


def transform_hyde_avg(
    llm: VLLMRunner,
    questions: List[str],
    retriever,
    n_samples: int = 4,
    include_query: bool = True,
    temperature: float = HYDE_SAMPLING.temperature,
) -> QueryTransformResult:
    """
    Multi-sample HyDE — Gao et al. 2022, Eq. 6:

        v_q = (1 / (N+1)) * [ f(q) + sum_k f(d_hat_k) ]      if include_query
        v_q = (1 / N)     * sum_k f(d_hat_k)                 otherwise

    where f is the contriever encoder and d_hat_k are N pseudo-docs sampled
    from the generative model.

    The averaged vector is then L2-renormalised so FAISS inner-product still
    corresponds to cosine similarity.
    """
    sampling = HYDE_SAMPLING
    if temperature != sampling.temperature:
        sampling = type(sampling)(**{**sampling.__dict__, "temperature": temperature})
    prompts = [build_hyde_prompt(q) for q in questions]
    samples = llm.generate_n_samples(prompts, sampling, n_samples=n_samples)
    # samples: list (len = n_questions) of list[str] (len = n_samples)

    flat_docs = [doc for row in samples for doc in row]
    flat_emb = encode_texts(retriever, flat_docs)
    # back to shape (n_questions, n_samples, dim)
    emb = flat_emb.reshape(len(questions), n_samples, -1)
    avg = emb.mean(axis=1)  # (n_questions, dim)

    if include_query:
        q_emb = encode_texts(retriever, questions)
        avg = (avg * n_samples + q_emb) / (n_samples + 1)

    # renormalise so inner-product == cosine
    norms = np.linalg.norm(avg, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    avg = (avg / norms).astype("float32")

    # store the first pseudo-doc as a representative intermediate
    intermediates = [row[0] for row in samples]
    return QueryTransformResult(intermediates=intermediates, retrieval_embeddings=avg)


# ── Query2doc (Wang et al. 2023) ──────────────────────────────────────────────
def transform_query2doc(
    llm: VLLMRunner,
    questions: List[str],
    repeat_n: int = 1,
) -> QueryTransformResult:
    """
    Query2doc: 4-shot generate a pseudo-doc, then build `q (×repeat_n) [SEP] d'`.

    `repeat_n=5` is the sparse-retrieval recommendation of the paper. For dense
    retrieval (our setup) `repeat_n=1` is the paper-faithful default — see
    prompts.compose_query2doc docstring.
    """
    prompts = [build_query2doc_prompt(q) for q in questions]
    pseudo_docs = llm.generate(prompts, QUERY2DOC_SAMPLING)
    expanded = [
        compose_query2doc(q, d, repeat_n=repeat_n)
        for q, d in zip(questions, pseudo_docs)
    ]
    return QueryTransformResult(intermediates=pseudo_docs, retrieval_texts=expanded)


# ── Rewrite-Retrieve-Read (Ma et al. 2023) ────────────────────────────────────
# M3 NEW: transform_rewrite now accepts `n_queries` to support both the M2
# single-query variant (n_queries=1, default) and the multi-query canonical
# variant from Ma et al. 2023 (n_queries>1). This is the central new ablation
# axis for the Rewrite method in M3.
def transform_rewrite(
    llm: VLLMRunner,
    questions: List[str],
    n_queries: int = 1,                # M3 NEW
    retriever=None,                    # M3 NEW: required when n_queries > 1
    k: int = 5,                        # M3 NEW: chunks per generated query
) -> QueryTransformResult:
    """
    LLM-based query rewriter.

    n_queries=1 (M2 variant, default):
        Single-query rewrite. The rewritten string is returned as
        `retrieval_texts` and the pipeline performs one FAISS search.

    n_queries>1 (M3 NEW — Ma et al. 2023 canonical variant):
        The LLM generates `n_queries` complementary queries separated by ';'.
        Each query is fed to the retriever (k chunks each), and the chunks
        are merged + deduplicated to top-k overall. Because retrieval is
        done INSIDE the transform, we return the merged chunks as
        `precomputed_chunks` and the pipeline skips its own retrieval step.
        Requires `retriever=` to be passed.
    """
    if n_queries == 1:
        # ── M2 single-query path — UNCHANGED behaviour ────────────────────────
        prompts = [build_rewrite_prompt(q) for q in questions]
        raw = llm.generate(prompts, REWRITE_SAMPLING)
        rewritten = [_clean_single_rewrite(orig, r)
                     for orig, r in zip(questions, raw)]
        return QueryTransformResult(intermediates=rewritten,
                                    retrieval_texts=rewritten)

    # ── M3 NEW: multi-query path (Ma et al. 2023 canonical) ───────────────────
    if retriever is None:
        raise ValueError(
            "transform_rewrite with n_queries>1 requires `retriever=` "
            "because multi-query rewriting runs FAISS searches internally."
        )

    prompts = [build_rewrite_multi_prompt(q, n_queries=n_queries)
               for q in questions]
    raw = llm.generate(prompts, REWRITE_MULTI_SAMPLING)

    all_chunks, intermediates = [], []
    for orig, r in zip(questions, raw):
        # Parse the ';'-separated queries; defensive fallback to the original
        # question if the LLM produces an unusable output.
        queries = [q.strip() for q in r.split(';') if q.strip()]
        if not queries:
            queries = [orig]
        # Cap to n_queries (the LLM sometimes emits more than requested).
        queries = queries[:n_queries]
        intermediates.append(" ; ".join(queries))

        # Batched FAISS search across the N queries, then merge + dedup.
        chunk_lists, _ = retrieve_by_text(retriever, queries, k=k)
        seen, merged = set(), []
        for cl in chunk_lists:
            for c in cl:
                key = c[:80]            # short fingerprint for dedup
                if key not in seen:
                    seen.add(key)
                    merged.append(c)
        all_chunks.append(merged[:k])

    return QueryTransformResult(
        intermediates=intermediates,
        precomputed_chunks=all_chunks,  # M3 NEW: signals pipeline to skip retrieve()
    )


# M3 NEW: factored out of the old transform_rewrite body so the n_queries=1
# branch above can reuse the exact same parsing the M2 version used.
def _clean_single_rewrite(orig: str, raw: str) -> str:
    """Parse a single rewrite, defensively stripping preambles/quotes.
    Falls back to `orig` if the cleaned string is empty.
    """
    cleaned = raw.strip()
    for prefix in ("Rewritten:", "Rewrite:", "Query:", "Answer:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    cleaned = cleaned.split("\n", 1)[0].strip().strip('"').strip("'")
    return cleaned if cleaned else orig