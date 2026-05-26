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
    compose_query2doc,
)
from retrieval_utils import encode_texts
from vllm_runner import (
    HYDE_SAMPLING,
    QUERY2DOC_SAMPLING,
    REWRITE_SAMPLING,
    VLLMRunner,
)


@dataclass
class QueryTransformResult:
    """
    Output of a query-stage transform.

    Exactly one of {retrieval_texts, retrieval_embeddings} is populated;
    the pipeline calls retrieve_by_text or retrieve_by_embedding accordingly.

    Fields:
        intermediates: per-question artefact for analysis / case studies
                       (the pseudo-doc, the rewritten query, etc.)
        retrieval_texts: text to feed the retriever's encoder (or None)
        retrieval_embeddings: pre-encoded embeddings (or None)
    """
    intermediates: List[str]
    retrieval_texts: Optional[List[str]] = None
    retrieval_embeddings: Optional[np.ndarray] = None


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
def transform_rewrite(
    llm: VLLMRunner,
    questions: List[str],
) -> QueryTransformResult:
    """
    LLM-based query rewriter. Ma et al. propose a multi-query, step-by-step
    output; we simplify to a single rewrite because (i) NQ Open is single-hop
    and (ii) downstream we encode a single string. The rewritten string is
    parsed defensively in case the LLM adds preamble.
    """
    prompts = [build_rewrite_prompt(q) for q in questions]
    raw = llm.generate(prompts, REWRITE_SAMPLING)

    rewritten = []
    for orig, r in zip(questions, raw):
        cleaned = r.strip()
        # drop common preambles like "Rewritten:" if the model echoes the tag
        for prefix in ("Rewritten:", "Rewrite:", "Query:", "Answer:"):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        # one line only
        cleaned = cleaned.split("\n", 1)[0].strip().strip('"').strip("'")
        rewritten.append(cleaned if cleaned else orig)
    return QueryTransformResult(intermediates=rewritten, retrieval_texts=rewritten)
