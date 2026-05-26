"""
Batched retrieval helpers built on top of code/baseline/retriever.py.

The shared `Retriever.retrieve` method encodes one query at a time. For P4 we
need to score hundreds of queries (transformed and not), so we expose two
batched helpers that reuse the SAME embedding model and FAISS index loaded by
the shared `Retriever`:

    retrieve_by_text(retriever, texts, k)
        Encode `texts` in a single SentenceTransformer batch, search FAISS.
        Used by: baseline (texts = original questions), HyDE (texts = pseudo
        documents), Query2doc (texts = q ⊕ pseudo-doc), Rewrite (texts =
        rewritten queries).

    retrieve_by_embedding(retriever, embeddings, k)
        Skip encoding and search FAISS directly. Used by the HyDE
        "n samples averaged" ablation, where we mean-pool N hypothetical
        document embeddings before search (Gao et al. 2022, Eq. 6).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def retrieve_by_text(retriever, texts: List[str], k: int = 5, batch_size: int = 64
                     ) -> Tuple[List[List[str]], np.ndarray]:
    """
    Encode `texts` with contriever, search the FAISS index.

    Args:
        retriever: a code.baseline.retriever.Retriever with a built/loaded index
        texts:     list of query texts (may be raw questions, pseudo-docs, etc.)
        k:         number of chunks to return per query
        batch_size: encoder batch size

    Returns:
        chunks_per_query: list of lists, shape (len(texts), k)
        scores:           ndarray of shape (len(texts), k), FAISS inner-product scores
    """
    if retriever.index is None:
        raise RuntimeError("Retriever index not loaded. Call load_index() first.")

    embeddings = retriever.model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    scores, indices = retriever.index.search(embeddings, k)
    chunks_per_query = [
        [retriever.chunks[i] for i in row if 0 <= i < len(retriever.chunks)]
        for row in indices
    ]
    return chunks_per_query, scores


def retrieve_by_embedding(retriever, embeddings: np.ndarray, k: int = 5
                          ) -> Tuple[List[List[str]], np.ndarray]:
    """
    Search FAISS with pre-computed L2-normalised embeddings.

    Used by the HyDE multi-sample ablation: average N pseudo-document
    embeddings (already L2-normalised) and search once.
    """
    if retriever.index is None:
        raise RuntimeError("Retriever index not loaded. Call load_index() first.")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    scores, indices = retriever.index.search(embeddings, k)
    chunks_per_query = [
        [retriever.chunks[i] for i in row if 0 <= i < len(retriever.chunks)]
        for row in indices
    ]
    return chunks_per_query, scores


def encode_texts(retriever, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """L2-normalised contriever embeddings for a batch of texts."""
    return retriever.model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
