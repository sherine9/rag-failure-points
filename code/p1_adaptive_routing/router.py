"""
Selective retrieval router for P1.

Uses the shared Mistral generator for a per-query retrieve / skip decision.
Optional FAISS score threshold can override low-confidence retrievals.
"""

from generator_p1 import GeneratorP1


def max_retrieval_score(retriever, question):
    """Return top-1 cosine similarity score for a query (index must exist)."""
    if retriever.index is None:
        raise RuntimeError("Retriever index not loaded.")

    query_emb = retriever.model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, _ = retriever.index.search(query_emb, 1)
    return float(scores[0][0])


class RetrievalRouter:
    """
    Decides whether to retrieve before generation.

    Strategy (Milestone 2):
        1. LLM router prompt (GeneratorP1.route_retrieval)
        2. Optional score gate: skip retrieval if top-1 score < score_threshold
    """

    def __init__(self, generator: GeneratorP1, score_threshold=None):
        """
        Args:
            generator: loaded GeneratorP1 instance
            score_threshold: if set, skip retrieval when max FAISS score is below this
        """
        self.generator = generator
        self.score_threshold = score_threshold

    def decide(self, question, retriever=None):
        """
        Args:
            question: NQ Open question string
            retriever: baseline Retriever (required when score_threshold is set)

        Returns:
            dict with keys:
                should_retrieve (bool)
                router_response (str)
                max_score (float | None)
                score_gate_applied (bool)
        """
        should_retrieve, router_response = self.generator.route_retrieval(question)
        max_score = None
        score_gate_applied = False

        if self.score_threshold is not None:
            if retriever is None:
                raise ValueError("retriever is required when score_threshold is set")
            max_score = max_retrieval_score(retriever, question)
            if should_retrieve and max_score < self.score_threshold:
                should_retrieve = False
                score_gate_applied = True

        return {
            "should_retrieve": should_retrieve,
            "router_response": router_response,
            "max_score": max_score,
            "score_gate_applied": score_gate_applied,
        }
