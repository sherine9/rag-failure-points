import sys
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

nltk.download('punkt_tab', quiet=True)

# lazy-loaded module-level singleton, avoids reloading when chunking many docs
_model = None
_MAX_WORDS = 512


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("facebook/contriever-msmarco")
    return _model


def chunk(text: str, model=None, threshold_percentile: int = 25) -> list[str]:
    """
    Split text into semantically coherent chunks.

    Embeds every sentence with contriever-msmarco, then greedily merges
    adjacent sentences until the cosine similarity between the last sentence
    of the current group and the next sentence falls below the
    threshold_percentile of all pairwise similarities in the document.
    A hard 512-word cap forces a split regardless of similarity.

    Args:
        text: raw document text
        model: optional pre-loaded SentenceTransformer (avoids reloading)
        threshold_percentile: similarity percentile below which a split occurs

    Returns:
        list of text chunks
    """
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    m = model if model is not None else _get_model()
    embeddings = m.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # dot product of L2-normalised vectors = cosine similarity
    sims = [float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)]
    threshold = float(np.percentile(sims, threshold_percentile))

    chunks = []
    current_sents = [sentences[0]]
    current_words = len(sentences[0].split())

    for i, sim in enumerate(sims):
        next_sent = sentences[i + 1]
        next_words = len(next_sent.split())
        force_split = (current_words + next_words) > _MAX_WORDS

        if sim < threshold or force_split:
            chunks.append(' '.join(current_sents))
            current_sents = [next_sent]
            current_words = next_words
        else:
            current_sents.append(next_sent)
            current_words += next_words

    if current_sents:
        chunks.append(' '.join(current_sents))

    return [c for c in chunks if c.strip()]
