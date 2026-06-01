import nltk
nltk.download('punkt_tab', quiet=True)

_WINDOW = 1  # sentences before and after the anchor sentence


def get_window_pairs(text: str, window: int = _WINDOW):
    """
    Yield (sentence, window_context) pairs for a document.

    The sentence is the retrieval unit (embedded in the FAISS index).
    The window_context is the generation unit: the sentence plus up to
    `window` sentences before and after it, constrained to the same
    paragraph to preserve coherence.

    Args:
        text:   raw document text
        window: number of surrounding sentences on each side (default 1 → 3-sentence window)

    Yields:
        (sentence, window_context) tuples
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    for para in paragraphs:
        sents = nltk.sent_tokenize(para)
        for i, sent in enumerate(sents):
            sent = sent.strip()
            if not sent:
                continue
            start = max(0, i - window)
            end   = min(len(sents), i + window + 1)
            yield sent, ' '.join(sents[start:end])


def chunk(text: str, window: int = _WINDOW) -> list[str]:
    """
    Return sentence-window strings as chunks.

    NOTE: when used in the standard pipeline build path, this embeds the
    window text (not the sentence). The correct build path uses
    build_sentence_variants.py or _build_sentence_window_index in
    pipeline_p3.py, which embeds the anchor sentence but stores the window.

    Args:
        text:   raw document text
        window: number of surrounding sentences on each side

    Returns:
        list of window-context strings
    """
    return [ctx for _, ctx in get_window_pairs(text, window=window)]
