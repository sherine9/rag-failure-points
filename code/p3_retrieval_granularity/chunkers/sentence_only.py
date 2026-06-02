import nltk
nltk.download('punkt_tab', quiet=True)


def chunk(text: str) -> list[str]:
    """
    Return individual sentences as chunks.

    Each sentence is both the retrieval unit (embedded) and the generation
    unit (returned to the generator). This is the fine-grained extreme of
    the P3 ablation grid: fine retrieval, short context.

    Compare with small_to_big: same sentence embeddings for retrieval, but
    small_to_big returns the full parent paragraph to the generator.
    The delta between the two isolates the contribution of paragraph context.

    Args:
        text: raw document text

    Returns:
        list of sentence strings
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    sentences = []
    for para in paragraphs:
        for sent in nltk.sent_tokenize(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences
