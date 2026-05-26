import nltk

nltk.download('punkt_tab', quiet=True)


def get_pairs(text: str) -> list[tuple[str, str]]:
    """
    Return (sentence, parent_paragraph) pairs for a document.

    Each sentence is paired with the full paragraph it came from.
    The pipeline indexes by sentence embedding but returns the paragraph
    to the generator, decoupling retrieval precision from generation context.

    Args:
        text: raw document text

    Returns:
        list of (sentence, paragraph) tuples
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    pairs = []
    for para in paragraphs:
        for sent in nltk.sent_tokenize(para):
            sent = sent.strip()
            if sent:
                pairs.append((sent, para))
    return pairs
