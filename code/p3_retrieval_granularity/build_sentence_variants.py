"""
Build sentence_only and sentence_window indices without re-encoding.

The p3_small_to_big FAISS index already contains sentence embeddings.
The only difference between the three sentence-level variants is what
is stored in chunks.pkl (what is returned to the generator):

  small_to_big    → parent paragraph   (existing)
  sentence_only   → the sentence itself
  sentence_window → sentence ± 1 within its paragraph (3-sentence window)

This script extracts sentences and windows in the same order as the
small_to_big index was built, then copies the faiss.index and writes
new chunks.pkl files. No GPU compute required.

Prerequisite:
  pipeline_p3.py --chunker small_to_big must have been run first.

Usage (from repo root, inside the RCP job):
  python code/p3_retrieval_granularity/build_sentence_variants.py
"""
import os
import sys
import pickle
import shutil
import faiss
from pathlib import Path

_P3       = Path(__file__).resolve().parent
_BASELINE = _P3.parent / "baseline"
_CHUNKERS = _P3 / "chunkers"
sys.path.insert(0, str(_BASELINE))
sys.path.insert(0, str(_CHUNKERS))

import nltk
nltk.download('punkt_tab', quiet=True)

from datasets import load_dataset

_S2B_INDEX  = "/scratch/faiss_index/p3_small_to_big"
_OUT_SENT   = "/scratch/faiss_index/p3_sentence_only"
_OUT_WIN    = "/scratch/faiss_index/p3_sentence_window"
_N_WIKI     = 200_000
_WINDOW     = 1   # sentences before and after anchor → 3-sentence window


def _extract(n_wiki: int):
    """
    Extract sentences and windows in the same order as the s2b index.

    Processes paragraphs exactly as small_to_big.get_pairs does so that
    position i in sentences[] and windows[] aligns with vector i in
    the s2b FAISS index.

    Returns:
        sentences : list[str]  — anchor sentences (retrieval unit)
        windows   : list[str]  — ±1 sentence window within paragraph
    """
    print(f"Loading {n_wiki:,} Wikipedia articles...")
    wiki = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
    wiki = wiki.select(range(n_wiki))

    sentences, windows = [], []

    for i, doc in enumerate(wiki["text"]):
        for para in [p.strip() for p in doc.split('\n') if p.strip()]:
            sents = nltk.sent_tokenize(para)
            for j, sent in enumerate(sents):
                sent = sent.strip()
                if not sent:
                    continue
                sentences.append(sent)
                start = max(0, j - _WINDOW)
                end   = min(len(sents), j + _WINDOW + 1)
                windows.append(' '.join(sents[start:end]))

        if i % 10_000 == 0:
            print(f"  {i:,}/{n_wiki:,} articles → {len(sentences):,} sentences")

    print(f"Extracted {len(sentences):,} sentences / windows")
    return sentences, windows


def build(n_wiki: int = _N_WIKI):
    s2b_index_file = f"{_S2B_INDEX}/faiss.index"
    if not os.path.exists(s2b_index_file):
        raise FileNotFoundError(
            f"small_to_big FAISS index not found at {_S2B_INDEX}.\n"
            "Run pipeline_p3.py --chunker small_to_big first."
        )

    sentences, windows = _extract(n_wiki)

    # Verify alignment with the existing index
    idx = faiss.read_index(s2b_index_file)
    if idx.ntotal != len(sentences):
        raise ValueError(
            f"Mismatch: s2b FAISS index has {idx.ntotal:,} vectors but "
            f"extracted {len(sentences):,} sentences. "
            f"Ensure n_wiki={n_wiki} matches the original s2b build."
        )
    print(f"Alignment verified: {idx.ntotal:,} vectors == {len(sentences):,} sentences")

    for out_dir, chunks, label in [
        (_OUT_SENT, sentences, "sentence_only"),
        (_OUT_WIN,  windows,  "sentence_window"),
    ]:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(s2b_index_file, f"{out_dir}/faiss.index")
        with open(f"{out_dir}/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Built {label}: {out_dir}  ({len(chunks):,} entries)")


if __name__ == "__main__":
    build()
