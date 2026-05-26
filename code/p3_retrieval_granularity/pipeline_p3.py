"""
P3 Retrieval Granularity Pipeline

Swaps the chunking strategy while keeping all other pipeline components
(retriever model, generator, evaluation) identical to the shared baseline.

Supported chunkers: fixed | semantic | small_to_big | proposition

Usage:
    python pipeline_p3.py --chunker semantic
    python pipeline_p3.py --chunker proposition --n_wiki 1000
"""
import os
import sys
import json
import argparse
import faiss
from pathlib import Path

_P3       = Path(__file__).resolve().parent
_BASELINE = _P3.parent / 'baseline'
_CHUNKERS = _P3 / 'chunkers'

sys.path.insert(0, str(_BASELINE))
sys.path.insert(0, str(str(_CHUNKERS)))

from datasets import load_dataset
from retriever import Retriever
from generator import Generator
from evaluate import evaluate, print_metrics
import fixed
import semantic
import small_to_big
import proposition

_CHUNKER_MODS = {
    'fixed':        fixed,
    'semantic':     semantic,
    'small_to_big': small_to_big,
    'proposition':  proposition,
}


def run_pipeline(
    chunker_name: str,
    n_wiki: int = 200_000,
    n_questions: int = 100,
    k: int = 5,
    index_path: str = None,
    results_path: str = None,
) -> tuple:
    """
    Run the RAG pipeline with a swappable chunking strategy.

    All components except chunking are identical to the shared baseline:
    contriever-msmarco retrieval, FAISS index, Mistral-7B-Instruct generation,
    and evaluation on NQ Open validation split.

    Args:
        chunker_name:  one of fixed | semantic | small_to_big | proposition
        n_wiki:        number of Wikipedia articles to index
        n_questions:   number of NQ Open questions to evaluate
        k:             number of chunks to retrieve per question
        index_path:    where to save/load the FAISS index
        results_path:  where to save results as JSON

    Returns:
        (results, metrics)
    """
    if index_path is None:
        index_path = f"/scratch/faiss_index/p3_{chunker_name}"
    if results_path is None:
        results_path = f"/scratch/results/p3_{chunker_name}.json"

    retriever = Retriever()
    generator = None

    # Build or load FAISS index
    _chunks_exist = (
        os.path.exists(f"{index_path}/chunks.pkl") or
        os.path.exists(f"{index_path}/chunks.npy")
    )
    if os.path.exists(f"{index_path}/faiss.index") and _chunks_exist:
        print(f"Loading existing index for '{chunker_name}'...")
        retriever.load_index(index_path)
    else:
        print(f"Loading {n_wiki} Wikipedia articles...")
        wiki = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
        wiki = wiki.select(range(n_wiki))
        documents = wiki["text"]

        if chunker_name == 'proposition':
            # load generator now: shared between chunking and generation
            generator = Generator()
            _build_proposition_index(retriever, documents, n_wiki, generator)
        elif chunker_name == 'small_to_big':
            _build_s2b_index(retriever, documents, n_wiki)
        else:
            _build_standard_index(retriever, documents, n_wiki, chunker_name)

        os.makedirs(index_path, exist_ok=True)
        retriever.save_index(index_path)
        print(f"Index saved to {index_path}")

    if generator is None:
        generator = Generator()

    # Evaluate on NQ Open
    print(f"\nLoading {n_questions} NQ Open questions...")
    ds = load_dataset("nq_open", split="validation")
    ds = ds.select(range(n_questions))

    print(f"Running pipeline ({chunker_name}) on {n_questions} questions...")
    results = []
    for i, example in enumerate(ds):
        question     = example["question"]
        gold_answers = example["answer"]
        chunks       = retriever.retrieve(question, k=k)
        prediction   = generator.generate(question, chunks)
        results.append({
            "question":     question,
            "prediction":   prediction,
            "gold_answers": gold_answers,
            "chunks":       chunks,
        })
        if i % 10 == 0:
            print(f"  {i}/{n_questions}")

    metrics = evaluate(results)
    print_metrics(metrics, label=f"P3 — {chunker_name}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {"chunker": chunker_name, "n_wiki": n_wiki, "metrics": metrics, "results": results},
            f, indent=2,
        )
    print(f"Results saved to {results_path}")
    return results, metrics


# Index building helpers 

def _build_standard_index(retriever, documents, n_wiki, chunker_name):
    mod = _CHUNKER_MODS[chunker_name]
    all_chunks = []
    for i, doc in enumerate(documents):
        all_chunks.extend(mod.chunk(doc))
        if i % 5000 == 0:
            print(f"  Chunked {i}/{n_wiki} articles → {len(all_chunks)} chunks")
    print(f"Total chunks ({chunker_name}): {len(all_chunks)}")
    retriever.build_index(all_chunks)


def _build_s2b_index(retriever, documents, n_wiki):
    """
    Small-to-big: embed sentences, but store parent paragraphs.

    FAISS index is built from sentence embeddings for precise retrieval.
    retriever.chunks holds paragraphs, so retrieve() returns paragraph
    context to the generator, decoupling retrieval unit from generation unit.
    """
    sentences, paragraphs = [], []
    for i, doc in enumerate(documents):
        for sent, para in small_to_big.get_pairs(doc):
            sentences.append(sent)
            paragraphs.append(para)
        if i % 5000 == 0:
            print(f"  Processed {i}/{n_wiki} articles → {len(sentences)} pairs")
    print(f"Total sentence-paragraph pairs: {len(sentences)}")

    print(f"Encoding {len(sentences)} sentences...")
    embeddings = retriever.model.encode(
        sentences,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=256,
        normalize_embeddings=True,
    )
    retriever.chunks = paragraphs
    retriever.index  = faiss.IndexFlatIP(embeddings.shape[1])
    retriever.index.add(embeddings)
    print(f"Small-to-big index: {retriever.index.ntotal} sentence vectors → paragraphs returned on retrieval")


def _build_proposition_index(retriever, documents, n_wiki, generator):
    all_props = []
    for i, doc in enumerate(documents):
        all_props.extend(
            proposition.chunk(doc, tokenizer=generator.tokenizer, model=generator.model)
        )
        if i % 100 == 0:
            print(f"  Processed {i}/{n_wiki} articles → {len(all_props)} propositions")
    print(f"Total propositions: {len(all_props)}")
    retriever.build_index(all_props)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3 retrieval granularity pipeline")
    parser.add_argument("--chunker",     required=True, choices=list(_CHUNKER_MODS))
    parser.add_argument("--n_wiki",      type=int, default=200_000)
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--k",           type=int, default=5)
    args = parser.parse_args()

    run_pipeline(
        chunker_name=args.chunker,
        n_wiki=args.n_wiki,
        n_questions=args.n_questions,
        k=args.k,
    )
