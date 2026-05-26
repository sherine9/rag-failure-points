import os
import json
from datasets import load_dataset
from chunking import chunk_text
from retriever import Retriever
from generator import Generator
from evaluate import evaluate, print_metrics


def run_pipeline(
    n_wiki=200000,        # number of Wikipedia articles to index
    n_questions=100,     # number of NQ questions to evaluate
    chunk_size=256,      # for fixed-size chunking
    overlap=32,          # overlap between chunks
    k=5,                 # number of chunks to retrieve
    index_path="/scratch/faiss_index/baseline",  # save index here
    results_path="/scratch/results/baseline.json"
):
    """
    Full RAG pipeline:
        1. Load Wikipedia subset
        2. Chunk documents (fixed-size)
        3. Build FAISS index (or load if already built)
        4. For each NQ Open question: retrieve + generate
        5. Evaluate and save results

    Args:
        n_wiki: number of Wikipedia articles to index
        n_questions: number of NQ Open questions to evaluate
        chunk_size: number of words per chunk
        overlap: number of overlapping words between chunks
        k: number of chunks to retrieve per question
        index_path: where to save/load the FAISS index
        results_path: where to save results as JSON
    
    Returns:
        results: list of dicts with question, prediction, gold_answers, chunks
        metrics: dict with exact_match, rouge_l, n_samples
    """

    # Step 1: Initialize retriever 
    retriever = Retriever()

    # Step 2: Build or load FAISS index 
    if os.path.exists(f"{index_path}/faiss.index"):
        # load pre-built index to avoid rebuilding every run
        print("Loading existing FAISS index...")
        retriever.load_index(index_path)
    else:
        # build index from scratch
        print("Loading Wikipedia...")
        wiki = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
        wiki = wiki.select(range(n_wiki))
        documents = wiki["text"]

        print("Chunking documents...")
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
            if i % 5000 == 0:
                print(f"  Chunked {i}/{n_wiki} articles, {len(all_chunks)} chunks so far")
        print(f"Total chunks: {len(all_chunks)}")

        print("Building FAISS index...")
        retriever.build_index(all_chunks)

        # save index so we don't rebuild next time
        os.makedirs(index_path, exist_ok=True)
        retriever.save_index(index_path)

    # Step 3: Load NQ Open questions 
    print("Loading NQ Open...")
    ds = load_dataset("nq_open", split="validation")
    ds = ds.select(range(n_questions))

    # Step 4: Initialize generator 
    generator = Generator()

    # Step 5: Run pipeline 
    print(f"\nRunning pipeline on {n_questions} questions...")
    results = []
    for i, example in enumerate(ds):
        question = example["question"]
        gold_answers = example["answer"]

        # retrieve top-k chunks
        chunks = retriever.retrieve(question, k=k)

        # generate answer
        prediction = generator.generate(question, chunks)

        results.append({
            "question": question,
            "prediction": prediction,
            "gold_answers": gold_answers,
            "chunks": chunks
        })

        if i % 10 == 0:
            print(f"  Processed {i}/{n_questions} questions")

    # Step 6: Evaluate 
    metrics = evaluate(results)
    print_metrics(metrics, label="Baseline Results")

    # Step 7: Save results 
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"Results saved to {results_path}")

    return results, metrics


if __name__ == "__main__":
    results, metrics = run_pipeline(
        n_wiki=200000,
        n_questions=100,
        chunk_size=256,
        overlap=32,
        k=5
    )