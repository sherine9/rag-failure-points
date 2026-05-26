"""
P4 — Query-Stage Rewriting Pipeline

Compares four conditions on the shared baseline:
    baseline   — vanilla RAG (no query transformation)
    hyde       — Hypothetical Document Embeddings (Gao et al., 2022)
    query2doc  — Query expansion via LLM pseudo-document (Wang et al., 2023)
    rewrite    — LLM-based query rewriter (Ma et al., 2023, single-query variant)

Everything else is held identical to code/baseline:
    Retriever    : facebook/contriever-msmarco + FAISS (index loaded from disk)
    Generator    : mistralai/Mistral-7B-Instruct-v0.2 via vLLM
    Reader prompt: same shape as code/baseline/generator.py
    Eval         : EM / ROUGE-L / Token-F1 / Retrieval-Hit-Rate from evaluate.py
    Dataset      : nq_open validation slice (same offsets as baseline)

Usage (cluster):
    # Requires the baseline index at /scratch/faiss_index/baseline
    python pipeline_p4.py --method all      --n_questions 100
    python pipeline_p4.py --method hyde     --n_questions 100 --hyde_n_samples 4
    python pipeline_p4.py --method query2doc --n_questions 100 --q2d_repeat 5

The script writes one JSON per method under /scratch/results/p4/ plus a
combined `p4_ablation.json` summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# ── sys.path: shared baseline + this directory ────────────────────────────────
_P4       = Path(__file__).resolve().parent
_BASELINE = _P4.parent / "baseline"
sys.path.insert(0, str(_BASELINE))
sys.path.insert(0, str(_P4))

# IMPORTANT: vLLM must be initialised before any other CUDA call, so we import
# the runner first and instantiate it before the Retriever (which loads a
# SentenceTransformer to GPU).
from vllm_runner import READER_SAMPLING, VLLMRunner       # noqa: E402
from prompts import build_reader_prompt                   # noqa: E402
from query_transforms import (                            # noqa: E402
    QueryTransformResult,
    transform_hyde,
    transform_hyde_avg,
    transform_query2doc,
    transform_rewrite,
)
from retrieval_utils import retrieve_by_embedding, retrieve_by_text  # noqa: E402


METHODS = ("baseline", "hyde", "query2doc", "rewrite")
DEFAULT_INDEX = "/scratch/faiss_index/baseline"
DEFAULT_RESULTS_DIR = "/scratch/results/p4"


# ── Loading ───────────────────────────────────────────────────────────────────
def load_retriever(index_path: str = DEFAULT_INDEX):
    """Load the shared baseline FAISS index (no rebuild)."""
    from retriever import Retriever  # noqa: PLC0415 — must come AFTER vLLM init

    if not os.path.exists(f"{index_path}/faiss.index"):
        raise FileNotFoundError(
            f"No FAISS index at {index_path}. Run code/baseline/pipeline.py "
            "on the cluster first."
        )
    retriever = Retriever()
    retriever.load_index(index_path)
    return retriever


def load_questions(n_questions: int, offset: int = 0):
    """NQ Open validation slice. Matches the shared baseline slicing."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("nq_open", split="validation")
    return ds.select(range(offset, offset + n_questions))


# ── Query-stage step ──────────────────────────────────────────────────────────
def transform_queries(
    method: str,
    llm: VLLMRunner,
    retriever,
    questions: List[str],
    *,
    hyde_n_samples: int = 1,
    hyde_include_query: bool = True,
    hyde_temperature: float = 0.7,
    q2d_repeat: int = 1,
) -> QueryTransformResult:
    """Dispatch to the right transform; returns a QueryTransformResult."""
    if method == "baseline":
        return QueryTransformResult(intermediates=list(questions),
                                    retrieval_texts=list(questions))
    if method == "hyde":
        if hyde_n_samples <= 1:
            return transform_hyde(llm, questions, temperature=hyde_temperature)
        return transform_hyde_avg(
            llm, questions, retriever,
            n_samples=hyde_n_samples,
            include_query=hyde_include_query,
            temperature=hyde_temperature,
        )
    if method == "query2doc":
        return transform_query2doc(llm, questions, repeat_n=q2d_repeat)
    if method == "rewrite":
        return transform_rewrite(llm, questions)
    raise ValueError(f"Unknown method: {method}")


# ── Retrieval step ────────────────────────────────────────────────────────────
def retrieve(retriever, transform: QueryTransformResult, k: int):
    """Calls the right retrieval helper for the transform output."""
    if transform.retrieval_embeddings is not None:
        chunks_per_q, scores = retrieve_by_embedding(
            retriever, transform.retrieval_embeddings, k=k
        )
    else:
        chunks_per_q, scores = retrieve_by_text(
            retriever, transform.retrieval_texts, k=k
        )
    return chunks_per_q, scores


# ── Reader step ───────────────────────────────────────────────────────────────
def generate_answers(llm: VLLMRunner, questions: List[str],
                     chunks_per_q: List[List[str]]) -> List[str]:
    """Batched answer generation with the shared reader prompt."""
    prompts = [build_reader_prompt(q, ctx) for q, ctx in zip(questions, chunks_per_q)]
    return llm.generate(prompts, READER_SAMPLING)


def _extract_short_answer(text: str) -> str:
    """
    Mirror of code/baseline/generator.py:_extract_short_answer so the reader
    output format matches the baseline and the evaluate.py expectations.
    """
    text = text.strip()
    if "i don't know" in text.lower() or "not mentioned" in text.lower():
        return text
    for sep in [". ", ".\n", ", which", ", but", " (", " - "]:
        if sep in text:
            text = text.split(sep)[0]
    prefixes = [
        "the answer is ", "it was ", "it is ", "they were ",
        "he was ", "she was ", "this was ", "that was ",
    ]
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


# ── End-to-end method runner ──────────────────────────────────────────────────
def run_method(
    method: str,
    llm: VLLMRunner,
    retriever,
    questions_ds,
    *,
    k: int = 5,
    hyde_n_samples: int = 1,
    hyde_include_query: bool = True,
    hyde_temperature: float = 0.7,
    q2d_repeat: int = 1,
) -> List[Dict]:
    """Run one P4 method end-to-end over `questions_ds`. Returns per-query records."""
    questions = [ex["question"] for ex in questions_ds]
    gold      = [ex["answer"]   for ex in questions_ds]

    print(f"\n[{method}] Query-stage transform on {len(questions)} questions...")
    transform = transform_queries(
        method, llm, retriever, questions,
        hyde_n_samples=hyde_n_samples,
        hyde_include_query=hyde_include_query,
        hyde_temperature=hyde_temperature,
        q2d_repeat=q2d_repeat,
    )

    print(f"[{method}] FAISS search...")
    chunks_per_q, scores = retrieve(retriever, transform, k=k)

    print(f"[{method}] Reader generation...")
    raw_answers = generate_answers(llm, questions, chunks_per_q)
    predictions = [_extract_short_answer(a) for a in raw_answers]

    records: List[Dict] = []
    for q, g, p, c, s, inter in zip(
        questions, gold, predictions, chunks_per_q, scores, transform.intermediates,
    ):
        records.append({
            "question": q,
            "prediction": p,
            "gold_answers": g,
            "chunks": c,
            "method": method,
            "intermediate": inter,           # pseudo-doc / rewrite / None
            "max_retrieval_score": float(max(s)) if len(s) else None,
        })
    return records


# ── Persistence ───────────────────────────────────────────────────────────────
def save_run(method: str, records: List[Dict], metrics: Dict,
             results_dir: str, **meta) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"p4_{method}.json")
    payload = {"method": method, "metrics": metrics, "results": records, **meta}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved → {path}")
    return path


# ── Entry point ───────────────────────────────────────────────────────────────
def run_pipeline(
    method: str,
    n_questions: int = 100,
    question_offset: int = 0,
    k: int = 5,
    index_path: str = DEFAULT_INDEX,
    results_dir: str = DEFAULT_RESULTS_DIR,
    hyde_n_samples: int = 1,
    hyde_include_query: bool = True,
    hyde_temperature: float = 0.7,
    q2d_repeat: int = 1,
    llm: VLLMRunner = None,
    retriever=None,
):
    """
    Run one method (or all) end-to-end.

    If `llm` and `retriever` are provided (e.g. when called from a notebook
    that has already initialised them in the right order), they are reused —
    otherwise this function constructs them here. vLLM MUST be initialised
    before the retriever; we enforce that by importing the runner first.
    """
    from evaluate import evaluate, print_metrics  # noqa: PLC0415

    # IMPORTANT: vLLM first.
    if llm is None:
        llm = VLLMRunner()
    if retriever is None:
        retriever = load_retriever(index_path)

    questions_ds = load_questions(n_questions, question_offset)
    modes = METHODS if method == "all" else (method,)

    all_records, all_metrics = {}, {}
    for m in modes:
        records = run_method(
            m, llm, retriever, questions_ds,
            k=k,
            hyde_n_samples=hyde_n_samples,
            hyde_include_query=hyde_include_query,
            hyde_temperature=hyde_temperature,
            q2d_repeat=q2d_repeat,
        )
        metrics = evaluate(records)
        print_metrics(metrics, label=f"P4 — {m}")
        save_run(
            m, records, metrics, results_dir,
            n_questions=n_questions, question_offset=question_offset, k=k,
            hyde_n_samples=hyde_n_samples,
            hyde_include_query=hyde_include_query,
            hyde_temperature=hyde_temperature,
            q2d_repeat=q2d_repeat,
        )
        all_records[m] = records
        all_metrics[m] = metrics

    if method == "all":
        ablation_path = os.path.join(results_dir, "p4_ablation.json")
        with open(ablation_path, "w") as f:
            json.dump({"metrics": all_metrics}, f, indent=2)
        print(f"\nAblation summary → {ablation_path}")

    return all_records, all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4 query-stage rewriting pipeline")
    parser.add_argument("--method", required=True,
                        choices=[*METHODS, "all"],
                        help="Query-stage method to run (or 'all').")
    parser.add_argument("--n_questions",      type=int, default=100)
    parser.add_argument("--question_offset",  type=int, default=0)
    parser.add_argument("--k",                type=int, default=5)
    parser.add_argument("--index_path",       default=DEFAULT_INDEX)
    parser.add_argument("--results_dir",      default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--hyde_n_samples",   type=int, default=1,
                        help="N hypothetical docs for HyDE averaging (Eq. 6).")
    parser.add_argument("--hyde_no_query",    action="store_true",
                        help="Drop the original query embedding from HyDE averaging.")
    parser.add_argument("--hyde_temperature", type=float, default=0.7)
    parser.add_argument("--q2d_repeat",       type=int, default=1,
                        help="Query repetition for Query2doc (paper uses 5 for sparse).")
    args = parser.parse_args()

    run_pipeline(
        method=args.method,
        n_questions=args.n_questions,
        question_offset=args.question_offset,
        k=args.k,
        index_path=args.index_path,
        results_dir=args.results_dir,
        hyde_n_samples=args.hyde_n_samples,
        hyde_include_query=not args.hyde_no_query,
        hyde_temperature=args.hyde_temperature,
        q2d_repeat=args.q2d_repeat,
    )
