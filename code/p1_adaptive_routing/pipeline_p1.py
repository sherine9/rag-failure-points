"""
P1 — Adaptive / Selective Retrieval Pipeline

Compares three controlled conditions on the shared baseline:
    no_retrieval      — generator answers without retrieval
    always_retrieve   — standard baseline RAG (always top-k)
    selective         — router decides whether to retrieve

All other components match code/baseline/: same index, retriever, generator model,
NQ Open slice, k, and evaluate.py metrics.

Usage:
    python pipeline_p1.py --mode always_retrieve --n_questions 50
    python pipeline_p1.py --mode all --n_questions 50
    python pipeline_p1.py --mode selective --score_threshold 0.35
"""

import os
import sys
import json
import argparse
from pathlib import Path

_P1 = Path(__file__).resolve().parent
_BASELINE = _P1.parent / "baseline"
sys.path.insert(0, str(_BASELINE))
sys.path.insert(0, str(_P1))

from datasets import load_dataset  # noqa: E402
from retriever import Retriever  # noqa: E402
from evaluate import evaluate, print_metrics  # noqa: E402

from generator_p1 import GeneratorP1  # noqa: E402
from router import RetrievalRouter  # noqa: E402
from failure_labels import (  # noqa: E402
    enrich_result,
    apply_cross_mode_labels,
    failure_label_counts,
)

MODES = ("no_retrieval", "always_retrieve", "selective")
DEFAULT_INDEX = "/scratch/faiss_index/baseline"
DEFAULT_RESULTS_DIR = "/scratch/results"


def load_queries(n_questions, question_offset=0):
    """Load a fixed slice of NQ Open validation questions."""
    ds = load_dataset("nq_open", split="validation")
    end = question_offset + n_questions
    return ds.select(range(question_offset, end))


def load_retriever(index_path=DEFAULT_INDEX):
    retriever = Retriever()
    if not os.path.exists(f"{index_path}/faiss.index"):
        raise FileNotFoundError(
            f"No FAISS index at {index_path}. "
            "Run code/baseline/pipeline.py once on the cluster to build it."
        )
    retriever.load_index(index_path)
    return retriever


def run_one_example(
    question,
    gold_answers,
    mode,
    retriever,
    generator,
    router,
    k=5,
):
    """Run a single query under the requested mode."""
    record = {
        "question": question,
        "gold_answers": gold_answers,
        "mode": mode,
        "retrieval_decision": False,
        "router_response": None,
        "max_retrieval_score": None,
        "score_gate_applied": False,
        "chunks": [],
        "prediction": "",
    }

    if mode == "no_retrieval":
        record["prediction"] = generator.generate_closed_book(question)

    elif mode == "always_retrieve":
        record["retrieval_decision"] = True
        record["chunks"] = retriever.retrieve(question, k=k)
        record["prediction"] = generator.generate(question, record["chunks"])

    elif mode == "selective":
        decision = router.decide(question, retriever=retriever)
        record["retrieval_decision"] = decision["should_retrieve"]
        record["router_response"] = decision["router_response"]
        record["max_retrieval_score"] = decision["max_score"]
        record["score_gate_applied"] = decision["score_gate_applied"]

        if record["retrieval_decision"]:
            record["chunks"] = retriever.retrieve(question, k=k)
            record["prediction"] = generator.generate(question, record["chunks"])
        else:
            record["prediction"] = generator.generate_closed_book(question)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return record


def run_mode(
    mode,
    retriever,
    generator,
    router,
    questions,
    k=5,
):
    """Run pipeline for one mode over all questions."""
    results = []
    for i, example in enumerate(questions):
        record = run_one_example(
            example["question"],
            example["answer"],
            mode,
            retriever,
            generator,
            router,
            k=k,
        )
        enrich_result(record)
        results.append(record)

        if i % 10 == 0:
            print(f"  [{mode}] {i}/{len(questions)}")

    return results


def save_run(mode, results, metrics, n_questions, question_offset, k, results_dir):
    path = os.path.join(results_dir, f"p1_{mode}.json")
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        "mode": mode,
        "n_questions": n_questions,
        "question_offset": question_offset,
        "k": k,
        "metrics": metrics,
        "failure_label_counts": failure_label_counts(results),
        "pct_retrieved": round(
            sum(1 for r in results if r["retrieval_decision"]) / len(results), 4
        )
        if results
        else 0.0,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {path}")
    return path


def run_pipeline(
    mode,
    n_questions=50,
    question_offset=0,
    k=5,
    index_path=DEFAULT_INDEX,
    results_dir=DEFAULT_RESULTS_DIR,
    score_threshold=None,
):
    """
    Run one or all P1 modes.

    Returns:
        If mode == 'all': dict mode -> (results, metrics)
        Else: (results, metrics)
    """
    questions = load_queries(n_questions, question_offset)
    retriever = load_retriever(index_path)
    generator = GeneratorP1()
    router = RetrievalRouter(generator, score_threshold=score_threshold)

    modes_to_run = MODES if mode == "all" else (mode,)

    results_by_mode = {}
    for m in modes_to_run:
        print(f"\nRunning P1 mode: {m} ({n_questions} questions)...")
        results_by_mode[m] = run_mode(
            m, retriever, generator, router, questions, k=k
        )

    if mode == "all" or len(results_by_mode) > 1:
        apply_cross_mode_labels(results_by_mode)
        for m in results_by_mode:
            metrics = evaluate(results_by_mode[m])
            save_run(
                m,
                results_by_mode[m],
                metrics,
                n_questions,
                question_offset,
                k,
                results_dir,
            )
            print_metrics(metrics, label=f"P1 — {m}")
            print(f"  Failure labels: {failure_label_counts(results_by_mode[m])}")
            print(f"  % retrieved: {sum(1 for r in results_by_mode[m] if r['retrieval_decision']) / len(results_by_mode[m]):.1%}")

        ablation_path = os.path.join(results_dir, "p1_ablation.json")
        ablation = {
            "n_questions": n_questions,
            "question_offset": question_offset,
            "k": k,
            "modes": {
                m: {
                    "metrics": evaluate(results_by_mode[m]),
                    "failure_label_counts": failure_label_counts(results_by_mode[m]),
                    "pct_retrieved": round(
                        sum(1 for r in results_by_mode[m] if r["retrieval_decision"])
                        / len(results_by_mode[m]),
                        4,
                    ),
                }
                for m in results_by_mode
            },
        }
        with open(ablation_path, "w") as f:
            json.dump(ablation, f, indent=2)
        print(f"\nAblation summary saved to {ablation_path}")

        return results_by_mode

    m = modes_to_run[0]
    results = results_by_mode[m]
    metrics = evaluate(results)
    save_run(m, results, metrics, n_questions, question_offset, k, results_dir)
    print_metrics(metrics, label=f"P1 — {m}")
    print(f"Failure labels: {failure_label_counts(results)}")
    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 adaptive retrieval pipeline")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[*MODES, "all"],
        help="Retrieval condition to run (or 'all' for full ablation)",
    )
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--question_offset", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index_path", default=DEFAULT_INDEX)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Optional FAISS score gate for selective mode",
    )
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        n_questions=args.n_questions,
        question_offset=args.question_offset,
        k=args.k,
        index_path=args.index_path,
        results_dir=args.results_dir,
        score_threshold=args.score_threshold,
    )
