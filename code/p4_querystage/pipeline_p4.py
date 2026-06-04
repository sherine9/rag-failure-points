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
DEFAULT_RESULTS_DIR = "/scratch/results/p4/m3"


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
    rewrite_n_queries: int = 1,    # M3 NEW: number of queries for multi-query rewrite
    k: int = 5,                    # M3 NEW: forwarded to multi-query rewrite for its
                                   #          internal FAISS searches (one per query)
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
        # M3 NEW: pass retriever + k so the multi-query branch
        # (rewrite_n_queries > 1) can run its own FAISS searches internally.
        return transform_rewrite(
            llm, questions,
            n_queries=rewrite_n_queries,
            retriever=retriever,
            k=k,
        )
    raise ValueError(f"Unknown method: {method}")


# ── Retrieval step ────────────────────────────────────────────────────────────
def retrieve(retriever, transform: QueryTransformResult, k: int):
    """Calls the right retrieval helper for the transform output."""
    # M3 NEW: multi-query rewrite does retrieval itself (one search per
    # generated query, then merge+dedup) and returns the merged chunks as
    # `precomputed_chunks`. We just forward them. Scores are filled with
    # zeros because the per-chunk similarity is not naturally defined
    # across multiple query embeddings; downstream `max_retrieval_score`
    # will therefore be 0 for multi-query rewrite — keep that in mind
    # when interpreting the bias_analysis correlations.
    if transform.precomputed_chunks is not None:
        import numpy as _np                    # local import — keeps top clean
        scores = _np.zeros((len(transform.precomputed_chunks), k), dtype="float32")
        return transform.precomputed_chunks, scores
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
    rewrite_n_queries: int = 1,    # M3 NEW: multi-query rewrite knob
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
        rewrite_n_queries=rewrite_n_queries,    # M3 NEW
        k=k,                                    # M3 NEW: forwarded to multi-query rewrite
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
# modified for m3; added a label
def save_run(method: str, label: str, records: List[Dict], metrics: Dict,
             results_dir: str, **meta) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"p4_{label}.json")
    payload = {"method": method, 'label': label, "metrics": metrics, "results": records, **meta}
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


# ──────────────────────────────────────────────────────────────────────────────
# M3 NEW — tuning ablation + held-out multi-seed test
# ──────────────────────────────────────────────────────────────────────────────

# M3 NEW: pre-committed tuning ablation grid (slice indices 0-999).
# Format: (dispatch_method, json_label, hyde_n_samples, hyde_include_query,
#          q2d_repeat, rewrite_n_queries).
# Defaults that don't apply to a given method (e.g. hyde_n_samples for
# query2doc) are ignored by transform_queries. Pre-committing this grid is
# what makes the selection step in BEST_CONFIGS_M3 defensible against
# researcher degrees of freedom.
METHODS_M3 = (
    # HyDE — N-samples × include_query
    ("hyde",      "hyde_n1_wq",       1, True,  1, 1),
    ("hyde",      "hyde_n4_wq",       4, True,  1, 1),
    ("hyde",      "hyde_n8_wq",       8, True,  1, 1),
    ("hyde",      "hyde_n4_noq",      4, False, 1, 1),
    ("hyde",      "hyde_n8_noq",      8, False, 1, 1),
    # Query2doc — query repetition factor
    ("query2doc", "query2doc_rep1",   1, True,  1, 1),
    ("query2doc", "query2doc_rep3",   1, True,  3, 1),
    ("query2doc", "query2doc_rep5",   1, True,  5, 1),
    # Rewrite — number of search queries generated per question
    # (Ma et al. 2023 canonical: n>1; M2 simplification: n=1)
    ("rewrite",   "rewrite_n1",       1, True,  1, 1),
    ("rewrite",   "rewrite_n2",       1, True,  1, 2),
    ("rewrite",   "rewrite_n3",       1, True,  1, 3),
    # test baseline to compare
    # ("hyde",      "hyde_n1_wq_100q",       1, True,  1, 1),
    # ("hyde",      "hyde_n4_wq_100q",       4, True,  1, 1),
    # ("hyde",      "hyde_n8_wq_100q",       8, True,  1, 1),
    # ("hyde",      "hyde_n4_noq_100q",      4, False, 1, 1),
    # ("hyde",      "hyde_n8_noq_100q",      8, False, 1, 1),
)


# M3 NEW: after inspecting results/p4/m3/p4_m3_ablation.json, REPLACE the
# placeholder configs below with the actual ablation winners. Each entry
# follows the same format as METHODS_M3, plus the "best_*" label gets
# expanded with the seed in run_m3_test (e.g. "best_hyde_seed0").
BEST_CONFIGS_M3 = (
    # (method,      label,            n_samples, include_q, q2d_rep, rw_n)
    # ("baseline",    "best_baseline",  1, True,  1, 1),                   # always greedy
    # ("hyde",        "best_hyde",      8, True,  1, 1),                   # ← REPLACE after ablation
    # ("query2doc",   "best_query2doc", 1, True,  1, 1),                   # ← REPLACE after ablation
    # ("rewrite",     "best_rewrite",   1, True,  1, 1),                   # ← REPLACE after ablation
    ("hyde",        "hyde_n8_wq_1000qavgtest",         8, True,  1, 1),
    ("hyde",        "hyde_n4_noq_1000qavgtest",        4, False, 1, 1),
    ("query2doc",   "query2doc_rep1_1000qavgtest",     1, True,  1, 1),
    ("query2doc",   "query2doc_rep3_1000qavgtest",     1, True,  3, 1),
)


def run_m3_ablation(
    n_questions: int = 1000,
    question_offset: int = 0,
    k: int = 5,
    index_path: str = DEFAULT_INDEX,
    results_dir: str = DEFAULT_RESULTS_DIR,
    hyde_temperature: float = 0.7,
    llm: VLLMRunner = None,
    retriever=None,
):
    """
    M3 tuning ablation — sweep the pre-committed METHODS_M3 grid on the
    first `n_questions` validation examples.

    Each config writes its own JSON named after `label`, plus a combined
    summary `p4_m3_ablation.json`. The grid is intentionally fixed in
    METHODS_M3 and not configurable from the CLI: pre-committing controls
    researcher degrees of freedom in the selection step.

    If `llm` and `retriever` are provided (e.g. from a notebook that has
    already initialised them in the right CUDA order), they are reused —
    otherwise this function constructs them here. vLLM MUST be initialised
    before the retriever; we enforce that by importing the runner first.
    """
    from evaluate import evaluate, print_metrics  # noqa: PLC0415

    # IMPORTANT: vLLM first (before the Retriever's SentenceTransformer).
    if llm is None:
        llm = VLLMRunner()
    if retriever is None:
        retriever = load_retriever(index_path)

    questions_ds = load_questions(n_questions, question_offset)
    all_metrics = {}

    # M3 NEW: iterate the pre-committed grid. Each row contains exactly the
    # hyperparameters of one config — no global state mutated across rows.
    for (method, label, n_samples, include_q, q2d_rep, rw_n) in METHODS_M3:
        print(f"\n=== {label} (method={method}, "
              f"n_samples={n_samples}, include_q={include_q}, "
              f"q2d_rep={q2d_rep}, rw_n={rw_n}) ===")
        records = run_method(
            method, llm, retriever, questions_ds,
            k=k,
            hyde_n_samples=n_samples,
            hyde_include_query=include_q,
            hyde_temperature=hyde_temperature,
            q2d_repeat=q2d_rep,
            rewrite_n_queries=rw_n,
        )
        metrics = evaluate(records)
        print_metrics(metrics, label=f"P4 — {label}")
        save_run(
            method, label, records, metrics, results_dir,
            n_questions=n_questions, question_offset=question_offset, k=k,
            hyde_n_samples=n_samples,
            hyde_include_query=include_q,
            hyde_temperature=hyde_temperature,
            q2d_repeat=q2d_rep,
            rewrite_n_queries=rw_n,
        )
        all_metrics[label] = metrics

    # M3 NEW: explicit M3 summary path so it doesn't collide with the M2
    # `p4_ablation.json` if both files coexist in results_dir.
    summary_path = os.path.join(results_dir, "p4_m3_ablation.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"n_questions": n_questions,
                   "question_offset": question_offset,
                   "k": k,
                   "metrics": all_metrics}, f, indent=2)
    print(f"\nM3 ablation summary → {summary_path}")
    return all_metrics


# M3 NEW: held-out test phase. Runs each best config with multiple seeds
# on the held-out slice (default: indices 1000..3609) so the headline
# numbers can be reported as mean ± std rather than single-run point
# estimates — the protocol we agreed on for HyDE T=0.7 and Q2D T=1.0.
def run_m3_test(
    n_questions: int = 2610,
    question_offset: int = 1000,
    k: int = 5,
    seeds=(0, 1, 2),
    index_path: str = DEFAULT_INDEX,
    results_dir: str = DEFAULT_RESULTS_DIR + "_test",
    hyde_temperature: float = 0.7,
    llm: VLLMRunner = None,
    retriever=None,
):
    """
    M3 held-out test phase — run each config in BEST_CONFIGS_M3 with each
    seed in `seeds` on the held-out slice (default: indices 1000..3609).

    For each (config, seed) pair, writes:
        {results_dir}/p4_<label>_seed<s>.json

    Plus an aggregated summary {results_dir}/p4_m3_test.json containing
    mean ± std per metric over the seeds (for the headline table).

    Note: the reader is greedy (T=0) so its output is deterministic given
    a prompt — the seed only matters for HyDE T=0.7 and Q2D T=1.0. We
    still iterate all seeds for the baseline and rewrite for narrative
    consistency, even though the metrics will be identical across seeds
    for those two methods.
    """
    from evaluate import evaluate, print_metrics  # noqa: PLC0415
    import numpy as _np                            # noqa: PLC0415

    if llm is None:
        llm = VLLMRunner()
    if retriever is None:
        retriever = load_retriever(index_path)

    questions_ds = load_questions(n_questions, question_offset)
    summary = {}

    for (method, label, n_samples, include_q, q2d_rep, rw_n) in BEST_CONFIGS_M3:
        per_seed_metrics = []
        if method != "baseline":
            for seed in seeds:
                print(f"\n=== {label} | seed={seed} ===")
                llm.set_seed(seed)        # M3 NEW: switch seed without re-loading the model
                records = run_method(
                    method, llm, retriever, questions_ds,
                    k=k,
                    hyde_n_samples=n_samples,
                    hyde_include_query=include_q,
                    hyde_temperature=hyde_temperature,
                    q2d_repeat=q2d_rep,
                    rewrite_n_queries=rw_n,
                )
                metrics = evaluate(records)
                print_metrics(metrics, label=f"P4 test — {label} seed={seed}")
                save_run(
                    method, f"{label}_seed{seed}", records, metrics, results_dir,
                    n_questions=n_questions, question_offset=question_offset, k=k,
                    seed=seed,
                    hyde_n_samples=n_samples,
                    hyde_include_query=include_q,
                    hyde_temperature=hyde_temperature,
                    q2d_repeat=q2d_rep,
                    rewrite_n_queries=rw_n,
                )
                per_seed_metrics.append(metrics)
        else:
            print(f"\n=== {label}")
            # llm.set_seed(seed)        # M3 NEW: switch seed without re-loading the model
            records = run_method(
                method, llm, retriever, questions_ds,
                k=k,
                hyde_n_samples=n_samples,
                hyde_include_query=include_q,
                hyde_temperature=hyde_temperature,
                q2d_repeat=q2d_rep,
                rewrite_n_queries=rw_n,
            )
            metrics = evaluate(records)
            print_metrics(metrics, label=f"P4 test — {label}")
            save_run(
                method, f"{label}", records, metrics, results_dir,
                n_questions=n_questions, question_offset=question_offset, k=k,
                hyde_n_samples=n_samples,
                hyde_include_query=include_q,
                hyde_temperature=hyde_temperature,
                q2d_repeat=q2d_rep,
                rewrite_n_queries=rw_n,
            )
            per_seed_metrics.append(metrics)

        # M3 NEW: aggregate across seeds. ddof=1 gives sample std; with
        # only 3 seeds this is noisy but it's the honest standard error.
        keys = ("exact_match", "rouge_l", "token_f1", "retrieval_hit_rate")
        agg = {}
        for key in keys:
            vals = [m[key] for m in per_seed_metrics]
            agg[key] = {
                "mean":   float(_np.mean(vals)),
                "std":    float(_np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
        summary[label] = {
            "method": method,
            "config": {
                "hyde_n_samples": n_samples,
                "hyde_include_query": include_q,
                "q2d_repeat": q2d_rep,
                "rewrite_n_queries": rw_n,
            },
            "seeds": list(seeds),
            "aggregated_metrics": agg,
        }

    summary_path = os.path.join(results_dir, "p4_m3_test.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"n_questions": n_questions,
                   "question_offset": question_offset,
                   "k": k,
                   "seeds": list(seeds),
                   "summary": summary}, f, indent=2)
    print(f"\nM3 test summary → {summary_path}")
    return summary

# end M3 NEW
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4 query-stage rewriting pipeline")
    # M3 NEW: `--phase` replaces the previous `--method` as the primary mode
    # selector. The three phases correspond to the M3 workflow:
    #   ablation : sweep METHODS_M3 grid on N=1000 (tuning slice)
    #   test     : multi-seed run of BEST_CONFIGS_M3 on N=2610 (held-out)
    #   single   : M2-style single-method run (kept for back-compat / debug)
    parser.add_argument("--phase", required=True,
                        choices=["ablation", "test", "single"],
                        help="ablation = M3 tuning sweep; "
                             "test = held-out multi-seed; "
                             "single = single-method M2-style run")
    parser.add_argument("--n_questions",     type=int, default=1000)
    parser.add_argument("--question_offset", type=int, default=0)
    parser.add_argument("--k",               type=int, default=5)
    parser.add_argument("--index_path",      default=DEFAULT_INDEX)
    parser.add_argument("--results_dir",     default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--hyde_temperature", type=float, default=0.7)
    # M3 NEW: seeds for the test phase (multi-seed reporting)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for --phase test (multi-seed reporting). "
                         "Only baseline is deterministic; HyDE, Query2doc "
                         "and Rewrite all sample at T>0.")
    # ── args for --phase single only (M2-style) ──
    parser.add_argument("--method", choices=[*METHODS, "all"], default=None,
                        help="(--phase single only) which method to run")
    parser.add_argument("--hyde_n_samples",    type=int, default=1)
    parser.add_argument("--hyde_no_query",     action="store_true")
    parser.add_argument("--q2d_repeat",        type=int, default=1)
    parser.add_argument("--rewrite_n_queries", type=int, default=1,
                        help="(--phase single, method=rewrite) "
                             "M3 NEW — number of search queries per rewrite")
    args = parser.parse_args()

    if args.phase == "ablation":
        run_m3_ablation(
            n_questions=args.n_questions,
            question_offset=args.question_offset,
            k=args.k,
            index_path=args.index_path,
            results_dir=args.results_dir,
            hyde_temperature=args.hyde_temperature,
        )
    elif args.phase == "test":
        run_m3_test(
            n_questions=args.n_questions,
            question_offset=args.question_offset,
            k=args.k,
            seeds=tuple(args.seeds),
            index_path=args.index_path,
            results_dir=args.results_dir,
            hyde_temperature=args.hyde_temperature,
        )
    elif args.phase == "single":
        if args.method is None:
            raise SystemExit("--method is required when --phase=single")
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
            # NOTE: --rewrite_n_queries is currently not threaded through
            # run_pipeline (it's only used by run_m3_ablation/run_m3_test
            # which know about rewrite_n_queries). If you need to do a
            # single multi-query rewrite run for debug, use --phase ablation
            # with a customized METHODS_M3 instead.
        )
