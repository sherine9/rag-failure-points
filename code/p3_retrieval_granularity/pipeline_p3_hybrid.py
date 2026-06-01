"""
P3 Hybrid Router Pipeline

Evaluates question-type routing over pre-computed per-chunker results.
No new generation required : routing decisions are applied to already
computed predictions from pipeline_p3.py runs.

Two routing strategies:
  rule  : per-question-type rules derived from the derivation split
  score : route to the chunker whose FAISS index returns the highest
          top-1 cosine score (unsupervised, no question-type labels)

The evaluation split keeps derivation and evaluation data strictly separate:
  questions 0 .. split-1   → derive routing table
  questions split .. end    → evaluate (never seen during derivation)

Optimisations (score strategy):
  - Encode all eval questions in one batched forward pass (single SGEMM).
  - sentence_only / sentence_window share the small_to_big FAISS binary,
    so only 3 distinct batch FAISS searches are run instead of 5.
  - tqdm progress bar during evaluation.

Usage:
    python pipeline_p3_hybrid.py --strategy rule  --suffix _n2610 --split 1305
    python pipeline_p3_hybrid.py --strategy score --suffix _n2610 --split 1305
    python pipeline_p3_hybrid.py --derive_only --suffix _n2610 --split 1305
"""

import os
import sys
import json
import pickle
import argparse
import faiss
import numpy as np
from pathlib import Path
from collections import defaultdict

_P3       = Path(__file__).resolve().parent
_BASELINE = _P3.parent / "baseline"
sys.path.insert(0, str(_BASELINE))

from evaluate import evaluate, print_metrics

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False
    print("tqdm not found — install with 'pip install tqdm' for a progress bar.")

# Defaults
_RESULTS_DIR = "/scratch/results"
_INDEX_PATHS = {
    "fixed":           "/scratch/faiss_index/p3_fixed",
    "semantic":        "/scratch/faiss_index/p3_semantic",
    "small_to_big":    "/scratch/faiss_index/p3_small_to_big",
    "sentence_only":   "/scratch/faiss_index/p3_sentence_only",
    "sentence_window": "/scratch/faiss_index/p3_sentence_window",
}
_CHUNKERS        = list(_INDEX_PATHS.keys())
_DEFAULT_CHUNKER = "sentence_only"

# The three sentence-based strategies share the exact same FAISS binary.
# We run one batch search for the family and let the tiebreak rule
# distinguish among them — avoiding two redundant identical searches.
_SENTENCE_FAMILY   = {"small_to_big", "sentence_only", "sentence_window"}
_SENTENCE_CANON    = "small_to_big"   # representative index for the family


# Question-type classifier 

_QT_RULES = [
    (("how many", "how much", "how long", "how old", "how far", "how"), "how"),
    (("who",),                                                           "who"),
    (("when", "what year", "what date", "in what year"),                 "when"),
    (("where",),                                                         "where"),
    (("what", "which"),                                                  "what"),
]

def _qtype(question: str) -> str:
    q = question.strip().lower()
    for prefixes, label in _QT_RULES:
        if any(q.startswith(p) for p in prefixes):
            return label
    return "other"


#  Data loading 

def load_chunker_results(results_dir, suffix, chunkers=None):
    chunkers = chunkers or _CHUNKERS
    loaded = {}
    for name in chunkers:
        path = os.path.join(results_dir, f"p3_{name}{suffix}.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping '{name}'")
            continue
        with open(path) as f:
            loaded[name] = json.load(f)["results"]
        print(f"  Loaded '{name}': {len(loaded[name]):,} questions")
    return loaded


#  Routing table derivation 

def derive_routing_table(chunker_results, split):
    type_em    = defaultdict(lambda: defaultdict(float))
    type_count = defaultdict(int)

    ref = next(iter(chunker_results))
    for i in range(split):
        qt = _qtype(chunker_results[ref][i]["question"])
        type_count[qt] += 1
        for name, results in chunker_results.items():
            pred  = results[i]["prediction"].strip().lower()
            is_em = any(pred == g.strip().lower() for g in results[i]["gold_answers"])
            type_em[qt][name] += int(is_em)

    names  = list(chunker_results.keys())
    header = f"{'Type':<10} {'n':>5}" + "".join(f"  {n[:13]:>13}" for n in names) + "  best"
    print(f"\nRouting table derivation (questions 0–{split - 1}):")
    print(header)
    print("─" * len(header))

    routing_table = {}
    for qt in sorted(type_count, key=lambda x: -type_count[x]):
        n         = type_count[qt]
        em_scores = {name: type_em[qt][name] / n for name in names}
        best      = max(em_scores, key=em_scores.get)
        routing_table[qt] = best
        row = f"{qt:<10} {n:>5}"
        for name in names:
            marker = "*" if name == best else " "
            row += f"  {em_scores[name]:>12.3f}{marker}"
        print(row + f"  {best}")

    return routing_table


#  Rule-based routing 

def route_rule(question, routing_table):
    return routing_table.get(_qtype(question), _DEFAULT_CHUNKER)


#  Score-based routing : FAISS index loading 

def _load_retrievers(chunkers):
    """
    Load FAISS indices for score-based routing.

    sentence_only and sentence_window reuse the small_to_big index object
    (same binary on disk), so only 3 distinct indices are loaded into RAM.
    """
    from retriever import Retriever

    retrievers = {}
    base_model = None
    loaded_indices = {}   # path → faiss.Index, to avoid loading the same binary twice

    for name in chunkers:
        path     = _INDEX_PATHS.get(name, "")
        idx_file = f"{path}/faiss.index"
        pkl_file = f"{path}/chunks.pkl"
        npy_file = f"{path}/chunks.npy"

        if not os.path.exists(idx_file):
            print(f"  WARNING: index not found for '{name}' — skipping")
            continue

        if base_model is None:
            base_model = Retriever()

        r       = Retriever.__new__(Retriever)
        r.model = base_model.model

        # Deduplicate by file size: sentence_only / sentence_window / small_to_big
        # are copies of the same binary (shutil.copy2) — same size, same content.
        # Using file size as key avoids loading 33GB three times into RAM.
        dedup_key = os.path.getsize(idx_file)
        if dedup_key in loaded_indices:
            r.index = loaded_indices[dedup_key]
            print(f"  Reused '{name}': {r.index.ntotal:,} vectors (shared index, same binary)")
        else:
            r.index = faiss.read_index(idx_file)
            loaded_indices[dedup_key] = r.index
            print(f"  Loaded '{name}': {r.index.ntotal:,} vectors")

        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                r.chunks = pickle.load(f)
        else:
            r.chunks = np.load(npy_file, allow_pickle=True).tolist()

        retrievers[name] = r

    return retrievers, base_model.model   # also return encoder for batch encoding


#  Batch score computation 

def compute_batch_scores(questions, retrievers, encoder):
    """
    Encode all questions once and run one batched FAISS search per distinct index.

    sentence_only / sentence_window / small_to_big share the same binary,
    so only 3 distinct FAISS searches are performed (fixed, semantic, sentence-family).

    Args:
        questions: list[str]  — all eval questions
        retrievers: dict      — name → Retriever
        encoder:              — shared SentenceTransformer model

    Returns:
        scores: dict[name → np.ndarray shape (n_questions,)]
                top-1 cosine similarity per question per strategy
    """
    n = len(questions)

    # Step 1: encode all questions in one forward pass
    print(f"Encoding {n} questions in one batch...")
    q_embs = encoder.encode(
        questions,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")   # FAISS requires float32

    # Step 2: identify distinct FAISS indices (sentence family shares one binary)
    seen_index_ids = {}   # id(index) → canonical name
    distinct = []         # list of (canonical_name, index)
    family_canon = None

    for name, r in retrievers.items():
        idx_id = id(r.index)
        if idx_id not in seen_index_ids:
            seen_index_ids[idx_id] = name
            distinct.append((name, r.index))
            if name in _SENTENCE_FAMILY:
                family_canon = name

    print(f"Running {len(distinct)} distinct batch FAISS searches "
          f"(sentence family counts as one)...")

    # Step 3: batch search for each distinct index
    canon_scores = {}
    for canon_name, index in distinct:
        print(f"  Searching {canon_name} index ({index.ntotal:,} vectors × {n} queries)...")
        D, _ = index.search(q_embs, 1)   # D: (n, 1)
        canon_scores[canon_name] = D[:, 0]   # shape (n,)

    # Step 4: map back to all strategy names
    scores = {}
    for name, r in retrievers.items():
        idx_id  = id(r.index)
        canon   = seen_index_ids[idx_id]
        scores[name] = canon_scores[canon]

    return scores   # dict[name → ndarray(n,)]


#  Hybrid evaluation 

def evaluate_hybrid(chunker_results, strategy, split,
                    routing_table=None, retrievers=None, encoder=None):
    """
    Simulate hybrid routing on the held-out split (split..end).

    For 'score' strategy: encodes all questions once and runs batched
    FAISS searches — O(3 × n × d) instead of O(5 × n × d) sequential calls.
    """
    n_total        = len(next(iter(chunker_results.values())))
    ref            = next(iter(chunker_results))
    n_eval         = n_total - split
    hybrid_results = []
    routing_counts = defaultdict(int)

    if strategy == "score":
        # ── Optimised path: batch encode + batch FAISS ────────────────────────
        questions = [chunker_results[ref][i]["question"] for i in range(split, n_total)]
        batch_scores = compute_batch_scores(questions, retrievers, encoder)

        names = list(retrievers.keys())
        print(f"\nApplying routing decisions for {n_eval} questions...")
        iter_range = tqdm(range(n_eval), desc="Routing", unit="q") if _TQDM \
                     else range(n_eval)

        for j in iter_range:
            i = split + j
            question = questions[j]

            # Scores for this question across all strategies
            q_scores = {name: float(batch_scores[name][j]) for name in names}
            max_score = max(q_scores.values())

            # Tiebreak: sentence family always ties — use rule table within family
            tied = [name for name, s in q_scores.items() if s == max_score]
            if len(tied) == 1:
                chosen = tied[0]
            elif routing_table is not None:
                rule_choice = routing_table.get(_qtype(question), _DEFAULT_CHUNKER)
                chosen = rule_choice if rule_choice in tied else tied[0]
            else:
                chosen = tied[0]

            if chosen not in chunker_results:
                chosen = _DEFAULT_CHUNKER

            routing_counts[chosen] += 1
            hybrid_results.append({**chunker_results[chosen][i], "routed_to": chosen})

    else:
        iter_range = tqdm(range(split, n_total), desc="Routing", unit="q") if _TQDM \
                     else range(split, n_total)

        for i in iter_range:
            question = chunker_results[ref][i]["question"]
            chosen   = route_rule(question, routing_table)
            if chosen not in chunker_results:
                chosen = _DEFAULT_CHUNKER
            routing_counts[chosen] += 1
            hybrid_results.append({**chunker_results[chosen][i], "routed_to": chosen})

    metrics = evaluate(hybrid_results)
    return hybrid_results, metrics, dict(routing_counts)


#  Comparison table 

def print_comparison(chunker_results, split, hybrid_metrics, strategy):
    print(f"\n{'Chunker':<20} {'EM':>6} {'ROUGE-L':>8} {'F1':>7} {'Hit':>6}")
    print("─" * 52)
    for name, results in chunker_results.items():
        m = evaluate(results[split:])
        print(f"{name:<20} {m['exact_match']:>6.4f} {m['rouge_l']:>8.4f} "
              f"{m['token_f1']:>7.4f} {m['retrieval_hit_rate']:>6.4f}")
    m = hybrid_metrics
    print(f"{'hybrid (' + strategy + ')':<20} {m['exact_match']:>6.4f} {m['rouge_l']:>8.4f} "
          f"{m['token_f1']:>7.4f} {m['retrieval_hit_rate']:>6.4f}")


#  Main entry point 

def run(strategy="rule", suffix="_n2610", split=1305,
        results_dir=_RESULTS_DIR, chunkers=None,
        results_path=None, derive_only=False):

    chunkers     = chunkers or _CHUNKERS
    results_path = results_path or os.path.join(
        results_dir, f"p3_hybrid_{strategy}{suffix}.json"
    )

    print(f"Loading pre-computed results (suffix='{suffix}')...")
    chunker_results = load_chunker_results(results_dir, suffix, chunkers)
    if not chunker_results:
        raise RuntimeError("No result files found. Run pipeline_p3.py first.")

    routing_table = derive_routing_table(chunker_results, split)

    if derive_only:
        return {}, {}, routing_table

    retrievers = None
    encoder    = None

    if strategy == "score":
        print("\nLoading FAISS indices for score-based routing...")
        retrievers, encoder = _load_retrievers(chunkers)

    n_eval = len(next(iter(chunker_results.values()))) - split
    print(f"\nEvaluating hybrid [{strategy}] on {n_eval} held-out questions "
          f"(questions {split}–{split + n_eval - 1})...")

    hybrid_results, metrics, routing_counts = evaluate_hybrid(
        chunker_results, strategy, split,
        routing_table=routing_table,
        retrievers=retrievers,
        encoder=encoder,
    )

    print_metrics(metrics, label=f"P3 — hybrid ({strategy})")
    print(f"Routing distribution: {routing_counts}")
    print_comparison(chunker_results, split, metrics, strategy)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "strategy":       strategy,
            "suffix":         suffix,
            "split":          split,
            "routing_table":  routing_table,
            "routing_counts": routing_counts,
            "metrics":        metrics,
            "results":        hybrid_results,
        }, f, indent=2)
    print(f"\nSaved → {results_path}")

    return hybrid_results, metrics, routing_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3 hybrid router evaluation")
    parser.add_argument("--strategy",    default="rule", choices=["rule", "score"])
    parser.add_argument("--suffix",      default="_n2610")
    parser.add_argument("--split",       type=int, default=1305)
    parser.add_argument("--results_dir", default=_RESULTS_DIR)
    parser.add_argument("--derive_only", action="store_true")
    args = parser.parse_args()

    run(
        strategy=args.strategy,
        suffix=args.suffix,
        split=args.split,
        results_dir=args.results_dir,
        derive_only=args.derive_only,
    )
