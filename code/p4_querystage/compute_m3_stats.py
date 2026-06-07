"""
compute_m3_stats.py - read the FULL uncondensed P4 held-out test JSONs
(N=2610, 3 seeds for the best config per method) and emit small, committable
stats JSONs the notebook can load without ever touching a model or FAISS.

Reads ONLY the 10 best-config files:
    p4_best_baseline.json
    p4_best_hyde_seed{0,1,2}.json
    p4_best_query2doc_seed{0,1,2}.json
    p4_best_rewrite_seed{0,1,2}.json

Ablation runs (N=1000, single seed) are intentionally skipped - their
aggregate `metrics` field is already preserved in the condensed JSONs
committed under results/p4/, so the notebook reads those directly for the
ablation curve plot.

Outputs to <dst>/:
    p4_stats.json         - aggregated stats, p-values, CIs, bias rho
    p4_sample_level.json  - per-question EM/hit bools (multi-seed for stochastic methods)
    p4_cases.json         - top-K recoveries / regressions vs baseline for case studies

Usage (on the cluster):

    cd /scratch/open-project-m3-mnlp212/code/p4_querystage
    python compute_m3_stats.py \
        --src /scratch/results/p4/m3 \
        --dst /scratch/open-project-m3-mnlp212/results/p4

Then commit the three output JSONs from results/p4/.

The script needs only numpy, scipy, and rouge_score - all already installed
in the baseline env. No GPU, no vLLM, no FAISS. Runtime: ~2 minutes on CPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sst

# ---- sys.path: shared baseline + this directory ------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "baseline"))
sys.path.insert(0, str(HERE))

from evaluate import exact_match, rouge_l, token_f1                # noqa: E402
from bias_analysis import feature_score_correlations               # noqa: E402


# ----------------------------------------------------------------------------
# Question typing and abstention (mirrors Rania's P3 notebook so cross-track
# analyses are directly comparable)
# ----------------------------------------------------------------------------
def qtype(q: str) -> str:
    q = q.strip().lower()
    for prefixes, label in [
        (("how many", "how much", "how long", "how old", "how far", "how"), "how"),
        (("who",), "who"),
        (("when", "what year", "what date", "in what year"), "when"),
        (("where",), "where"),
        (("what", "which"), "what"),
    ]:
        if any(q.startswith(p) for p in prefixes):
            return label
    return "other"


_ABSTENTION_PATS = (
    "context does not", "not provide", "no information", "cannot find",
    "not found in", "does not mention", "not mentioned", "i don",
    "cannot provide", "the passage does not", "there is no",
    "no context", "unable to find", "not stated",
)


def is_abstention(prediction: str) -> bool:
    p = prediction.lower()
    return any(pat in p for pat in _ABSTENTION_PATS)


# ----------------------------------------------------------------------------
# Per-sample scoring
# ----------------------------------------------------------------------------
def score_records(records: List[Dict]) -> Dict[str, np.ndarray]:
    """Recompute per-sample EM/hit/F1/ROUGE-L and abstention bool from raw
    records (matches evaluate.py exactly for the metric quartet)."""
    n = len(records)
    em       = np.zeros(n, dtype=np.int8)
    hit      = np.zeros(n, dtype=np.int8)
    abstain  = np.zeros(n, dtype=np.int8)
    f1       = np.zeros(n, dtype=np.float32)
    rl       = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(records):
        pred = r["prediction"]
        gold = r["gold_answers"]
        em[i]      = exact_match(pred, gold)
        f1[i]      = token_f1(pred, gold)
        rl[i]      = rouge_l(pred, gold)
        abstain[i] = int(is_abstention(pred))
        chunks_text = " ".join(r.get("chunks") or []).lower()
        hit[i] = int(any(a.lower() in chunks_text for a in gold))
    return {"em": em, "hit": hit, "f1": f1, "rouge_l": rl, "abstain": abstain}


# ----------------------------------------------------------------------------
# Statistical tests
# ----------------------------------------------------------------------------
def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> Dict:
    """Two-sided exact McNemar test for paired binary outcomes.

    H0: P(x=0, y=1) == P(x=1, y=0).
    Returns {p_value, b, c, n_discordant} where b = (x=0 & y=1) and
    c = (x=1 & y=0).
    """
    assert x.shape == y.shape and x.ndim == 1
    b = int(((x == 0) & (y == 1)).sum())  # control wrong, treated right (a recovery)
    c = int(((x == 1) & (y == 0)).sum())  # control right, treated wrong (a regression)
    if b + c == 0:
        return {"p_value": 1.0, "b": b, "c": c, "n_discordant": 0}
    k = min(b, c)
    p = float(min(1.0, 2.0 * sst.binom.cdf(k, b + c, 0.5)))
    return {"p_value": p, "b": b, "c": c, "n_discordant": b + c}


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Dict:
    """Paired t-test on y - x (treatment - control). Returns mean_diff and two-sided p."""
    diff = y.astype(np.float64) - x.astype(np.float64)
    if not np.any(diff):
        return {"mean_diff": 0.0, "p_value": 1.0, "n": int(x.size)}
    _, p = sst.ttest_rel(y, x)
    return {"mean_diff": float(diff.mean()), "p_value": float(p), "n": int(x.size)}


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                      seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for mean(x)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000,
                             alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Paired bootstrap CI for mean(y) - mean(x). Resamples question indices,
    not the two arms independently - preserves the within-question pairing."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = y[idx].mean(axis=1) - x[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_run(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
BEST_LABELS = ("best_baseline", "best_hyde", "best_query2doc", "best_rewrite")
SEEDS = (0, 1, 2)


def collect_test(src: str, n_boot: int):
    """For each best config x seed, score records and aggregate."""
    test_stats: Dict[str, Dict] = {}
    sample_level: Dict[str, Dict] = {}
    records_lookup: Dict[Tuple[str, object], List[Dict]] = {}  # (label, seed) -> raw records

    for label in BEST_LABELS:
        is_baseline = (label == "best_baseline")
        seed_list = (None,) if is_baseline else SEEDS

        per_seed_scores = {}
        for seed in seed_list:
            fname = f"p4_{label}.json" if seed is None else f"p4_{label}_seed{seed}.json"
            path = os.path.join(src, fname)
            if not os.path.exists(path):
                print(f"  [test] MISSING: {fname}")
                continue
            data = load_run(path)
            s = score_records(data["results"])
            per_seed_scores[seed] = s
            records_lookup[(label, seed)] = data["results"]

        if not per_seed_scores:
            print(f"  [test] {label}: no files found, skipping")
            continue

        n = next(iter(per_seed_scores.values()))["em"].size
        metric_keys = ("em", "hit", "rouge_l", "f1")
        per_seed_means = {
            k: [float(s[k].mean()) for s in per_seed_scores.values()]
            for k in metric_keys
        }
        agg = {}
        for k in metric_keys:
            vals = per_seed_means[k]
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [round(v, 5) for v in vals],
            }
        # Bootstrap CI from seed 0 (single, canonical seed)
        seed0_key = None if is_baseline else 0
        if seed0_key not in per_seed_scores:
            seed0_key = next(iter(per_seed_scores.keys()))
        s0 = per_seed_scores[seed0_key]
        ci_em  = bootstrap_ci_mean(s0["em"],      n_boot=n_boot)
        ci_hit = bootstrap_ci_mean(s0["hit"],     n_boot=n_boot)
        ci_f1  = bootstrap_ci_mean(s0["f1"],      n_boot=n_boot)
        ci_rl  = bootstrap_ci_mean(s0["rouge_l"], n_boot=n_boot)

        test_stats[label] = {
            "n_samples": int(n),
            "seeds": [k for k in per_seed_scores.keys()],
            "metrics_agg": agg,
            "bootstrap_ci_95_seed0": {
                "em":       [round(ci_em[0],  4), round(ci_em[1],  4)],
                "hit":      [round(ci_hit[0], 4), round(ci_hit[1], 4)],
                "rouge_l":  [round(ci_rl[0],  4), round(ci_rl[1],  4)],
                "token_f1": [round(ci_f1[0],  4), round(ci_f1[1],  4)],
            },
        }
        print(f"  [test] {label:<18} N={n}  "
              f"EM={agg['em']['mean']:.4f} +/- {agg['em']['std']:.4f}  "
              f"Hit={agg['hit']['mean']:.4f} +/- {agg['hit']['std']:.4f}")

        # Sample-level: store EM and Hit only (binary, small) for downstream tests
        if is_baseline:
            sample_level[label] = {
                "em":  per_seed_scores[None]["em"].astype(int).tolist(),
                "hit": per_seed_scores[None]["hit"].astype(int).tolist(),
            }
        else:
            sample_level[label] = {}
            for seed, s in per_seed_scores.items():
                sample_level[label][f"seed_{seed}"] = {
                    "em":  s["em"].astype(int).tolist(),
                    "hit": s["hit"].astype(int).tolist(),
                }

    return test_stats, sample_level, records_lookup


def paired_tests_vs_baseline(sample_level: Dict, records_lookup: Dict, n_boot: int) -> Dict:
    """McNemar on EM/Hit and paired bootstrap on metric deltas, per method vs baseline.

    For multi-seed methods, runs the test once per seed and reports all three
    p-values; the summary uses the WORST (max) p-value to be conservative.
    Continuous metrics (F1, ROUGE-L) get a paired t-test on seed 0 records.
    """
    if "best_baseline" not in sample_level:
        return {}
    base = sample_level["best_baseline"]
    base_em  = np.array(base["em"],  dtype=np.int8)
    base_hit = np.array(base["hit"], dtype=np.int8)

    # Pre-score baseline records (for F1/ROUGE-L per-sample comparison)
    base_recs = records_lookup.get(("best_baseline", None))
    base_scores = score_records(base_recs) if base_recs else None

    out = {}
    for label in ("best_hyde", "best_query2doc", "best_rewrite"):
        if label not in sample_level:
            continue
        sl = sample_level[label]
        per_seed = {}
        for seed_key, vals in sl.items():
            seed_int = int(seed_key.split("_")[1])
            em  = np.array(vals["em"],  dtype=np.int8)
            hit = np.array(vals["hit"], dtype=np.int8)
            entry = {
                "em_mcnemar":   mcnemar_exact(base_em,  em),
                "hit_mcnemar":  mcnemar_exact(base_hit, hit),
                "em_diff_ci_95":  list(paired_bootstrap_diff_ci(base_em,  em,  n_boot=n_boot)),
                "hit_diff_ci_95": list(paired_bootstrap_diff_ci(base_hit, hit, n_boot=n_boot)),
                "em_diff_pp":   round(float((em.mean()  - base_em.mean()) * 100),  4),
                "hit_diff_pp":  round(float((hit.mean() - base_hit.mean()) * 100), 4),
            }
            # Continuous metrics: re-score from records on this seed
            recs = records_lookup.get((label, seed_int))
            if recs is not None and base_scores is not None:
                t_scores = score_records(recs)
                entry["f1_paired_t"]      = paired_ttest(base_scores["f1"],      t_scores["f1"])
                entry["rouge_l_paired_t"] = paired_ttest(base_scores["rouge_l"], t_scores["rouge_l"])
                entry["f1_diff_ci_95"] = list(paired_bootstrap_diff_ci(
                    base_scores["f1"], t_scores["f1"], n_boot=n_boot))
                entry["rouge_l_diff_ci_95"] = list(paired_bootstrap_diff_ci(
                    base_scores["rouge_l"], t_scores["rouge_l"], n_boot=n_boot))
            per_seed[seed_key] = entry

        em_diffs  = [v["em_diff_pp"]  for v in per_seed.values()]
        hit_diffs = [v["hit_diff_pp"] for v in per_seed.values()]
        em_ps     = [v["em_mcnemar"]["p_value"]  for v in per_seed.values()]
        hit_ps    = [v["hit_mcnemar"]["p_value"] for v in per_seed.values()]
        out[label] = {
            "per_seed": per_seed,
            "summary": {
                "em_diff_pp_mean":   round(float(np.mean(em_diffs)), 4),
                "em_diff_pp_std":    round(float(np.std(em_diffs, ddof=1)) if len(em_diffs) > 1 else 0.0, 4),
                "em_mcnemar_pvalue_max":  round(float(max(em_ps)),  6),
                "hit_diff_pp_mean":  round(float(np.mean(hit_diffs)), 4),
                "hit_diff_pp_std":   round(float(np.std(hit_diffs, ddof=1)) if len(hit_diffs) > 1 else 0.0, 4),
                "hit_mcnemar_pvalue_max": round(float(max(hit_ps)), 6),
            },
        }
        print(f"  [paired] {label:<18} dEM={out[label]['summary']['em_diff_pp_mean']:+.2f}pp  "
              f"McNemar(EM) max-p={out[label]['summary']['em_mcnemar_pvalue_max']:.4g}")
    return out


def em_given_hit_table(sample_level: Dict) -> Dict:
    """Conditional EM | hit / EM | miss per method (Rania-style decomposition)."""
    out = {}
    for label, sl in sample_level.items():
        if "em" in sl:
            seeds_data = [(np.array(sl["em"],  dtype=np.int8),
                           np.array(sl["hit"], dtype=np.int8))]
        else:
            seeds_data = []
            for seed_key, vals in sl.items():
                seeds_data.append((np.array(vals["em"],  dtype=np.int8),
                                   np.array(vals["hit"], dtype=np.int8)))
        eh_vals, em_vals = [], []
        n_hits_list, n_miss_list = [], []
        for em, hit in seeds_data:
            mh, mm = (hit == 1), (hit == 0)
            eh_vals.append(float(em[mh].mean()) if mh.sum() else 0.0)
            em_vals.append(float(em[mm].mean()) if mm.sum() else 0.0)
            n_hits_list.append(int(mh.sum()))
            n_miss_list.append(int(mm.sum()))
        out[label] = {
            "em_given_hit_mean":  round(float(np.mean(eh_vals)), 4),
            "em_given_hit_std":   round(float(np.std(eh_vals, ddof=1)) if len(eh_vals)  > 1 else 0.0, 4),
            "em_given_miss_mean": round(float(np.mean(em_vals)), 4),
            "em_given_miss_std":  round(float(np.std(em_vals, ddof=1)) if len(em_vals) > 1 else 0.0, 4),
            "n_hits_mean":   round(float(np.mean(n_hits_list)), 2),
            "n_misses_mean": round(float(np.mean(n_miss_list)), 2),
        }
    return out


def collect_bias_diagnostic(src: str) -> Dict:
    """Spearman rho between top-1 retrieval score and surface features
    (brevity, literal overlap, answer position, query-term repetition).

    Note: rewrite multi-query stores max_retrieval_score=0 by construction
    (see pipeline_p4.py retrieve()), so its rho is meaningless. Skipped.
    """
    out = {}
    targets = [
        # Held-out test (N=2610, seed 0 for stochastic methods)
        ("best_baseline",           "p4_best_baseline.json"),
        ("best_hyde_seed0",         "p4_best_hyde_seed0.json"),
        ("best_query2doc_seed0",    "p4_best_query2doc_seed0.json"),
    ]
    for key, fname in targets:
        path = os.path.join(src, fname)
        if not os.path.exists(path):
            print(f"  [bias] MISSING: {fname}")
            continue
        data = load_run(path)
        rho = feature_score_correlations(data["results"])
        out[key] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rho.items()}
        print(f"  [bias] {key:<24} brevity={rho['brevity_rho']:+.3f}  "
              f"literal={rho['literal_rho']:+.3f}  pos={rho['position_rho']:+.3f}  "
              f"rep={rho['repetition_rho']:+.3f}  n={rho['n']}")
    return out


def collect_qtype_breakdown(records_lookup: Dict) -> Dict:
    """EM and Hit broken down by question type per method.

    For test-set methods with 3 seeds, reports mean/std across seeds.
    Question type is assigned once from the baseline records (questions are
    identical across methods by construction; using the baseline as the
    authoritative source avoids inconsistencies).
    """
    if ("best_baseline", None) not in records_lookup:
        return {}
    base_recs = records_lookup[("best_baseline", None)]
    questions = [r["question"] for r in base_recs]
    qts = np.array([qtype(q) for q in questions])
    types_in_order = ["who", "when", "what", "where", "how", "other"]
    counts = {t: int((qts == t).sum()) for t in types_in_order}

    def by_qtype(em: np.ndarray, hit: np.ndarray) -> Dict[str, Dict]:
        out = {}
        for t in types_in_order:
            mask = (qts == t)
            n_t = int(mask.sum())
            if n_t == 0:
                continue
            out[t] = {
                "n":  n_t,
                "em":  float(em[mask].mean()),
                "hit": float(hit[mask].mean()),
            }
        return out

    out: Dict[str, Dict] = {"__counts__": counts}
    for label in BEST_LABELS:
        is_baseline = (label == "best_baseline")
        seeds = (None,) if is_baseline else SEEDS
        per_seed = []
        for seed in seeds:
            recs = records_lookup.get((label, seed))
            if recs is None:
                continue
            s = score_records(recs)
            per_seed.append(by_qtype(s["em"], s["hit"]))
        if not per_seed:
            continue
        if is_baseline:
            out[label] = per_seed[0]   # single seed -> store directly
        else:
            agg = {}
            for t in types_in_order:
                em_vals  = [d[t]["em"]  for d in per_seed if t in d]
                hit_vals = [d[t]["hit"] for d in per_seed if t in d]
                if not em_vals:
                    continue
                agg[t] = {
                    "n":       per_seed[0][t]["n"],
                    "em_mean": float(np.mean(em_vals)),
                    "em_std":  float(np.std(em_vals, ddof=1)) if len(em_vals) > 1 else 0.0,
                    "hit_mean": float(np.mean(hit_vals)),
                    "hit_std":  float(np.std(hit_vals, ddof=1)) if len(hit_vals) > 1 else 0.0,
                }
            out[label] = agg
        print(f"  [qtype] {label} broken down across {len(types_in_order)} types")
    return out


def collect_abstention(records_lookup: Dict) -> Dict:
    """Abstention rate (refusal-to-answer) per method, decomposed by hit/miss.

    A 'false abstention' is the model refusing even though the gold answer
    appears in the retrieved chunks - a pure generation failure that hurts
    EM without retrieval being the bottleneck. A 'justified abstention' is
    the model refusing when retrieval missed - the correct RAG behaviour.
    """
    out: Dict[str, Dict] = {}
    for label in BEST_LABELS:
        is_baseline = (label == "best_baseline")
        seeds = (None,) if is_baseline else SEEDS
        per_seed_total, per_seed_false, per_seed_just = [], [], []
        for seed in seeds:
            recs = records_lookup.get((label, seed))
            if recs is None:
                continue
            s = score_records(recs)
            abst = s["abstain"]
            hit  = s["hit"]
            n = abst.size
            per_seed_total.append(float(abst.mean()))
            per_seed_false.append(float(((abst == 1) & (hit == 1)).sum() / n))
            per_seed_just .append(float(((abst == 1) & (hit == 0)).sum() / n))
        if not per_seed_total:
            continue
        def agg(vals):
            return {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        out[label] = {
            "abstention_rate":       agg(per_seed_total),
            "false_abstention_rate": agg(per_seed_false),
            "just_abstention_rate":  agg(per_seed_just),
        }
        print(f"  [abstention] {label:<18} total={out[label]['abstention_rate']['mean']:.3f}  "
              f"false(hit=1)={out[label]['false_abstention_rate']['mean']:.3f}  "
              f"just(hit=0)={out[label]['just_abstention_rate']['mean']:.3f}")
    return out


def collect_intermediate_lengths(records_lookup: Dict) -> Dict:
    """Mean/median/std of the LLM intermediate output length (words) per method.

    Contextualises bias_diagnostic['brevity_rho']: if HyDE pseudo-docs average
    150 words and Q2D averages 40, then any brevity bias in retrieval has very
    different effective strength across methods.

    Handles three shapes of `intermediate`:
        - str  (Q2D single doc, baseline echo of the question, rewrite single)
        - list of str (HyDE n_samples>1, rewrite multi-query)
        - None (defensive)
    """
    out: Dict[str, Dict] = {}
    for label in BEST_LABELS:
        is_baseline = (label == "best_baseline")
        seeds = (None,) if is_baseline else (0,)   # seed 0 is enough; not stochastic in structure
        for seed in seeds:
            recs = records_lookup.get((label, seed))
            if recs is None:
                continue
            lens = []
            for r in recs:
                inter = r.get("intermediate")
                if inter is None:
                    continue
                if isinstance(inter, str):
                    lens.append(len(inter.split()))
                elif isinstance(inter, list):
                    for x in inter:
                        if isinstance(x, str):
                            lens.append(len(x.split()))
            if not lens:
                continue
            arr = np.asarray(lens, dtype=np.float32)
            out[label] = {
                "n_intermediates": int(arr.size),
                "mean_words":   float(arr.mean()),
                "median_words": float(np.median(arr)),
                "std_words":    float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "p10_words":    float(np.percentile(arr, 10)),
                "p90_words":    float(np.percentile(arr, 90)),
            }
            print(f"  [intermediate] {label:<18} N={arr.size:>5}  "
                  f"mean={arr.mean():6.1f}w  median={np.median(arr):6.1f}w  "
                  f"p10={np.percentile(arr, 10):5.1f}  p90={np.percentile(arr, 90):6.1f}")
    return out


def collect_cases(records_lookup: Dict, sample_level: Dict, top_k: int = 25) -> Dict:
    """Top-K recoveries and regressions per method (seed 0) vs baseline.

    Recovery   : baseline EM=0 AND treated EM=1
    Regression : baseline EM=1 AND treated EM=0
    Selection criterion within each category: simply the first K indices,
    since all are equally "extreme" for binary EM. Stored fields preview
    chunks (max 400 chars) and intermediates (full).
    """
    if ("best_baseline", None) not in records_lookup:
        return {}
    base_recs = records_lookup[("best_baseline", None)]
    base_em   = np.array(sample_level["best_baseline"]["em"], dtype=np.int8)
    n = len(base_em)

    out = {}
    for label in ("best_hyde", "best_query2doc", "best_rewrite"):
        if (label, 0) not in records_lookup:
            continue
        treat_recs = records_lookup[(label, 0)]
        if f"seed_0" not in sample_level.get(label, {}):
            continue
        treat_em = np.array(sample_level[label]["seed_0"]["em"], dtype=np.int8)
        recoveries  = [i for i in range(n) if base_em[i] == 0 and treat_em[i] == 1]
        regressions = [i for i in range(n) if base_em[i] == 1 and treat_em[i] == 0]

        def to_case(i):
            return {
                "i":             int(i),
                "question":      base_recs[i]["question"],
                "gold":          base_recs[i]["gold_answers"],
                "baseline_pred": base_recs[i]["prediction"],
                "treated_pred":  treat_recs[i]["prediction"],
                "baseline_top1_chunk": (base_recs[i]["chunks"] or [""])[0][:400],
                "treated_top1_chunk":  (treat_recs[i]["chunks"] or [""])[0][:400],
                "intermediate":  treat_recs[i].get("intermediate"),
            }

        out[label] = {
            "n_recoveries_total":  len(recoveries),
            "n_regressions_total": len(regressions),
            "recoveries":  [to_case(i) for i in recoveries[:top_k]],
            "regressions": [to_case(i) for i in regressions[:top_k]],
        }
        print(f"  [cases] {label:<18} {len(recoveries):>4} recoveries  "
              f"{len(regressions):>4} regressions  (kept top {top_k} each)")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/scratch/results/p4/m3",
                        help="Directory containing the uncondensed P4 JSONs "
                             "(p4_<label>.json and p4_best_*_seed*.json).")
    parser.add_argument("--dst", default=str(HERE.parent.parent / "results" / "p4"),
                        help="Output directory for the committable stats JSONs.")
    parser.add_argument("--n_boot", type=int, default=5000,
                        help="Bootstrap replicates per CI (default 5000).")
    parser.add_argument("--top_k_cases", type=int, default=25,
                        help="Number of recoveries / regressions per method to keep.")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"--src directory does not exist: {args.src}")
    os.makedirs(args.dst, exist_ok=True)

    print(f"== Reading from {args.src}")
    print(f"== Writing  to  {args.dst}")
    print(f"== n_boot = {args.n_boot}, top_k_cases = {args.top_k_cases}")

    print("\n== Test phase (N=2610, 3 seeds, best configs only) ==")
    test_stats, sample_level, records_lookup = collect_test(args.src, n_boot=args.n_boot)

    print("\n== Paired tests vs baseline ==")
    paired = paired_tests_vs_baseline(sample_level, records_lookup, n_boot=args.n_boot)

    print("\n== EM | Hit decomposition ==")
    em_hit = em_given_hit_table(sample_level)

    print("\n== Bias diagnostic (Spearman rho, surface features) ==")
    bias = collect_bias_diagnostic(args.src)

    print("\n== Question-type breakdown ==")
    qtype_breakdown = collect_qtype_breakdown(records_lookup)

    print("\n== Abstention rates ==")
    abstention = collect_abstention(records_lookup)

    print("\n== Intermediate (pseudo-doc / rewrite) length stats ==")
    inter_lens = collect_intermediate_lengths(records_lookup)

    print(f"\n== Case studies (top {args.top_k_cases} per method) ==")
    cases = collect_cases(records_lookup, sample_level, top_k=args.top_k_cases)

    # --- p4_stats.json ---
    stats = {
        "metadata": {
            "test_n":         2610,
            "test_offset":    1000,
            "seeds":          list(SEEDS),
            "n_boot":         args.n_boot,
            "ci_alpha":       0.05,
            "src":            args.src,
            "note":           "Stats computed only on the held-out test slice. "
                              "Ablation curves are plotted from the condensed "
                              "p4_<config>.json files already committed under "
                              "results/p4/ (their `metrics` field was preserved).",
        },
        "test":                 test_stats,
        "paired_vs_baseline":   paired,
        "em_given_hit":         em_hit,
        "bias_diagnostic":      bias,
        "qtype_breakdown":      qtype_breakdown,
        "abstention":           abstention,
        "intermediate_lengths": inter_lens,
    }
    out_stats = os.path.join(args.dst, "p4_stats.json")
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[OK] stats   -> {out_stats}  ({os.path.getsize(out_stats)/1024:.1f} KB)")

    # --- p4_sample_level.json --- (no indent: keep small)
    sample_level["__metadata__"] = {
        "n":      2610,
        "offset": 1000,
        "seeds":  list(SEEDS),
        "note":   "Per-question EM/hit bools (0/1). Best baseline is a single "
                  "deterministic run; HyDE/Q2D/Rewrite each have 3 seeds.",
    }
    out_sl = os.path.join(args.dst, "p4_sample_level.json")
    with open(out_sl, "w", encoding="utf-8") as f:
        json.dump(sample_level, f)   # no indent on purpose
    print(f"[OK] samples -> {out_sl}  ({os.path.getsize(out_sl)/1024:.1f} KB)")

    # --- p4_cases.json ---
    cases["__metadata__"] = {
        "top_k_per_method": args.top_k_cases,
        "selection":        "first-K indices satisfying EM-flip criterion (seed 0)",
        "note":              "Chunks truncated to 400 chars. Use for case study display only.",
    }
    out_cases = os.path.join(args.dst, "p4_cases.json")
    with open(out_cases, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"[OK] cases   -> {out_cases}  ({os.path.getsize(out_cases)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
