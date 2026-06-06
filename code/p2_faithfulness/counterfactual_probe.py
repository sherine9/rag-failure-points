"""
Counterfactual (Wallat-style) Faithfulness Probe — P2 Track (Leïla Benjelloun)
===============================================================================
Diagnoses whether unfaithful answers stem from:
  (C) Context ignored / post-rationalization — the model answers from
      parametric memory regardless of what context is provided.
  (A) Retrieval failure — the right document was never fetched, so the
      model had no useful signal.

Probe design (inspired by Wallat et al., 2024):
  For each candidate sample, re-run the generator with its retrieved chunks
  replaced by randomly selected chunks from a DIFFERENT question.
  - Stable answer (high F1 vs original) → model ignored context → post-rationalization.
  - Changed answer (low F1 vs original) → model was reading context → retrieval failure.

Answer stability is measured with token-level F1 (same metric as NQ Open eval).
"""

import json
import os
import re
import sys
import numpy as np
from tqdm import tqdm


# ── 1. Answer normalisation & comparison ──────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between two answer strings (after normalisation)."""
    p_toks = set(_normalize(pred).split())
    r_toks = set(_normalize(ref).split())
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0
    common = len(p_toks & r_toks)
    prec = common / len(p_toks)
    rec  = common / len(r_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def is_stable(original: str, counterfactual: str, f1_threshold: float = 0.5) -> bool:
    """
    Returns True if the counterfactual answer is close enough to the original
    to suggest the model was not reading the context (post-rationalization).
    """
    if _normalize(original) == _normalize(counterfactual):
        return True
    return token_f1(original, counterfactual) >= f1_threshold


# ── 2. Random-chunk selection ──────────────────────────────────────────────────

def get_random_chunks(results: list, idx: int, rng: np.random.Generator) -> list:
    """
    Return chunks from a randomly selected OTHER sample.
    Using another sample's chunks guarantees genuine topic mismatch.
    """
    other_idxs = [j for j in range(len(results)) if j != idx]
    donor = int(rng.choice(other_idxs))
    return results[donor]['chunks']


# ── 3. Main probe ──────────────────────────────────────────────────────────────

def run_counterfactual_probe(
    results: list,
    candidate_indices: list,
    generator,
    seed: int = 42,
    f1_threshold: float = 0.5,
) -> dict:
    """
    Run the Wallat-style counterfactual probe on a subset of samples.

    Args:
        results:           full list of pipeline result dicts
        candidate_indices: indices to probe (typically low-faith samples)
        generator:         Generator instance (Mistral-7B)
        seed:              random seed for reproducible chunk sampling
        f1_threshold:      token F1 cutoff for 'stable' answer judgement

    Returns:
        dict with aggregate metrics and per-sample probe results
    """
    rng = np.random.default_rng(seed)
    probe_results = []

    for idx in tqdm(candidate_indices, desc="Counterfactual probe"):
        r = results[idx]
        random_chunks = get_random_chunks(results, idx, rng)
        cf_answer = generator.generate(r['question'], random_chunks)

        f1     = token_f1(r['prediction'], cf_answer)
        stable = is_stable(r['prediction'], cf_answer, f1_threshold)

        probe_results.append({
            'index':                 idx,
            'question':              r['question'],
            'original_answer':       r['prediction'],
            'counterfactual_answer': cf_answer,
            'random_chunk_donor':    int(rng.integers(0, len(results))),  # for reference
            'answer_stable':         stable,
            'token_f1':              round(f1, 4),
            'original_ragas':        r.get('faithfulness_score'),
            'original_nli':          r.get('faithfulness_nli_score'),
        })

    n_stable  = sum(1 for p in probe_results if p['answer_stable'])
    n_changed = len(probe_results) - n_stable
    stab_rate = n_stable / len(probe_results) if probe_results else 0.0

    print(f"\n{'='*52}")
    print(f"  Counterfactual Probe Results")
    print(f"{'='*52}")
    print(f"  Probed samples   : {len(probe_results)}")
    print(f"  Stable answers   : {n_stable:3d}  ({stab_rate:.0%})  → post-rationalization")
    print(f"  Changed answers  : {n_changed:3d}  ({1-stab_rate:.0%})  → retrieval failure")
    print(f"{'='*52}\n")

    return {
        'n_probed':       len(probe_results),
        'n_stable':       n_stable,
        'n_changed':      n_changed,
        'stability_rate': round(stab_rate, 4),
        'f1_threshold':   f1_threshold,
        'seed':           seed,
        'per_sample':     probe_results,
    }


# ── 4. Analysis helpers ───────────────────────────────────────────────────────

def get_stable_examples(probe_output: dict, n: int = 5) -> list:
    """Most stable answers — strongest post-rationalization evidence."""
    stable = [p for p in probe_output['per_sample'] if p['answer_stable']]
    return sorted(stable, key=lambda p: p['token_f1'], reverse=True)[:n]


def get_changed_examples(probe_output: dict, n: int = 5) -> list:
    """Examples where the answer changed — retrieval failure evidence."""
    changed = [p for p in probe_output['per_sample'] if not p['answer_stable']]
    return sorted(changed, key=lambda p: p['token_f1'])[:n]


def print_probe_example(p: dict, label: str = ""):
    sep = "─" * 88
    print(sep)
    if label:
        print(f"  {label}")
        print(sep)
    print(f"  Question     : {p['question']}")
    print(f"  Original     : {p['original_answer']}")
    print(f"  Counterfact. : {p['counterfactual_answer']}")
    print(f"  Token F1     : {p['token_f1']:.2f}  |  Stable: {p['answer_stable']}")
    print(f"  [RAGAS={p['original_ragas']:.2f}  NLI={p['original_nli']:.2f}]")
    print()


# ── 5. Save / load ────────────────────────────────────────────────────────────

def save_probe_results(
    probe_output: dict,
    path: str = "/scratch/results/faithfulness_counterfactual.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(probe_output, f, indent=2)
    print(f"Saved to {path}")


def load_probe_results(
    path: str = "/scratch/results/faithfulness_counterfactual.json",
) -> dict:
    with open(path) as f:
        return json.load(f)


# ── 6. CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, "/scratch/open-project-m3-mnlp212/code/baseline")
    from generator import Generator

    with open('/scratch/results/faithfulness_baseline.json') as f:
        ragas_data = json.load(f)
    with open('/scratch/results/faithfulness_nli.json') as f:
        nli_data = json.load(f)

    results = ragas_data['results']
    for r, n in zip(results, nli_data['results']):
        r['faithfulness_nli_score'] = n['faithfulness_nli_score']

    rs = np.array([r['faithfulness_score']     for r in results], dtype=float)
    ns = np.array([r['faithfulness_nli_score'] for r in results], dtype=float)

    candidates = [
        i for i in range(len(rs))
        if not np.isnan(rs[i]) and not np.isnan(ns[i])
        and rs[i] == 0 and ns[i] >= 0.5
    ]
    print(f"Probing {len(candidates)} post-rationalization suspects...")

    gen = Generator()
    probe_output = run_counterfactual_probe(results, candidates, gen)
    save_probe_results(probe_output)

    print("Stable examples (post-rationalization):")
    for p in get_stable_examples(probe_output, n=3):
        print_probe_example(p, "STABLE")

    print("Changed examples (retrieval failure):")
    for p in get_changed_examples(probe_output, n=3):
        print_probe_example(p, "CHANGED")
