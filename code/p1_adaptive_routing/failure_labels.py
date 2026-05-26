"""
Rule-based failure labels for P1 retrieval-stage analysis.

Labels:
    none                  — correct answer
    retrieval_miss        — retrieved but gold not in chunks
    ignored_evidence      — retrieved, gold in chunks, wrong answer
    wrong_skip_decision   — skipped retrieval but always-retrieve would help
    unnecessary_retrieval   — retrieved though no-retrieval was already correct
    bad_generation        — wrong answer not explained by retrieval failure
"""

import sys
from pathlib import Path

_BASELINE = Path(__file__).resolve().parent.parent / "baseline"
sys.path.insert(0, str(_BASELINE))

from evaluate import exact_match, rouge_l, token_f1  # noqa: E402


def gold_in_chunks(chunks, gold_answers):
    if not chunks:
        return False
    text = " ".join(chunks).lower()
    return any(ans.lower() in text for ans in gold_answers)


def per_query_scores(prediction, gold_answers):
    return {
        "exact_match": exact_match(prediction, gold_answers),
        "rouge_l": round(rouge_l(prediction, gold_answers), 4),
        "token_f1": round(token_f1(prediction, gold_answers), 4),
    }


def label_failure(record, refs=None):
    """
    Assign a failure label to one query result.

    Args:
        record: result dict for the mode being labeled
        refs: optional dict with sibling runs for the same question:
              {"no_retrieval": {...}, "always_retrieve": {...}}
    """
    correct = record.get("exact_match", 0) == 1
    if correct:
        return "none"

    retrieved = record.get("retrieval_decision", False)
    chunks = record.get("chunks") or []
    hit = record.get("retrieval_hit", gold_in_chunks(chunks, record["gold_answers"]))
    mode = record.get("mode", "")

    if not retrieved:
        if refs and refs.get("always_retrieve"):
            always = refs["always_retrieve"]
            always_hit = always.get(
                "retrieval_hit",
                gold_in_chunks(always.get("chunks") or [], record["gold_answers"]),
            )
            always_correct = always.get("exact_match", 0) == 1
            if always_hit or always_correct:
                return "wrong_skip_decision"
        return "bad_generation" if mode == "no_retrieval" else "wrong_skip_decision"

    if not hit:
        return "retrieval_miss"

    if refs and refs.get("no_retrieval"):
        no_ret = refs["no_retrieval"]
        if no_ret.get("exact_match", 0) == 1:
            return "unnecessary_retrieval"

    return "ignored_evidence"


def enrich_result(record, refs=None):
    """Add per-query metrics, retrieval_hit, and failure_label."""
    scores = per_query_scores(record["prediction"], record["gold_answers"])
    record.update(scores)

    chunks = record.get("chunks") or []
    record["retrieval_hit"] = gold_in_chunks(chunks, record["gold_answers"])
    record["failure_label"] = label_failure(record, refs=refs)
    return record


def apply_cross_mode_labels(results_by_mode):
    """
    Re-label selective (and optionally other) runs using all three modes.

    Args:
        results_by_mode: dict mode -> list of per-query result dicts (aligned order)

    Returns:
        Updated results_by_mode with failure_label set using cross-mode refs.
    """
    modes = list(results_by_mode.keys())
    n = len(results_by_mode[modes[0]])

    for i in range(n):
        refs = {m: results_by_mode[m][i] for m in modes}
        for mode in modes:
            record = results_by_mode[mode][i]
            sibling_refs = {m: refs[m] for m in modes if m != mode}
            enrich_result(record, refs=sibling_refs)

    return results_by_mode


def failure_label_counts(results):
    """Count failure labels in a result list."""
    counts = {}
    for r in results:
        label = r.get("failure_label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts
