"""
Grounding Prompt Sweep — P2 Mitigation 3b (Leïla Benjelloun)
=============================================================
Tests four grounding prompts from least to most restrictive to find the
best balance between faithfulness gain and answer quality.

Key design axis: keeping "Give a SHORT answer" matters — verbose answers
fail NLI's strict entailment far more often than short ones, even when
the content is correct. All prompts except P3 retain the short-answer
instruction.

Prompts (P0 = baseline, P1–P3 = mitigations):
  P0  Baseline         — current generator prompt (weak "ONLY context")
  P1  Soft grounding   — baseline + one extra "no external knowledge" line
  P2  Strict-short     — full grounding rules, short answer kept
  P3  Strict-verbose   — full grounding rules, no short-answer constraint
"""

import json
import os
from tqdm import tqdm


# ── Prompt definitions ────────────────────────────────────────────────────────

PROMPTS = {
    "P1_soft": (
        "Soft grounding — baseline + no external knowledge rule",
        """[INST] You are a precise question answering system.
    Answer the question using ONLY the provided context.
    Give a SHORT answer — a name, date, number, or brief phrase.
    Do NOT write full sentences. Do NOT explain.
    Do NOT use any knowledge that is not written in the context.
    If the answer is not in the context, write "unknown".

    Context:
    {context}

    Question: {question}

    Short answer: [/INST]"""
    ),

    "P2_strict_short": (
        "Strict-short — full grounding rules, short answer kept",
        """[INST] You are a strict question answering system.
    Answer ONLY using information explicitly stated in the passages below.
    Rules:
      - If the answer is NOT explicitly stated in the passages, respond with exactly: I don't know
      - Do NOT use any knowledge from outside the provided passages
      - Do NOT infer, guess, or add facts not directly written in the passages
      - Give a SHORT answer — a name, date, number, or brief phrase

    Passages:
    {context}

    Question: {question}

    Short answer (from passages only, or "I don't know"): [/INST]"""
    ),

    "P3_strict_verbose": (
        "Strict-verbose — full grounding rules, no short-answer constraint",
        """[INST] Answer the question using ONLY facts explicitly written in the passages below.
    If the answer is not in the passages, respond with: I don't know
    Do NOT add any information not directly stated in the passages.

    Passages:
    {context}

    Question: {question}

    Answer: [/INST]"""
    ),
}


# ── Generation helper ─────────────────────────────────────────────────────────

def _build_prompt(template: str, question: str, chunks: list) -> str:
    context = "\n\n".join([f"Passage {i+1}: {c}" for i, c in enumerate(chunks)])
    return template.format(context=context, question=question)


def _generate(generator, question: str, chunks: list, template: str) -> str:
    import torch
    prompt = _build_prompt(template, question, chunks)
    inputs = generator.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(next(generator.model.parameters()).device)
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = generator.tokenizer.decode(generated, skip_special_tokens=True)
    return generator._extract_short_answer(answer).strip()


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_prompt_sweep(results: list, generator, prompt_keys: list = None) -> dict:
    """
    Run all prompts in PROMPTS over the result list.

    Args:
        results:     baseline pipeline results (question + chunks)
        generator:   Generator instance (Mistral-7B)
        prompt_keys: subset of PROMPTS keys to run (default: all)

    Returns:
        dict mapping prompt_key -> list of result dicts with new predictions
    """
    keys = prompt_keys or list(PROMPTS.keys())
    sweep_results = {}

    for key in keys:
        label, template = PROMPTS[key]
        print(f"\nRunning {key}: {label}")
        new_results = []
        for r in tqdm(results, desc=key):
            new_pred = _generate(generator, r["question"], r["chunks"], template)
            new_results.append({
                **r,
                "prediction":          new_pred,
                "original_prediction": r.get("original_prediction", r["prediction"]),
                "prompt_key":          key,
            })
        sweep_results[key] = new_results

    return sweep_results


def save_sweep_results(
    sweep_results: dict,
    base_path: str = "/scratch/results/p2_grounding_sweep",
) -> None:
    os.makedirs(base_path, exist_ok=True)
    for key, results in sweep_results.items():
        path = os.path.join(base_path, f"{key}.json")
        with open(path, "w") as f:
            json.dump({"prompt_key": key, "results": results}, f, indent=2)
        print(f"Saved {key} → {path}")


def load_sweep_results(
    base_path: str = "/scratch/results/p2_grounding_sweep",
) -> dict:
    sweep = {}
    for key in PROMPTS:
        path = os.path.join(base_path, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                sweep[key] = json.load(f)["results"]
    return sweep
