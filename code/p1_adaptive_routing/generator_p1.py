"""
P1 extensions to the shared baseline Generator.

Adds closed-book generation (no retrieval) without modifying code/baseline/generator.py.
"""

import sys
from pathlib import Path

_BASELINE = Path(__file__).resolve().parent.parent / "baseline"
sys.path.insert(0, str(_BASELINE))

from generator import Generator  # noqa: E402
import torch  # noqa: E402


class GeneratorP1(Generator):
    """Baseline generator plus closed-book and router prompts for P1."""

    def build_closed_book_prompt(self, question):
        return f"""[INST] You are a precise question answering system.
Answer the question from your own knowledge.
Give a SHORT answer — a name, date, number, or brief phrase.
Do NOT write full sentences. Do NOT explain.

Question: {question}

Short answer: [/INST]"""

    def build_router_prompt(self, question):
        return f"""[INST] You are a retrieval router for a question answering system.
Decide whether looking up external documents is needed to answer the question accurately.

Answer YES if the question asks about specific facts, entities, dates, places, or detailed knowledge.
Answer NO only if the question is simple and likely answerable from common general knowledge.

Question: {question}

Reply with exactly one word: YES or NO.
[/INST]"""

    def _generate_from_prompt(self, prompt, max_new_tokens=32):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def generate_closed_book(self, question, max_new_tokens=100):
        """Answer without retrieved context (no-retrieval baseline)."""
        prompt = self.build_closed_book_prompt(question)
        answer = self._generate_from_prompt(prompt, max_new_tokens=max_new_tokens)
        return self._extract_short_answer(answer)

    def route_retrieval(self, question, max_new_tokens=8):
        """
        Ask the generator whether retrieval is needed.

        Returns:
            (should_retrieve: bool, raw_response: str)
        """
        prompt = self.build_router_prompt(question)
        raw = self._generate_from_prompt(prompt, max_new_tokens=max_new_tokens)
        upper = raw.upper()
        if "YES" in upper and "NO" not in upper:
            return True, raw
        if "NO" in upper and "YES" not in upper:
            return False, raw
        # fallback: retrieve when ambiguous (safer for factual QA)
        return True, raw
