"""
Thin wrapper around vLLM for P4.

Why vLLM and not code/baseline/generator.py?
    The baseline Generator uses HF transformers with `model.generate(...)`. For
    P4 we need batched inference at two stages per query (query transformation
    + answer generation). vLLM's continuous batching is ~10-30x faster on an
    A100 for this workload, which lets us run HyDE/Query2doc/Rewrite + ablations
    in a single GPU hour.

vLLM must be the FIRST CUDA call in the process — loading the
SentenceTransformer (the retriever) before vLLM triggers
"Cannot re-initialize CUDA in forked subprocess". See rcp_support/README.md.

The reader prompt itself is the one from prompts.build_reader_prompt, which
is shape-identical to code/baseline/generator.py so any score delta we report
is attributable to the *retrieval* stage, not to a different reader prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SamplingConfig:
    """Sampling knobs grouped per usage site so the call sites stay tidy."""

    temperature: float
    max_tokens: int
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None


# Per-method defaults. Justifications:
#   - reader:    deterministic short factoid — matches code/baseline/generator.py
#                exactly (max_new_tokens=100, greedy decoding, EOS stop only).
#                `</s>` IS Mistral's EOS so the explicit stop is a no-op safeguard.
#   - hyde:      0.7 to match Gao et al. 2022 (OpenAI playground default)
#   - query2doc: 1.0 to match Wang et al. 2023 (default temperature)
#   - rewrite:   0.0 — Ma et al. use beam=4 deterministic; we use greedy
READER_SAMPLING    = SamplingConfig(temperature=0.0, max_tokens=100, stop=["</s>"])
HYDE_SAMPLING      = SamplingConfig(temperature=0.7, max_tokens=160, stop=["</s>", "\n\n"])
QUERY2DOC_SAMPLING = SamplingConfig(temperature=1.0, max_tokens=128, stop=["</s>", "\nQuery:"])
REWRITE_SAMPLING   = SamplingConfig(temperature=0.0, max_tokens=40,  stop=["</s>", "\n", "Original:"])


class VLLMRunner:
    """
    Owns the vLLM engine and the tokenizer's chat template.

    Construct ONCE per process, before anything that calls into CUDA (in
    particular before the Retriever / SentenceTransformer).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        # `float16` to match code/baseline/generator.py (torch_dtype=torch.float16)
        # so reader logits are numerically identical across the two implementations.
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.55,
        max_model_len: int = 4096,
        download_dir: Optional[str] = None,
    ):
        # Local import — keeps the module importable on machines without vLLM
        from vllm import LLM, SamplingParams  # noqa: F401

        self._SamplingParams = SamplingParams
        print(f"Loading vLLM generator: {model_name}")
        kwargs = dict(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        if download_dir is not None:
            kwargs["download_dir"] = download_dir
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        print("vLLM ready.")

    # ── Prompt formatting ─────────────────────────────────────────────────────
    def to_chat(self, user_prompt: str) -> str:
        """Wrap a user-turn prompt with Mistral's `[INST] ... [/INST]` tags."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ── Batched generation ────────────────────────────────────────────────────
    def generate(
        self,
        prompts: List[str],
        sampling: SamplingConfig,
        use_chat_template: bool = True,
    ) -> List[str]:
        """Run vLLM on a batch and return the first sample per prompt."""
        if use_chat_template:
            prompts = [self.to_chat(p) for p in prompts]

        params = self._SamplingParams(
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            top_p=sampling.top_p,
            n=sampling.n,
            stop=sampling.stop or [],
        )
        outputs = self.llm.generate(prompts, params)
        # vLLM returns outputs in input order
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_n_samples(
        self,
        prompts: List[str],
        sampling: SamplingConfig,
        n_samples: int,
        use_chat_template: bool = True,
    ) -> List[List[str]]:
        """
        Generate `n_samples` independent completions per prompt.

        Used by the HyDE ablation that averages embeddings of multiple
        hypothetical documents (Gao et al. 2022, Eq. 6).
        """
        if use_chat_template:
            prompts = [self.to_chat(p) for p in prompts]

        params = self._SamplingParams(
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            top_p=sampling.top_p,
            n=n_samples,
            stop=sampling.stop or [],
        )
        outputs = self.llm.generate(prompts, params)
        return [[o.text.strip() for o in out.outputs] for out in outputs]
