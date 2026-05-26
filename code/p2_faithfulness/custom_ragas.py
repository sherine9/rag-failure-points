"""
RAGAS Faithfulness Evaluation — P2 Track (Leïla Benjelloun)
============================================================
Uses Qwen2.5-3B-Instruct as the judge LLM (open-source, no API credits needed).
Deliberately different from the generator (Mistral-7B) to avoid circular evaluation.

Requirements (run once in the pod):
    pip install ragas litellm vllm datasets --break-system-packages
"""


# ── 0. Imports ────────────────────────────────────────────────────────────────
import json
import subprocess
import time
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import EvaluationDataset
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI


# ── 1. vLLM server + judge LLM ────────────────────────────────────────────────
# We use Qwen2.5-3B-Instruct as judge — lighter than Mistral-7B and different
# from the generator to avoid circular self-evaluation bias.
# vLLM serves the model locally as an OpenAI-compatible endpoint, enabling real
# GPU batching across concurrent RAGAS workers (no OpenAI credits needed).
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLLM_PORT   = 8000
VLLM_URL    = f"http://localhost:{VLLM_PORT}/v1"


def start_vllm_server(
    model_name: str = JUDGE_MODEL,
    port: int = VLLM_PORT,
    gpu_memory_utilization: float = 0.4,
    max_model_len: int = 8192,
    timeout: int = 240,
) -> subprocess.Popen:
    """
    Launch a vLLM OpenAI-compatible server in the background.
    gpu_memory_utilization=0.4 leaves room for the generator model on the same GPU.
    Returns the Popen handle so the caller can terminate it when done.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    log_path = f"/tmp/vllm_server_{port}.log"
    print(f"Starting vLLM server ({model_name}) on port {port}...")
    print(f"Server logs: {log_path}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    health_url = f"http://localhost:{port}/health"
    for _ in range(timeout // 2):
        if proc.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"vLLM server process exited (code {proc.returncode}). "
                f"Check logs: {log_path}"
            )
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                print("vLLM server ready.")
                return proc
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    proc.terminate()
    raise RuntimeError(
        f"vLLM server did not become ready within {timeout}s. "
        f"Check logs: {log_path}"
    )


def load_judge_llm(model_name: str = JUDGE_MODEL, port: int = VLLM_PORT):
    """
    Connect to a running vLLM server and return a RAGAS-compatible LLM.
    Call start_vllm_server() first if the server is not already running.
    max_tokens=128 is sufficient for claim extraction and Yes/No verification.
    """
    print(f"Connecting to vLLM at localhost:{port}...")
    langchain_llm = ChatOpenAI(
        model=model_name,
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
        max_tokens=1024,
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    print("Judge LLM ready.")
    return ragas_llm


# ── 2. Build RAGAS dataset from pipeline results ───────────────────────────────
def build_ragas_dataset(results: list[dict]) -> EvaluationDataset:
    """
    Convert the shared pipeline output format to a RAGAS-compatible dataset.

    Args:
        results: list of dicts with keys:
            - question:     str
            - prediction:   str   (the generated answer)
            - gold_answers: list[str]
            - chunks:       list[str]  (retrieved context chunks)

    Returns:
        EvaluationDataset with columns: question, answer, contexts
    """
    hf_dataset = Dataset.from_dict({
        "user_input":         [r["question"]   for r in results],
        "response":           [r["prediction"] for r in results],
        "retrieved_contexts": [r["chunks"]     for r in results],
    })
    return EvaluationDataset.from_hf_dataset(hf_dataset)


# ── 3. Run RAGAS faithfulness evaluation ──────────────────────────────────────
def evaluate_faithfulness(results: list[dict], judge_llm=None) -> dict:
    """
    Compute RAGAS faithfulness score over a list of pipeline results.

    RAGAS Faithfulness pipeline (what happens under the hood):
        1. The judge LLM decomposes each answer into atomic claims.
           e.g. "Einstein was born in Germany in 1879" →
                ["Einstein was born in Germany", "Einstein was born in 1879"]
        2. Each claim is verified against the retrieved chunks via NLI-style prompting.
        3. Faithfulness = (# supported claims) / (# total claims)

    Args:
        results:   list of pipeline result dicts
        judge_llm: RAGAS-compatible LLM (if None, loads Qwen2.5-3B-Instruct)

    Returns:
        dict with faithfulness score and per-sample scores
    """
    if judge_llm is None:
        judge_llm = load_judge_llm()

    dataset = build_ragas_dataset(results)

    # Define the metric with our judge
    faithfulness_metric = Faithfulness(llm=judge_llm)

    print(f"Running RAGAS faithfulness on {len(results)} samples...")
    # max_workers=1: local HuggingFace pipeline is synchronous — running 16 concurrent
    # workers (the default) causes GPU starvation and timeout errors.
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric],
        run_config=RunConfig(max_workers=8, timeout=120),
    )

    # Extract per-sample scores for deeper analysis
    scores_df = result.to_pandas()
    per_sample = scores_df["faithfulness"].tolist()

    output = {
        "faithfulness_mean":   round(float(scores_df["faithfulness"].mean()), 4),
        "faithfulness_std":    round(float(scores_df["faithfulness"].std()), 4),
        "faithfulness_min":    round(float(scores_df["faithfulness"].min()), 4),
        "faithfulness_max":    round(float(scores_df["faithfulness"].max()), 4),
        "per_sample_scores":   per_sample,
        "n_samples":           len(results),
    }

    print(f"\n{'='*45}")
    print(f"  RAGAS Faithfulness Results")
    print(f"{'='*45}")
    print(f"  Mean:  {output['faithfulness_mean']:.4f}")
    print(f"  Std:   {output['faithfulness_std']:.4f}")
    print(f"  Min:   {output['faithfulness_min']:.4f}")
    print(f"  Max:   {output['faithfulness_max']:.4f}")
    print(f"  N:     {output['n_samples']}")
    print(f"{'='*45}\n")

    return output


# ── 4. Add faithfulness scores back to results ────────────────────────────────
def annotate_results_with_faithfulness(
    results: list[dict],
    faithfulness_scores: list[float]
) -> list[dict]:
    """
    Attach per-sample faithfulness scores back to the pipeline result dicts.
    Useful for error analysis and identifying unfaithful examples.

    Args:
        results:              original pipeline results
        faithfulness_scores:  per-sample scores from evaluate_faithfulness()

    Returns:
        results with added 'faithfulness_score' field
    """
    for r, score in zip(results, faithfulness_scores):
        r["faithfulness_score"] = score
    return results


# ── 5. Helper: find the most unfaithful examples ──────────────────────────────
def get_unfaithful_examples(results: list[dict], threshold: float = 0.5, n: int = 10):
    """
    Return examples where faithfulness score is below threshold.
    Useful for Section 6 (Error Analysis) of the notebook.

    Args:
        results:   annotated results (must have 'faithfulness_score')
        threshold: score below which an answer is considered unfaithful
        n:         max number of examples to return

    Returns:
        list of the worst-scoring examples, sorted ascending by score
    """
    unfaithful = [
        r for r in results
        if r.get("faithfulness_score", 1.0) < threshold
    ]
    unfaithful.sort(key=lambda x: x["faithfulness_score"])
    return unfaithful[:n]


# ── 6. Save results ───────────────────────────────────────────────────────────
def save_faithfulness_results(
    metrics: dict,
    results: list[dict],
    path: str = "/scratch/results/faithfulness_baseline.json"
):
    """Save faithfulness evaluation results to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"Saved to {path}")


# ── 7. Main: run on baseline results ──────────────────────────────────────────
if __name__ == "__main__":
    # Load baseline results produced by pipeline.py
    RESULTS_PATH = "/scratch/results/baseline.json"

    print(f"Loading baseline results from {RESULTS_PATH}...")
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    results = data["results"]

    # Load judge once (expensive — reuse across experiments)
    judge_llm = load_judge_llm()

    # Run evaluation
    faithfulness_metrics = evaluate_faithfulness(results, judge_llm=judge_llm)

    # Annotate and save
    results = annotate_results_with_faithfulness(
        results, faithfulness_metrics["per_sample_scores"]
    )
    save_faithfulness_results(
        faithfulness_metrics, results,
        path="/scratch/results/faithfulness_baseline.json"
    )

    # Preview most unfaithful examples
    bad_examples = get_unfaithful_examples(results, threshold=0.5, n=5)
    print(f"\nTop {len(bad_examples)} most unfaithful examples:")
    for ex in bad_examples:
        print(f"\n  Q:     {ex['question']}")
        print(f"  Pred:  {ex['prediction']}")
        print(f"  Score: {ex['faithfulness_score']:.2f}")
        print(f"  Gold:  {ex['gold_answers']}")