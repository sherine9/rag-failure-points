import re
from collections import Counter
from rouge_score import rouge_scorer as rouge_scorer_lib


def exact_match(prediction, gold_answers):
    """
    Check if prediction exactly matches any gold answer (case insensitive).

    Args:
        prediction: generated answer string
        gold_answers: list of acceptable gold answers

    Returns:
        1 if match found, 0 otherwise

    Example:
        exact_match("1889", ["1889", "in 1889"]) → 1
        exact_match("around 1889", ["1889"]) → 0
    """
    prediction = prediction.strip().lower()
    return int(any(
        prediction == ans.strip().lower()
        for ans in gold_answers
    ))


def rouge_l(prediction, gold_answers):
    """
    ROUGE-L measures longest common subsequence overlap.
    More lenient than exact match : partial credit for close answers.

    Args:
        prediction: generated answer string
        gold_answers: list of acceptable gold answers

    Returns:
        float between 0 and 1

    Example:
        rouge_l("built in 1889", ["1889"]) → ~0.5
        rouge_l("1889", ["1889"]) → 1.0
    """
    scorer = rouge_scorer_lib.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )
    scores = [
        scorer.score(prediction, ans)["rougeL"].fmeasure
        for ans in gold_answers
    ]
    return max(scores)


def token_f1(prediction, gold_answers):
    """
    Measures word overlap between prediction and gold answer.
    More meaningful than EM for verbose predictions.

    Args:
        prediction: generated answer string
        gold_answers: list of acceptable gold answers

    Returns:
        float between 0 and 1

    Example:
        token_f1("built in 1889", ["1889"]) → 0.67
        token_f1("1889", ["1889"]) → 1.0
    """
    def get_tokens(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def f1(pred_tokens, gold_tokens):
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    pred_tokens = get_tokens(prediction)
    scores = [
        f1(pred_tokens, get_tokens(ans))
        for ans in gold_answers
    ]
    return max(scores)


def retrieval_hit_rate(results):
    """
    Measure how often the correct answer appears in retrieved chunks.
    This separates retrieval quality from generation quality.

    Args:
        results: list of dicts with keys chunks and gold_answers

    Returns:
        float between 0 and 1
    """
    hits = 0
    for r in results:
        chunks_text = " ".join(r["chunks"]).lower()
        if any(ans.lower() in chunks_text for ans in r["gold_answers"]):
            hits += 1
    return round(hits / len(results), 4)


def evaluate(results):
    """
    Compute aggregate metrics over all results.

    Args:
        results: list of dicts with keys:
            - question: str
            - prediction: str
            - gold_answers: list of str
            - chunks: list of str

    Returns:
        dict with aggregate metrics:
            - exact_match: float (0-1)
            - rouge_l: float (0-1)
            - token_f1: float (0-1)
            - retrieval_hit_rate: float (0-1)
            - n_samples: int
    """
    if not results:
        raise ValueError("Results list is empty.")

    em_scores = []
    rouge_scores = []
    f1_scores = []

    for r in results:
        em_scores.append(exact_match(r["prediction"], r["gold_answers"]))
        rouge_scores.append(rouge_l(r["prediction"], r["gold_answers"]))
        f1_scores.append(token_f1(r["prediction"], r["gold_answers"]))

    metrics = {
        "exact_match": round(sum(em_scores) / len(em_scores), 4),
        "rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "retrieval_hit_rate": retrieval_hit_rate(results),
        "n_samples": len(results)
    }

    return metrics


def print_metrics(metrics, label="Results"):
    """Pretty print metrics."""
    print(f"\n{'='*40}")
    print(f"{label}")
    print(f"{'='*40}")
    print(f"Exact Match:        {metrics['exact_match']:.4f}")
    print(f"ROUGE-L:            {metrics['rouge_l']:.4f}")
    print(f"Token F1:           {metrics['token_f1']:.4f}")
    print(f"Retrieval Hit Rate: {metrics['retrieval_hit_rate']:.4f}")
    print(f"Samples:            {metrics['n_samples']}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    # quick test
    sample_results = [
        {
            "question": "When was the Eiffel Tower built?",
            "prediction": "1889",
            "gold_answers": ["1889", "in 1889"],
            "chunks": ["The Eiffel Tower was built in 1889."]
        },
        {
            "question": "Who created Python?",
            "prediction": "Guido van Rossum created Python",
            "gold_answers": ["Guido van Rossum"],
            "chunks": ["Python was created by Guido van Rossum."]
        },
        {
            "question": "When was the moon landing?",
            "prediction": "I don't know",
            "gold_answers": ["July 20 1969", "1969"],
            "chunks": ["The moon landing was in 1969."]
        }
    ]

    metrics = evaluate(sample_results)
    print_metrics(metrics, label="Test Results")