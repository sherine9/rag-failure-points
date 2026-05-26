# P1 — Adaptive / Selective Retrieval (Sherine)

Controlled ablation on the shared baseline: **when is retrieval useful vs harmful?**

## Modes

| Mode | Retrieval | Generation |
|------|-----------|------------|
| `no_retrieval` | Never | Closed-book Mistral prompt |
| `always_retrieve` | Always top-k | Baseline RAG prompt |
| `selective` | LLM router (+ optional score gate) | RAG or closed-book per decision |

Everything else matches `code/baseline/`: Contriever + FAISS index, Mistral-7B-Instruct, NQ Open validation slice, `evaluate.py` metrics.

## Run on cluster

```bash
# Requires baseline index at /scratch/faiss_index/baseline
cd code/p1_adaptive_routing
python pipeline_p1.py --mode all --n_questions 50
python compare_results.py
```

Single mode:

```bash
python pipeline_p1.py --mode selective --n_questions 50
python pipeline_p1.py --mode selective --score_threshold 0.35
```

## Outputs

| File | Description |
|------|-------------|
| `/scratch/results/p1_no_retrieval.json` | No-retrieval run |
| `/scratch/results/p1_always_retrieve.json` | Always-retrieve (= baseline behaviour) |
| `/scratch/results/p1_selective.json` | Selective routing run |
| `/scratch/results/p1_ablation.json` | Combined metrics + failure counts |

Each per-query record includes: `retrieval_decision`, `chunks`, `prediction`, per-query EM/ROUGE/F1, `retrieval_hit`, `failure_label`, and router fields for selective mode.

## Failure labels

- `none` — correct
- `retrieval_miss` — retrieved but gold not in chunks
- `ignored_evidence` — gold in chunks, wrong answer
- `wrong_skip_decision` — skipped retrieval but always-retrieve would help
- `unnecessary_retrieval` — retrieved though no-retrieval was correct
- `bad_generation` — wrong without a clear retrieval cause

Cross-mode labels use all three runs (`--mode all`).
