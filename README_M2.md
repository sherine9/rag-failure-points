# Where Does RAG Lose Information?
**CS-552 MNLP: Team MNLP212**  
Rania Hatrouhou · Leïla Benjelloun · Sacha Frankhauser · Sherine Adrou Bakkali

## Overview
A controlled study of four RAG failure modes. Each track modifies exactly
one stage of a shared baseline to isolate its causal contribution.

| Track | Owner | Stage modified |
|-------|-------|----------------|
| P1 | Sherine | Retrieval decision (no / always / selective) |
| P2 | Leïla | Faithfulness (RAGAS, NLI, counterfactual probe) |
| P3 | Rania | Chunking granularity (fixed / semantic / small-to-big / proposition) |
| P4 | Sacha | Query rewriting (HyDE / Query2doc / Rewrite) |

**Shared baseline:** contriever-msmarco + FAISS + Mistral-7B-Instruct-v0.2,
200k Wikipedia articles, evaluated on NQ Open (EM=0.05, Hit=0.35).

## Repo structure
```
code/
├── baseline/                 # shared pipeline (chunking, retrieval, generation, eval)
├── p1_adaptive_routing/      # Sherine
├── p2_faithfulness/          # Leïla
├── p3_retrieval_granularity/ # Rania
└── p4_querystage/            # Sacha
notebooks/                    # one notebook per team member
report/                       # milestone reports
```



## Running
```bash
# Baseline
python code/baseline/pipeline.py
```

Results are saved to `/scratch/results/` and reloaded by each notebook.

## Models & data

| Component | HuggingFace ID |
|-----------|----------------|
| Retriever | `facebook/contriever-msmarco` |
| Generator | `mistralai/Mistral-7B-Instruct-v0.2` |
| Corpus | `wikimedia/wikipedia 20231101.en` |
| Eval set | `nq_open` (validation split) |
