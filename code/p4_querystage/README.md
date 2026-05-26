# P4 — Query-Stage Rewriting (Sacha)

Controlled ablation on the shared baseline: **does transforming the query
before retrieval close the lexical/semantic gap?**

## Methods

| Method      | Idea                                                                 | Reference            |
|-------------|----------------------------------------------------------------------|----------------------|
| `baseline`  | Raw question → contriever → FAISS (= shared baseline behaviour)      | —                    |
| `hyde`      | LLM writes a hypothetical Wikipedia passage; its embedding is the query | Gao et al. 2022   |
| `query2doc` | 4-shot LLM pseudo-doc concatenated with the original query           | Wang et al. 2023     |
| `rewrite`   | LLM rewrites the question into a search-engine-friendly query        | Ma et al. 2023       |

All other components match `code/baseline/`: contriever-msmarco + the FAISS
index at `/scratch/faiss_index/baseline`, Mistral-7B-Instruct-v0.2 (via vLLM
for batched inference), NQ Open validation slice, `evaluate.py` metrics.

## Run on cluster

```bash
# Requires the baseline FAISS index at /scratch/faiss_index/baseline
cd code/p4_querystage

# Single method
python pipeline_p4.py --method hyde     --n_questions 100
python pipeline_p4.py --method query2doc --n_questions 100 --q2d_repeat 5
python pipeline_p4.py --method rewrite   --n_questions 100

# Full ablation
python pipeline_p4.py --method all --n_questions 100

# HyDE with N-sample averaging (Eq. 6 in Gao et al. 2022)
python pipeline_p4.py --method hyde --hyde_n_samples 4 --hyde_temperature 0.7
```

## Outputs (under `/scratch/results/p4/`)

| File                       | Description                                  |
|----------------------------|----------------------------------------------|
| `p4_baseline.json`         | vanilla RAG run                              |
| `p4_hyde.json`             | HyDE run (per-query incl. pseudo-doc)        |
| `p4_query2doc.json`        | Query2doc run (per-query incl. pseudo-doc)   |
| `p4_rewrite.json`          | Rewrite-Retrieve-Read run (per-query rewrite)|
| `p4_ablation.json`         | Combined metrics from `--method all`         |

Each per-query record exposes `intermediate` (the pseudo-doc or the rewritten
query) so the case-study cells in the notebook can compare what each method
actually sends to the retriever.

## Files

- `prompts.py` — all prompt templates + few-shot examples (HyDE, Query2doc, Rewrite, reader)
- `vllm_runner.py` — vLLM wrapper; **must be initialised before the Retriever**
- `query_transforms.py` — implements the four transforms, returns either text or pre-computed embeddings
- `retrieval_utils.py` — batched FAISS search (by text or by embedding) on top of `code/baseline/retriever.py`
- `pipeline_p4.py` — main runner; `--method {baseline,hyde,query2doc,rewrite,all}`
- `bias_analysis.py` — feature-score correlation diagnostic à la Goyal et al. 2026 (decorrelation vs variance-based bias reduction)
- `chunking.py` — token-level chunker (kept from earlier P4 work; not used at eval time since the baseline FAISS index already exists)

## Notebook

`notebooks/Sacha_Frankhauser_346608.ipynb` imports the modules above and runs
the comparison + ablations. It does NOT re-implement chunking, retrieval, or
evaluation: those come from `code/baseline/`.
