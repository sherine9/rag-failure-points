# Code Structure

RAG pipeline studying failure points across four tracks.
Each track modifies one component of the shared baseline and compares against it.

## Baseline Results

Evaluated on 100 NQ Open validation questions, indexing 200k Wikipedia articles
with fixed-size chunking (256 words, 32 overlap).

| Metric | Score |
|--------|-------|
| Exact Match | 0.0500 |
| ROUGE-L | 0.1350 |
| Token F1 | 0.1308 |
| Retrieval Hit Rate | 0.3500 |

Key finding: retrieval hit rate of 35% suggests chunk boundaries
are the main bottleneck, motivating the four individual tracks below.

## Structure

```
code/
├── baseline/                     # shared baseline (all tracks build on this)
│   ├── chunking.py               # fixed-size chunking (256 tokens, 32 overlap)
│   ├── retriever.py              # contriever-msmarco + FAISS retrieval
│   ├── generator.py              # Mistral-7B-Instruct answer generation
│   ├── evaluate.py               # EM, ROUGE-L, Token F1, Retrieval Hit Rate
│   ├── pipeline.py               # end-to-end pipeline
│   └── test_baseline.py          # unit tests
│
├── p1_adaptive_routing/          # Sherine: no / always / selective retrieval
│   ├── pipeline_p1.py            # three-mode ablation on shared baseline
│   ├── router.py                 # LLM retrieval decision module
│   ├── generator_p1.py           # closed-book generation
│   ├── failure_labels.py         # retrieval-stage failure taxonomy
│   └── compare_results.py        # ablation table
│
├── p2_faithfulness/              # Leïla: faithfulness and context grounding
│
├── p3_retrieval_granularity/     # Rania: chunking strategies comparison
│
└── p4_query_rewriting/           # Sacha: query reformulation strategies
```

## Running the Baseline

```bash
cd code/baseline
python pipeline.py
```

FAISS index saved to `/scratch/faiss_index/baseline/` after first run and reused automatically.
Results saved to `/scratch/results/baseline.json`.

## Running Individual Tracks

Each track has its own pipeline that swaps one component from the baseline.
Results are saved to `/scratch/results/<track_name>.json` for comparison.

### P1 — Adaptive retrieval (Sherine)

```bash
cd code/p1_adaptive_routing
python pipeline_p1.py --mode all --n_questions 50
python compare_results.py
```


## Running Tests

```bash
cd code/baseline
python test_baseline.py                         # all tests
python -m unittest test_baseline.TestChunking  # one class only
```

## Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| NQ Open | `nq_open` (HuggingFace) | Evaluation questions |
| Wikipedia | `wikimedia/wikipedia 20231101.en` | Document corpus |

## Models

| Model | Source | Usage |
|-------|--------|-------|
| contriever-msmarco | `facebook/contriever-msmarco` | Dense retrieval |
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.2` | Answer generation |