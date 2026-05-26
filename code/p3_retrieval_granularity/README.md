# P3 : Retrieval Granularity

Compares four chunking strategies on the shared RAG baseline (NQ Open, contriever-msmarco, Mistral-7B-Instruct).
All components except chunking are identical to the shared baseline.

## Strategies

| Chunker | Description | Corpus size |
|---|---|---|
| `fixed` | 256-word windows, 32-word overlap (matches baseline) | 200k articles |
| `semantic` | Sentence-boundary splits driven by embedding similarity | 200k articles |
| `small_to_big` | Index sentences, return parent paragraphs to generator | 200k articles |
| `proposition` | Mistral-7B decomposes paragraphs into atomic facts | 1k articles* |

*Proposition-level uses a smaller corpus because LLM decomposition is slow at index-build time.

## Setup

From the repo root:
```bash
pip install -r requirements.txt
```

## Running

All commands should be run from `code/p3_retrieval_granularity/`.

**Single chunker:**
```bash
python pipeline_p3.py --chunker semantic
python pipeline_p3.py --chunker semantic --n_wiki 200000 --n_questions 100 --k 5
```

**All experiments (recommended: background + log):**
```bash
mkdir -p /scratch/logs

nohup python run_experiments.py --chunkers fixed semantic small_to_big \
  > /scratch/logs/p3_main.log 2>&1 &

# proposition separately : slow, run last
nohup python pipeline_p3.py --chunker proposition --n_wiki 1000 \
  > /scratch/logs/p3_proposition.log 2>&1 &
```

**Check progress:**
```bash
tail -f /scratch/logs/p3_main.log
```

**Compare results after runs complete:**
```bash
python compare_results.py
```

## Outputs

| Path | Content |
|---|---|
| `/scratch/faiss_index/p3_{chunker}/` | Saved FAISS index (reused on next run) |
| `/scratch/results/p3_{chunker}.json` | Metrics + per-question predictions |

## File structure

```
p3_retrieval_granularity/
├── chunkers/
│   ├── fixed.py          # re-exports baseline chunk_text(256, 32)
│   ├── semantic.py       # similarity-threshold sentence merging
│   ├── small_to_big.py   # (sentence, parent_paragraph) pairs
│   └── proposition.py    # Mistral-7B atomic fact decomposition
├── pipeline_p3.py        # main pipeline — swap chunker via --chunker arg
├── run_experiments.py    # run all 4, print comparison table
└── compare_results.py    # load saved JSONs, print table
```
