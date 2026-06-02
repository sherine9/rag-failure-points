# P3 — Retrieval Granularity

**Owner:** Rania Hatrouhou (342084)  
**Track question:** Does the shape of a text chunk, not just whether the right
passage was retrieved, determine whether the generator can produce a correct answer?

Five chunking strategies are compared on the shared RAG baseline
(contriever-msmarco + FAISS + Mistral-7B-Instruct-v0.2, NQ Open).
Every pipeline component is held constant; only the chunking strategy changes.

---

## Strategies

| Strategy | Retrieval unit | Generation unit | Key idea |
|----------|---------------|-----------------|----------|
| `fixed` | 256-word sliding window (32-word overlap) | Same window | Shared baseline; coarse but complete |
| `semantic` | Coherent topic span | Same span | Splits where contriever cosine similarity between adjacent sentences drops below the 25th-percentile threshold (hard cap: 512 words) |
| `small_to_big` | Individual sentence | Parent paragraph | Decouples retrieval precision from generation context |
| `sentence_only` | Individual sentence | Same sentence | Maximum retrieval precision, minimum context |
| `sentence_window` | Individual sentence | Anchor ± 1 neighbour | Precision retrieval, tight local context |

> **Controlled ablation.** `small_to_big`, `sentence_only`, and `sentence_window`
> share the **exact same FAISS binary index** (10.9M sentence vectors, built once by
> `build_sentence_variants.py`). Only `chunks.pkl` differs. Any EM difference between
> the three is a pure generation effect, retrieval scores and rankings are identical.

Proposition-level chunking (M2, 1k articles) is excluded from M3 scale-up:
it requires one Mistral-7B forward pass per paragraph, making full-scale
indexing (~200k articles) infeasible on a single GPU node.

---

## Key results (n = 2,610 NQ Open validation questions)

| Strategy | EM | Hit Rate | EM\|Hit | Gen. Loss | Mean pred. len |
|----------|-----|---------|---------|----------|---------------|
| Fixed | 0.029 | **0.243** | 0.085 | 22.3% | 10.2w |
| Semantic | 0.046 | 0.223 | 0.180 | 18.3% | 7.3w |
| Small-to-Big | **0.081** | 0.214 | 0.286 | 15.3% | 6.5w |
| Sent-Only | 0.078 | 0.170 | **0.350** | 11.0% | 5.2w |
| Sent-Window | 0.079 | 0.197 | 0.310 | 13.6% | 5.9w |
| Hybrid (rule) | 0.068 | 0.165 | — | — | — |
| Hybrid (score) | 0.069 | 0.165 | — | — | — |
| Oracle | 0.114 | — | — | — | — |

Hybrid routers and oracle evaluated on the held-out eval split (questions 1,305–2,609, n = 1,305).  
EM\|Hit = Exact Match conditioned on retrieval success (generation quality metric).  
Gen. Loss = fraction of all queries where the answer was in the chunks but the generator failed.

**Main findings:**
- Small-to-Big achieves the best overall EM (2.8× Fixed) without improving Hit Rate, the gain is entirely from reduced generation failure.
- The controlled sentence ablation confirms context length is the primary generation lever: EM\|Hit decreases monotonically with context length (10w→39w) across three strategies sharing an identical index.
- Fixed generates verbose predictions (mean 10.2w vs 5.2w for Sent-Only); restricting to short predictions triples EM, revealing chunk density as a third failure mode affecting generation style.
- Both hybrid routers collapse to Sent-Only (~73% of routing decisions), because NQ Open is dominated by who/when/what questions. Neither improves over Small-to-Big.
- Oracle EM = 0.114 (eval split) vs best single strategy 0.074; the 4pp gap sits entirely in 12.6% of questions where routing could help.

---

## Setup

```bash
pip install -r requirements.txt   # from repo root
```

---

## Running

All commands from `code/p3_retrieval_granularity/`.

**Single chunker (any strategy):**
```bash
python pipeline_p3.py --chunker small_to_big
python pipeline_p3.py --chunker semantic --n_wiki 200000 --n_questions 2610 --k 5
```

**All five strategies at scale (n = 2,610):**
```bash
mkdir -p /scratch/logs

# fixed, semantic, small_to_big
nohup python run_experiments.py --chunkers fixed semantic small_to_big \
  > /scratch/logs/p3_main.log 2>&1 &

# sentence variants, build sentence_only and sentence_window indices
# from the small_to_big binary (no re-encoding needed):
python build_sentence_variants.py

# then run sentence_only and sentence_window
nohup python pipeline_p3.py --chunker sentence_only  --n_questions 2610 \
  > /scratch/logs/p3_sentence_only.log 2>&1 &
nohup python pipeline_p3.py --chunker sentence_window --n_questions 2610 \
  > /scratch/logs/p3_sentence_window.log 2>&1 &
```

**Hybrid routers (run after all five strategy results exist):**
```bash
# Rule-based router
nohup python pipeline_p3_hybrid.py --strategy rule  --suffix _n2610 --split 1305 \
  > /scratch/logs/p3_hybrid_rule.log 2>&1 &

# Score-based router 
nohup python pipeline_p3_hybrid.py --strategy score --suffix _n2610 --split 1305 \
  > /scratch/logs/p3_hybrid_score.log 2>&1 &
```

**Monitor progress:**
```bash
tail -f /scratch/logs/p3_main.log
tail -f /scratch/logs/p3_hybrid_score.log
```

**Compare all results:**
```bash
python compare_results.py
```

---

## Outputs

| Path | Content |
|------|---------|
| `/scratch/faiss_index/p3_{chunker}/faiss.index` | FAISS flat-IP index (reused on re-run) |
| `/scratch/faiss_index/p3_{chunker}/chunks.pkl` | Lookup table: vector position → text served to generator |
| `/scratch/results/p3_{chunker}_n2610.json` | Metrics + per-question predictions (n = 2,610) |
| `/scratch/results/p3_hybrid_rule_n2610.json` | Rule router: routing table, counts, metrics |
| `/scratch/results/p3_hybrid_score_n2610.json` | Score router: routing table, counts, metrics |

---

## File structure

```
p3_retrieval_granularity/
├── chunkers/
│   ├── fixed.py               # 256-word sliding window (baseline)
│   ├── semantic.py            # contriever-driven topic-shift detection
│   ├── small_to_big.py        # (sentence, parent_paragraph) pairs
│   ├── sentence_only.py       # sentence as both retrieval and generation unit
│   ├── sentence_window.py     # sentence retrieval, ±1 sentence window
│   └── proposition.py         # Mistral-7B atomic fact decomposition (M2 only)
├── pipeline_p3.py             # pluggable chunker runner (--chunker arg)
├── pipeline_p3_hybrid.py      # rule-based and score-based hybrid routers
│                              # optimised: batch FAISS + single encode pass
├── build_sentence_variants.py # copies small_to_big FAISS binary; writes
│                              # sentence_only/window chunks.pkl (no re-encoding)
├── run_experiments.py         # run fixed/semantic/small_to_big, print table
└── compare_results.py         # load saved JSONs, print comparison table
```

---

## EM|Hit metric

Standard Exact Match conflates retrieval failure with generation failure.
EM|Hit separates them:

```
EM|Hit = (queries answered correctly AND answer was in chunks)
         / (queries where answer was in chunks)
```

Generation loss = Pr[answer in chunks] − Pr[answer in chunks AND correctly extracted]

This is the primary analytical contribution of P3: by fixing retrieval and varying
only chunk shape, we show that chunk density is the main lever controlling whether
Mistral-7B extracts the answer or generates a verbose, semantically-correct but
exact-match-failing response.
