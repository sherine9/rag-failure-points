# P4 — Query-Stage Transformations (Sacha)

**Question:** does transforming the query *before* retrieval close the
lexical/semantic gap in the shared RAG pipeline?

Every other stage is held **byte-for-byte identical to `code/baseline/`**
(contriever-msmarco + the FAISS index at `/scratch/faiss_index/baseline`,
`Mistral-7B-Instruct-v0.2` reader, NQ Open validation slice, `evaluate.py`
metrics). Only the text/vector sent to the retriever changes, so any metric
delta is attributable to the query stage alone.

## Methods

| Method      | What it sends to the retriever                                                       | Reference         |
|-------------|--------------------------------------------------------------------------------------|-------------------|
| `baseline`  | the raw question (= shared baseline behaviour)                                       | —                 |
| `hyde`      | the LLM writes *N* hypothetical passages; their contriever embeddings are **averaged** (Eq. 6), optionally with the question embedding (Eq. 8); the averaged vector **replaces** the query | Gao et al. 2022   |
| `query2doc` | a 4-shot LLM pseudo-document is **concatenated** with the original question (`q [SEP] d'`) | Wang et al. 2023  |
| `rewrite`   | the LLM rewrites the question into one (single-query) or several (multi-query) search-engine queries; multi-query retrieves per query then merges + dedups | Ma et al. 2023    |

Decoding temperatures are paper-faithful (not tuned): reader T=0 (greedy,
matches baseline), HyDE T=0.7 (Gao §4.1), Query2doc T=1.0 (Wang §3.1),
Rewrite T=1.0.

## Architecture

`pipeline_p4.py` is the **orchestrator**. Each method follows the same
per-query flow, and only the first step differs:

```
question
  → transform_queries()   [LLM]    # the ONLY method-specific step
  → retrieve()            [FAISS]
  → generate_answers()    [LLM reader, prompt identical to baseline]
  → evaluate()                      # EM / ROUGE-L / Token-F1 / Hit-Rate
  → per-query JSON record (incl. `intermediate` = the pseudo-doc / rewrite)
```

| File                 | Role |
|----------------------|------|
| **`pipeline_p4.py`** | **Orchestrator.** `run_method()` runs one method end-to-end; `run_m3_ablation()` sweeps the frozen `METHODS_M3` grid (Stage 1); `run_m3_test()` runs `BEST_CONFIGS_M3` across seeds (Stage 3). 3-phase CLI (`--phase ablation|test|single`). Also contains the prompt-overflow fix (below). |
| `prompts.py`         | All prompt templates: HyDE, Query2doc (4-shot), Rewrite (single + multi-query), and the **reader prompt copied byte-for-byte from `baseline/generator.py`**. |
| `vllm_runner.py`     | `VLLMRunner`, a batched wrapper over vLLM (~10–30× faster than HF on the A100). Per-method `SamplingConfig`s and `set_seed()` for multi-seed runs without reloading the model. **Must be the first CUDA call in the process** (before the Retriever / SentenceTransformer). |
| `query_transforms.py`| The three transforms, each returning a `QueryTransformResult` (the `intermediate` text + exactly one of: a retrieval *text*, a precomputed *embedding* for HyDE-averaging, or precomputed *chunks* for multi-query rewrite). |
| `retrieval_utils.py` | Batched FAISS helpers (`retrieve_by_text`, `retrieve_by_embedding`, `encode_texts`) built on the shared `baseline/retriever.py`. |
| `bias_analysis.py`   | Spearman-ρ surface-feature diagnostic (brevity / literal overlap / answer position / query repetition) adapting Goyal et al. 2026. |
| `compute_m3_stats.py`| **Offline** stats over the full N=2610 held-out records → three small committable JSONs (`p4_stats.json`, `p4_cases.json`, `p4_sample_level.json`): McNemar tests, paired bootstrap CIs, bias ρ, EM\|Hit, qtype/abstention breakdowns, and the qualitative case sets. No GPU. |
| `condense_results.py`| Downsamples the raw per-query JSONs (200 uniform records, metrics preserved) so they fit in the repo. |

### Design invariants (what keeps the comparison fair)

- **vLLM initialised before the retriever** — else CUDA forking crashes.
- **Reader prompt identical to baseline** — any delta is the *retrieval* stage,
  not a different reader.
- **Pre-committed grids** (`METHODS_M3`, `BEST_CONFIGS_M3`) frozen in code
  before looking at numbers — no researcher degrees of freedom in selection.
- **Token-precise per-chunk prompt-overflow fix** (`generate_answers`): the
  baseline silently truncates the *end* of the prompt (cutting the question)
  on the ~1–5 % of NQ Open questions whose chunks tokenise past 4096; we
  truncate the *context chunks* instead, keeping the question intact, applied
  uniformly across methods.

## Run on the cluster

Requires the baseline FAISS index at `/scratch/faiss_index/baseline`
(build it once via `code/baseline/pipeline.py`). Results are written under
`/scratch/results/p4/m3/`.

```bash
cd code/p4_querystage

# Stage 1 — tuning ablation: frozen 11-config grid, N=1000, single seed
python pipeline_p4.py --phase ablation --n_questions 1000

# Stage 3 — held-out multi-seed test: BEST_CONFIGS_M3, N=2610 from index 1000
python pipeline_p4.py --phase test --n_questions 2610 --question_offset 1000 --seeds 0 1 2

# (debug) single M2-style run of one method
python pipeline_p4.py --phase single --method hyde --n_questions 100 \
    --hyde_n_samples 8 --hyde_temperature 0.7
```

(Stage 2, the multi-seed tiebreak on the top HyDE/Query2doc candidates at
N=1000, is run by temporarily pointing `BEST_CONFIGS_M3` at those candidates
and calling `--phase test --n_questions 1000 --question_offset 0`.)

### Outputs (under `/scratch/results/p4/m3/`)

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `p4_<label>.json`             | one per ablation config (e.g. `p4_hyde_n8_wq.json`), full per-query records + metrics |
| `p4_m3_ablation.json`         | combined metrics for the Stage-1 grid                    |
| `p4_<label>_seed<s>.json`     | one per (best config, seed) in the held-out test         |
| `p4_m3_test.json`             | mean ± std per metric over seeds (headline table)        |

Each per-query record exposes `intermediate` (the pseudo-doc or the rewritten
query/queries) so the notebook can show exactly what each method sent to the
retriever.

## Post-processing → committable artefacts

```bash
# 1) downsample raw JSONs so they fit in the repo (metrics preserved)
python condense_results.py --src /scratch/results/p4/m3 \
    --dst <repo>/results/p4 --sample 200

# 2) compute statistics over the FULL N=2610 records (best_* files only)
python compute_m3_stats.py --src /scratch/results/p4/m3 \
    --dst <repo>/results/p4   # -> p4_dryrun/{p4_stats,p4_cases,p4_sample_level}.json
```

## Notebook

`notebooks/Sacha_Frankhauser_346608.ipynb` is a **pure results reader**: it
loads the committed JSONs above and produces every table/plot on CPU in under a
minute. It does not re-run generation, retrieval, or evaluation — those come
from the cluster runs and from `code/baseline/`. Model/dataset loading is
demonstrated but optional and short (gated by `LOAD_*` flags), so the notebook
never times out at grading.
