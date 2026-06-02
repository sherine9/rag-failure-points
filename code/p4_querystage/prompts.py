"""
Prompt templates and few-shot examples for P4 query-stage methods.

Sources:
    - HyDE        — Gao et al. 2022, "Precise Zero-Shot Dense Retrieval
                    without Relevance Labels" (appendix A.1.6 — DBPedia-Entity prompt
                    is the closest match for NQ Open / Wikipedia retrieval).
    - Query2doc   — Wang et al. 2023, "Query2doc: Query Expansion with
                    Large Language Models" (Figure 1 + Table 11, 4-shot
                    on MS-MARCO; we re-write the few-shots for NQ Open).
    - Rewrite     — Ma et al. 2023, "Query Rewriting for Retrieval-Augmented
                    Large Language Models" (Table 1, open-domain QA prompt
                    simplified to a single rewritten query since NQ is single-hop).
    - Reader      — same shape as code/baseline/generator.py so all methods
                    share an identical answer-generation step.

All prompts target Mistral-7B-Instruct-v0.2 and therefore use the `[INST] ... [/INST]`
chat tags. They are wrapped via `tokenizer.apply_chat_template` inside vllm_runner.py.
"""


# ── HyDE ───────────────────────────────────────────────────────────────────────
# Gao et al. 2022, instruction A.1.6 ("Please write a passage to answer the
# question."). Kept short and zero-shot to match the paper's setup.
HYDE_INSTRUCTION = (
    "Please write a short Wikipedia-style passage (3-5 sentences) that "
    "directly answers the question. Do not refuse; if unsure, write the most "
    "likely passage. Output only the passage, no preamble."
)


def build_hyde_prompt(question: str) -> str:
    """HyDE: ask the LLM for a hypothetical passage that answers the question."""
    return (
        f"{HYDE_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        f"Passage:"
    )


# ── Query2doc ──────────────────────────────────────────────────────────────────
# Wang et al. 2023 use 4 in-context examples drawn from MS-MARCO. Since we
# evaluate on NQ Open / Wikipedia, we rewrite the few-shots with NQ-style
# factoid questions and short Wikipedia-style passages.
QUERY2DOC_INSTRUCTION = "Write a passage that answers the given query:"

QUERY2DOC_FEWSHOTS = [
    {
        "query": "who wrote the book the origin of species",
        "passage": (
            "On the Origin of Species, published on 24 November 1859, is a work "
            "of scientific literature by the English naturalist Charles Darwin. "
            "The book introduced the scientific theory that populations evolve "
            "over generations through a process of natural selection."
        ),
    },
    {
        "query": "what is the capital city of australia",
        "passage": (
            "Canberra is the capital city of Australia. Founded following the "
            "federation of the colonies of Australia as the seat of government "
            "for the new nation, it is Australia's largest inland city and the "
            "eighth-largest city overall. The city is located at the northern "
            "end of the Australian Capital Territory."
        ),
    },
    {
        "query": "when did the second world war end",
        "passage": (
            "World War II ended on 2 September 1945 with the formal surrender "
            "of Japan on board the USS Missouri in Tokyo Bay. The war in Europe "
            "had ended earlier, on 8 May 1945 (V-E Day), following the "
            "unconditional surrender of Nazi Germany."
        ),
    },
    {
        "query": "who painted the ceiling of the sistine chapel",
        "passage": (
            "The ceiling of the Sistine Chapel, painted by Michelangelo between "
            "1508 and 1512, is one of the most renowned artworks of the High "
            "Renaissance. Commissioned by Pope Julius II, the fresco covers "
            "the Sistine Chapel ceiling in the Apostolic Palace in Vatican City."
        ),
    },
]


def build_query2doc_prompt(question: str) -> str:
    """Query2doc: 4-shot prompt to generate a pseudo-document for `question`."""
    parts = [QUERY2DOC_INSTRUCTION, ""]
    for ex in QUERY2DOC_FEWSHOTS:
        parts.append(f"Query: {ex['query']}")
        parts.append(f"Passage: {ex['passage']}")
        parts.append("")
    parts.append(f"Query: {question}")
    parts.append("Passage:")
    return "\n".join(parts)


def compose_query2doc(question: str, pseudo_doc: str, repeat_n: int = 1) -> str:
    """
    Build the final retrieval query for query2doc.

    Wang et al. 2023 distinguish two regimes:
      - Sparse retrieval: repeat the query n=5 times then concat pseudo-doc,
        to keep the BM25 term weights balanced.
      - Dense retrieval: `concat(q, [SEP], d')` — a single copy is enough
        because dense encoders pool the full sequence.

    Since we use a dense bi-encoder (contriever-msmarco), repeat_n=1 is the
    paper-faithful default. The argument is kept so the ablation can sweep it.
    """
    return " ".join([question] * repeat_n) + " [SEP] " + pseudo_doc


# ── Query Rewriting ────────────────────────────────────────────────────────────
# Ma et al. 2023 prompt the LLM with "Think step by step ... provide search
# engine queries ... split with ';' ... end with '**'". We simplify to one
# rewritten query because (i) NQ Open is single-hop and (ii) downstream we
# concatenate the single rewrite — keeping the parsing trivial.
REWRITE_INSTRUCTION = (
    "Rewrite the question below into a single, self-contained search query "
    "that a web search engine could answer. Keep all named entities, dates, and "
    "numbers from the original. Output only the rewritten query, no preamble."
)

REWRITE_FEWSHOTS = [
    {
        "original": "who got the first nobel prize in physics",
        "rewritten": "first Nobel Prize in Physics winner",
    },
    {
        "original": "when is the next deadpool movie being released",
        "rewritten": "next Deadpool movie release date",
    },
    {
        "original": "how many episodes are in chicago fire season 4",
        "rewritten": "number of episodes Chicago Fire season 4",
    },
]


def build_rewrite_prompt(question: str) -> str:
    """LLM-based query rewriter (Ma et al. 2023, simplified single-query variant)."""
    parts = [REWRITE_INSTRUCTION, ""]
    for ex in REWRITE_FEWSHOTS:
        parts.append(f"Original: {ex['original']}")
        parts.append(f"Rewritten: {ex['rewritten']}")
        parts.append("")
    parts.append(f"Original: {question}")
    parts.append("Rewritten:")
    return "\n".join(parts)


# ── M3 NEW: Multi-query Rewrite (Ma et al. 2023 canonical variant) ────────────
# The M2 implementation simplified Ma et al. to a single-query rewrite. The
# canonical paper version produces MULTIPLE queries per question, separated
# by ';', and retrieves chunks from each before merging them for the reader.
# We re-introduce this variant as a tuning axis in M3: sweeping
# rewrite_n_queries ∈ {1, 2, 3} tells us whether the M2 failure mode came
# from the single-query simplification or from over-canonicalization
# (both hypotheses are testable on the resulting numbers).
REWRITE_MULTI_INSTRUCTION = (
    "Rewrite the question below into exactly {n} different search queries that a web "
    "search engine could answer. Each query should target a complementary facet of "
    "the question (e.g. one for the entity, one for the time/place, one for the "
    "relation). Keep all named entities, dates, and numbers from the original. "
    "Separate the queries with ';' and output only the queries, no preamble."
)

# Few-shots for the multi-query rewriter. Hand-crafted in NQ style (not drawn
# from the train set, see prompts.py header for the rationale).
#
# IMPORTANT — each example here lists the MAXIMUM number of queries we support
# (3, matching the largest n_queries value in our ablation grid). At prompt-build
# time, build_rewrite_multi_prompt truncates each example's queries to match
# the requested n_queries. This keeps the demonstrations consistent with the
# instruction's "{n}" — without this, in-context learning would dominate the
# instruction and the LLM would emit 3 queries even when we asked for 2.
REWRITE_MULTI_FEWSHOTS = [
    {
        "original": "who scored the first goal in the 2014 fifa world cup final",
        "rewritten": [
            "first goal 2014 FIFA World Cup final scorer",
            "2014 World Cup final goalscorers",
            "Germany Argentina 2014 World Cup final goals",
        ],
    },
    {
        "original": "when did the band the rolling stones release their first album",
        "rewritten": [
            "Rolling Stones debut album release date",
            "first Rolling Stones studio album year",
            "The Rolling Stones discography 1964",
        ],
    },
]


def build_rewrite_multi_prompt(question: str, n_queries: int = 3) -> str:
    """M3 multi-query rewriter prompt — produces `n_queries` queries separated by ';'.

    The few-shot examples in REWRITE_MULTI_FEWSHOTS are stored as lists of
    3 queries each and TRUNCATED here to `n_queries`. This keeps the
    demonstrations coherent with the instruction's "{n}" parameter — without
    this truncation, the LLM would tend to copy the demonstration count
    (3 queries) regardless of what the instruction asked for.

    Used by query_transforms.transform_rewrite when n_queries > 1. The
    n_queries=1 case is delegated to build_rewrite_prompt above to preserve
    M2 behaviour exactly.
    """
    if n_queries < 2:
        raise ValueError(
            f"build_rewrite_multi_prompt: n_queries must be ≥ 2 "
            f"(got {n_queries}). Use build_rewrite_prompt for n_queries=1."
        )
    if n_queries > len(REWRITE_MULTI_FEWSHOTS[0]["rewritten"]):
        # If we ever extend the ablation grid past n=3, we'd need to add more
        # queries to each example here. Fail loudly rather than silently
        # under-demonstrate the requested count.
        raise ValueError(
            f"build_rewrite_multi_prompt: only {len(REWRITE_MULTI_FEWSHOTS[0]['rewritten'])} "
            f"few-shot queries available per example, but n_queries={n_queries} requested. "
            f"Extend REWRITE_MULTI_FEWSHOTS."
        )

    parts = [REWRITE_MULTI_INSTRUCTION.format(n=n_queries), ""]
    for ex in REWRITE_MULTI_FEWSHOTS:
        # Truncate to n_queries so demonstrations match the instruction.
        rewritten = "; ".join(ex["rewritten"][:n_queries])
        parts.append(f"Original: {ex['original']}")
        parts.append(f"Rewritten: {rewritten}")
        parts.append("")
    parts.append(f"Original: {question}")
    parts.append("Rewritten:")
    return "\n".join(parts)
# ── end M3 NEW ────────────────────────────────────────────────────────────────


# ── Reader (answer generation) ─────────────────────────────────────────────────
# BYTE-FOR-BYTE identical to code/baseline/generator.py:Generator.build_prompt
# so any metric delta is attributable to the retrieval stage alone.
#
# Two subtleties of the baseline prompt we preserve:
#   1. Continuation lines start with 4 literal spaces (artefact of the original
#      f-string being placed inside an indented method body; the leading spaces
#      ARE part of the prompt string sent to the tokenizer).
#   2. The `[INST] ... [/INST]` wrapping is added downstream by
#      VLLMRunner.to_chat() via Mistral's chat template, which expands to
#      `<s>[INST] {content} [/INST]` — i.e. the same shape as the baseline
#      f-string. See VLLMRunner.to_chat docstring.
def build_reader_prompt(question: str, chunks) -> str:
    context = "\n\n".join(f"Passage {i + 1}: {c}" for i, c in enumerate(chunks))
    return (
        "You are a precise question answering system.\n"
        "    Answer the question using ONLY the provided context.\n"
        "    Give a SHORT answer — a name, date, number, or brief phrase.\n"
        "    Do NOT write full sentences. Do NOT explain.\n"
        '    If the answer is not in the context, write "unknown".\n\n'
        f"    Context:\n    {context}\n\n"
        f"    Question: {question}\n\n"
        "    Short answer:"
    )
