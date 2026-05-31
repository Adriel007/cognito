# Answer-Presence Benchmark for KV-Cache Compression

A controlled, adversarial benchmark that measures whether a KV-cache compression
or RAG context-selection method **preserves the token(s) that contain the answer**
when distractor passages are *topically relevant but answerless*.

## Motivation
KV-compression / context selectors keep tokens by a **relevance or attention**
signal. We show this signal scores *topical relevance*, not *answer presence*:
a passage that echoes the query but contains no answer can outrank the passage
that actually answers it. When that happens the method confidently keeps the
wrong content and evicts the answer.

## Construction (`answer_presence_bench.py`)
Each query has, among `n_passages`:
- **1 GOLD** passage — answer-bearing but **paraphrased** ("the access key … was set
  to *{value}*"), so it shares *few* query terms.
- **`n_hard` HARD-NEGATIVES** — echo the query verbatim ("the authorized access code
  for unit *{tag}* …") but contain **no value**.
- the rest **filler** (query-word-free).

By a superset argument a hard-negative matches a superset of the gold's query
terms, so any lexical/semantic relevance signal ranks a hard-negative above the
gold. **Verified:** BM25 gold mean-rank **3.0**, gold #1 in **0/100**; a dense
cross-encoder reranker (ms-marco-MiniLM-L-6-v2) gives the *same* gold rank 3.0.
A full-context ceiling answers **100%** (only the gold states a value), so any
drop under compression is attributable to **eviction quality**, not the base model.

## Audit results (`run_kvpress_audit.py`, NVIDIA kvpress, Mistral-7B-Instruct-v0.3 4-bit, n=40)

EM% by method × compression (fraction of context KV removed). Ceiling = full context.

| Method (kvpress)            | keep 75% | keep 50% | keep 25% |
|-----------------------------|:--------:|:--------:|:--------:|
| **CEILING (full context)**  | **100.0**|    —     |    —     |
| SnapKV (query-aware)        |   12.5   |   15.0   |   10.0   |
| ExpectedAttention           |   60.0   |    7.5   |    5.0   |
| PyramidKV                   |   55.0   |   17.5   |   10.0   |
| ObservedAttention           |   30.0   |   12.5   |    2.5   |
| TOVA                        |   85.0   |   27.5   |    5.0   |
| StreamingLLM (position)     |    0.0   |    0.0   |    0.0   |
| **Knorm (query-agnostic)**  | **100.0**| **95.0** |    2.5   |

### Findings (honest, nuanced)
1. **Query-aware / attention-based compression fails on answer-presence.** SnapKV —
   the canonical strong method — drops to **12.5%** while *keeping 75%* of the cache
   (ceiling 100%). ExpectedAttention/PyramidKV/ObservedAttention/TOVA all fall to
   ≤27.5% by 50% compression. They attend to the query-echoing hard-negatives and
   evict the paraphrased gold.
2. **Position-based (StreamingLLM) is 0% everywhere** — the gold sits mid-context and
   is never in the sink+recent window.
3. **A query-agnostic magnitude baseline (Knorm) is far more robust** (100% / 95% at
   keep 75% / 50%) because it is *not* misled by query overlap — but it too collapses
   at aggressive budgets (keep 25% → 2.5%).
4. **Takeaway:** standard long-context benchmarks omit *relevant-but-answerless*
   distractors and therefore miss this query-aware fragility. The result challenges
   the assumption that query-aware eviction is strictly better; on this axis the
   "dumb" query-agnostic baseline wins at moderate compression.

### Honest limitations
- v1 is **synthetic/controlled** (NIAH/RULER tradition); real-corpus grounding of
  hard-negatives is future work.
- Single model (Mistral-7B-4bit), single answer type (numeric code), n=40, one seed.
- Wrapper presses **AdaKV / ChunkKV / CriticalKV** (DefensiveKV lineage) had
  attention-mode conflicts on T4/4-bit/eager and were not measured (tooling, not method).
- "answer presence" is operationalized as EM of a unique value; broader answer types
  are future work.

## Reproduce
```bash
python answer_presence_bench.py --n 100 --diagnostic      # prove adversarial property (BM25)
python run_kvpress_audit.py --n 40 --out kvpress_audit.jsonl   # needs: kvpress, a GPU
```
Deterministic by `--seed`. Per-item records (incl. each item's BM25 gold rank) are
written to the JSONL for analysis.
