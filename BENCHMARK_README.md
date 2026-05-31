# Answer-Presence Benchmark for KV-Cache Compression / RAG Context Selection

A controlled, **adversarial** benchmark that measures whether a KV-cache compression
(or RAG context-selection) method **keeps the token(s) that contain the answer** when
distractor passages are *topically relevant but answerless*. It exposes a failure mode
that standard long-context benchmarks (NIAH/RULER/LongBench) do not contain and
therefore cannot detect.

Artifacts in this repo:
- `answer_presence_bench.py` — dataset generator + EM metric + adversarial-property diagnostic.
- `run_kvpress_audit.py` — audits faithful SOTA presses (NVIDIA **kvpress**) on the benchmark.
- this file — design, methodology, **all results**, findings, limitations, next step.

---

## 1. Motivation & hypothesis

KV-compression / context selectors keep tokens by a **relevance or attention** signal.
We claim that signal scores *topical relevance*, **not answer presence**: a passage that
echoes the query but contains no answer can outrank the passage that actually answers it.
When that happens the method confidently keeps the wrong content and evicts the answer.

**Falsifiable hypothesis.** A full-context ceiling answers correctly (the task is solvable,
the base model is robust to the distractors). Therefore any accuracy drop under compression
is attributable to **eviction quality**. We test, for each method `M` at budget `B`:
`EM(M,B) ≪ EM(full context)` on this benchmark.

---

## 2. Construction (`answer_presence_bench.py`)

Each query has, among `n_passages` (default 8):
- **1 GOLD** — answer-bearing but **paraphrased** ("the access *key* … was set to *{value}*"),
  so it shares *few* query terms.
- **`n_hard` HARD-NEGATIVES** (default 2) — echo the query verbatim ("the authorized access
  *code* for unit *{tag}* …") but contain **no value**.
- rest **filler** (query-word-free). Gold at a random slot (also stresses lost-in-the-middle).

**Adversarial property (verified).** A hard-negative matches a *superset* of the gold's
query terms, both share the rare `{tag}` → any lexical/semantic relevance signal ranks a
hard-negative above the gold:
- **BM25**: gold mean-rank **3.0**, gold is #1 in **0/100** queries (for `answer_kind` ∈ {code, hex, alnum}).
- **Dense reranker** (cross-encoder/ms-marco-MiniLM-L-6-v2): *same* gold mean-rank **3.0**, #1 in **0/24**.
  → the failure is **not** a BM25 artifact; it is fundamental to relevance scoring
  (topical relevance ≠ answer presence).

`answer_kind ∈ {code (4-digit), hex (4 hex chars), alnum (5 alphanumerics)}` varies only the
answer surface form. Deterministic by `--seed`. Difficulty knobs: `n_passages`, `n_hard`,
`passage_words`.

---

## 3. Audit methodology

`run_kvpress_audit.py` runs each press at matched compression ratios (fraction of context
KV **removed**) vs the full-context **ceiling** (no compression), and reports EM. The model
answers a short extraction question; EM = the gold value appears in the output.
Faithful SOTA implementations come from **NVIDIA kvpress**. Setup: Mistral-7B-Instruct-v0.3
(4-bit NF4) and Qwen2.5-3B-Instruct (fp16), single T4, n=40, eager attention for the core panel.

---

## 4. Results

EM% by method × compression. **Ceiling (full context) = 100.0%** in every configuration.

### 4a. Mistral-7B-Instruct (4-bit), answer_kind = **code** (4-digit)
| Method (kvpress)            | keep 75% | keep 50% | keep 25% |
|-----------------------------|:--------:|:--------:|:--------:|
| SnapKV (query-aware)        |   12.5   |   15.0   |   10.0   |
| ExpectedAttention           |   60.0   |    7.5   |    5.0   |
| PyramidKV                   |   55.0   |   17.5   |   10.0   |
| ObservedAttention           |   30.0   |   12.5   |    2.5   |
| TOVA                        |   85.0   |   27.5   |    5.0   |
| StreamingLLM (position)     |    0.0   |    0.0   |    0.0   |
| **Knorm (query-agnostic)**  | **100.0**| **95.0** |    2.5   |

### 4b. Mistral-7B-Instruct (4-bit), answer_kind = **hex**
| Method                      | keep 75% | keep 50% |
|-----------------------------|:--------:|:--------:|
| SnapKV                      |   55.0   |   17.5   |
| ExpectedAttention           |   92.5   |   35.0   |
| PyramidKV                   |   77.5   |   32.5   |
| ObservedAttention           |   62.5   |   25.0   |
| TOVA                        |   97.5   |   55.0   |
| StreamingLLM                |    0.0   |    0.0   |
| **Knorm**                   | **100.0**| **100.0**|

### 4c. Qwen2.5-3B-Instruct (fp16), answer_kind = **code**
| Method                      | keep 75% | keep 50% |
|-----------------------------|:--------:|:--------:|
| SnapKV                      |   20.0   |    5.0   |
| ExpectedAttention           |   37.5   |   10.0   |
| PyramidKV                   |   17.5   |    0.0   |
| ObservedAttention           |   15.0   |    2.5   |
| TOVA                        |   37.5   |   15.0   |
| StreamingLLM                |    0.0   |    0.0   |
| **Knorm**                   | **100.0**| **90.0** |

### 4d. Wrapper presses (DefensiveKV / ChunkKV lineage), Qwen-3B fp16, keep 50%
| Wrapper                     | attn=sdpa | attn=eager | note |
|-----------------------------|:---------:|:----------:|------|
| AdaKV(SnapKV) (DefensiveKV lineage) | **0.0** | fails | runs under sdpa, scores 0 |
| ChunkKV(SnapKV) (semantic chunks)   | **0.0** | fails | runs under sdpa, scores 0 |
| CriticalKV(SnapKV)          | not measurable | not measurable | kvpress×Qwen `head_dim` bug |

---

## 5. Findings (honest, nuanced)

1. **The gap is real and consistent.** At moderate-to-aggressive compression, every
   attention/query-aware press falls **far below the 100% ceiling**, across **two model
   families** (Mistral-7B, Qwen-3B) and **two answer surface forms** (code, hex). The
   pre-registered hypothesis holds.
2. **Severity is surface-form dependent (important nuance).** Plain numeric codes are
   *catastrophic* for query-aware presses (SnapKV **12.5%** at keep-75%); a more salient
   hex token is easier (SnapKV **55%** at keep-75%) — but still collapses by keep-50%
   (**17.5%**). So the *effect* is robust; its *magnitude* depends on how salient the answer
   token is to the attention mechanism.
3. **Position-based (StreamingLLM) = 0% everywhere** — the gold sits mid-context, never in
   the sink+recent window.
4. **A query-AGNOSTIC magnitude baseline (Knorm) is the standout.** It is **100% at keep-75%
   across both models and both surface forms**, and 90–100% at keep-50% — because it is *not*
   misled by query overlap. It collapses only at aggressive keep-25% (2.5%). This is the
   method-relevant result: **on the answer-presence axis, the trivial query-agnostic
   criterion beats every query-aware SOTA method at moderate compression.**
5. **Adaptive / semantic-chunk wrappers don't help.** AdaKV (DefensiveKV lineage) and ChunkKV
   (semantic-unit) both score **0%** (keep-50%, Qwen) — ChunkKV preserving "whole semantic
   units" doesn't help when the unit it preserves is the *query-matching* (answerless) one.
6. **Takeaway.** Standard long-context benchmarks omit *relevant-but-answerless* distractors
   and therefore cannot see this query-aware fragility. The result challenges the prevailing
   assumption that query-aware eviction is strictly better (cf. KVzip's claim that query-aware
   compression is fragile in multi-query settings); on this axis a query-agnostic baseline wins.

---

## 6. Next step — proposed method (specified, NOT yet validated)

The Knorm result points the direction: the robust signal is **query-agnostic**. The natural
method to attack the regime where even Knorm fails (aggressive compression) is
**Draft-Attention Eviction (DAE)** — *answer-conditioned*, training-free:
1. Generate a short draft answer (k≈8 tokens) with the full context.
2. Score each context token by the attention it receives **from the generated answer tokens**
   (not from the query) — this targets *answer-bearing* tokens directly.
3. Keep top-B by that score (re-rotating survivors' RoPE positions — validated separately),
   then regenerate from the compressed cache.
Falsifiable claim: DAE > query-aware presses (and ≥ Knorm at aggressive budgets) on this
benchmark, at the cost of k extra prefill-decode steps. **Risk/related work:** draft-based
answer-aware selection exists (LAQ, SpecKV; LookaheadKV trains to predict it) — DAE's
differentiator is training-free answer-attention; a lit-check vs LAQ/SpecKV is required
before claiming novelty. Status: **proposed, not implemented** (deliberately not rushed).

---

## 7. Honest limitations
- v1 **synthetic / controlled** (NIAH/RULER tradition); real-corpus grounding of
  hard-negatives is future work.
- 2 models, 2 surface forms, **n=40, single seed (0)** per cell; multi-seed CIs are future work.
- One answer *task* (value retrieval) with EM of a unique token; broader answer types future work.
- Wrapper presses **CriticalKV** not measurable here (kvpress×Qwen `head_dim` incompatibility);
  AdaKV/ChunkKV measured only under sdpa (eager mode conflicts) — tooling, not method.
- "answer presence" operationalized as EM of a unique value; semantic-answer tasks future work.

---

## 8. Reproduce
```bash
# adversarial property (no GPU)
python answer_presence_bench.py --n 100 --diagnostic --answer_kind code

# audit (needs GPU + kvpress, bitsandbytes for 4-bit)
python run_kvpress_audit.py --model mistralai/Mistral-7B-Instruct-v0.3 --answer_kind code \
       --ratios 0.25,0.5,0.75 --n 40 --out audit_mistral_code.jsonl
python run_kvpress_audit.py --model Qwen/Qwen2.5-3B-Instruct --fp16 --answer_kind code \
       --ratios 0.25,0.5 --n 40 --out audit_qwen_code.jsonl
```
Deterministic by `--seed`. Per-item records (incl. each item's BM25 gold rank) → JSONL.
