"""
answer_presence_bench.py — Adversarial Answer-Presence benchmark for KV-cache
compression / RAG context selection.

WHY THIS EXISTS
---------------
KV-cache compression and RAG context selectors decide *which* tokens/passages to
keep under a budget, using a relevance or attention signal. We found empirically
(Cognito, 2026) that these signals score TOPICAL RELEVANCE, not ANSWER PRESENCE —
so a passage that echoes the query but does not contain the answer can outrank the
passage that actually answers it. When that happens, the method confidently keeps
the wrong content and the answer is evicted.

This benchmark makes that failure *measurable and controlled*:

  - GOLD passage: contains the answer, but PARAPHRASED so it shares few query
    terms ("the access key ... was set to {value}", not "authorized access code").
  - HARD-NEGATIVE passages: echo the query verbatim ("the authorized access code
    for unit {tag} ...") but contain NO answer value.
  - FILLER passages: neutral, query-word-free.

By construction (superset argument: a hard-negative matches a SUPERSET of the
gold's query terms, both share the rare {tag}), any lexical/semantic relevance
signal ranks a hard-negative ABOVE the gold. We verified this holds for BM25 AND
for a dense cross-encoder reranker (ms-marco-MiniLM-L-6-v2): gold mean rank ~3.0,
never #1. So eviction-by-relevance keeps hard-negatives and drops the gold.

THE FALSIFIABLE CLAIM
---------------------
A NoPaging / full-context ceiling answers correctly (only the gold states a value
for the unit) → the task is solvable and the base model is robust to the
distractors. THEREFORE any accuracy drop under a compression method is attributable
to *eviction quality*, not to the base model. Hypothesis under test for each method
M at budget B:
        EM(M, B)  <<  EM(full context)   on this benchmark
i.e. current KV-compression methods (even SOTA: SnapKV, AdaKV, PyramidKV,
DefensiveKV, ChunkKV) fail to preserve answer-bearing content when relevance is
misleading. The benchmark is the instrument; the audit (run_kvpress_audit.py)
is the measurement.

DESIGN KNOBS (difficulty)
-------------------------
  n_passages    : passages per query (more → harder retention)
  n_hard        : hard-negatives per query (more → relevance more misleading)
  passage_words : approx words per passage (longer → tighter budget)
  gold can sit at any slot (random) → also stresses lost-in-the-middle.

REPRODUCIBILITY: deterministic given `seed`. Each item carries gold_idx/hard_idxs
and (optionally) the BM25 gold rank so the adversarial property is auditable.

This is a v1 synthetic, controlled benchmark (in the tradition of NIAH/RULER).
Real-corpus grounding of the hard-negatives is a documented v2 extension.
"""
from __future__ import annotations
import argparse
import json
import random
import re
import string
from dataclasses import dataclass, asdict, field

# Filler sentences contain NONE of the query terms (access/code/unit/authorized/key)
# so BM25/relevance ranking is driven only by the gold vs hard-negative phrasing.
FILLER_SENTENCES = [
    "The grass is green and the sky is blue.",
    "Water flows downhill and fire burns upward.",
    "Birds fly through clouds during the daytime.",
    "Mountains rise above valleys and rivers.",
    "Stars appear at night when the sun has set.",
    "Trees produce oxygen and consume carbon dioxide.",
    "Salt dissolves in water but oil does not.",
    "Sound travels slower than light in the air.",
    "Iron rusts when exposed to oxygen and moisture.",
    "Glass shatters when struck with sufficient force.",
]

# GOLD: answer-bearing, paraphrased (synonym 'access key', avoids 'authorized'/'code').
GOLD_TEMPLATE = ("After the unit {tag} compliance review, the access key for that "
                 "unit was set to {value} and stored securely.")

# HARD-NEGATIVES: echo the query phrase ("authorized access code for unit {tag}")
# but state NO numeric value → high relevance, zero answer content.
HARD_TEMPLATES = [
    "The authorized access code for unit {tag} is reviewed annually by the compliance board and then archived.",
    "Personnel assigned to unit {tag} must rotate their authorized access code on the schedule set by policy.",
    "Documentation describing the authorized access code for unit {tag} is maintained in the central registry.",
    "An audit of the authorized access code workflow for unit {tag} found no irregularities during this cycle.",
]

QUERY_TEMPLATE = "What is the authorized access code for unit {tag}?"


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _filler(n_words: int, rng: random.Random) -> str:
    out = []
    while len(_words(" ".join(out))) < n_words:
        out.append(rng.choice(FILLER_SENTENCES))
    # trim to approx n_words
    text = " ".join(out)
    toks = text.split()
    return " ".join(toks[: max(n_words, 1)])


@dataclass
class BenchItem:
    id: int
    query: str
    gold_value: str
    tag: str
    passages: list[str]
    gold_idx: int
    hard_idxs: list[int]
    gold_bm25_rank: int | None = None  # filled by attach_bm25_ranks()

    def context(self) -> str:
        return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(self.passages))


def _gen_value(kind: str, rng: random.Random) -> str:
    """Answer surface form. Lets us test whether robustness (e.g. Knorm) depends on
    the answer being a NUMBER specifically, or generalizes across surface forms."""
    if kind == "code":   # 4-digit decimal
        return f"{rng.randint(1000, 9999)}"
    if kind == "hex":    # 4 hex chars (uppercase)
        return "".join(rng.choice("0123456789ABCDEF") for _ in range(4))
    if kind == "alnum":  # mixed letters+digits token
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))
    raise ValueError(f"unknown answer_kind: {kind}")


def build_dataset(n_queries: int = 100, n_passages: int = 8, n_hard: int = 2,
                  passage_words: int = 80, seed: int = 0,
                  answer_kind: str = "code") -> list[BenchItem]:
    """Deterministic adversarial dataset. Gold answer-bearing but low-overlap;
    n_hard query-echoing answerless distractors; rest filler; gold at random slot.
    `answer_kind` ∈ {code, hex, alnum} varies only the answer surface form."""
    assert 1 + n_hard <= n_passages, "need room for gold + hard-negatives"
    rng = random.Random(seed)
    items: list[BenchItem] = []
    for qi in range(n_queries):
        tag = f"{rng.randint(100, 999)}{qi:03d}"
        value = _gen_value(answer_kind, rng)
        query = QUERY_TEMPLATE.format(tag=tag)
        gold = GOLD_TEMPLATE.format(tag=tag, value=value)
        gold = (gold + " " + _filler(max(0, passage_words - len(gold.split())), rng)).strip()

        slots = list(range(n_passages))
        rng.shuffle(slots)
        gold_idx = slots[0]
        hard_idxs = sorted(slots[1: 1 + n_hard])

        passages: list[str] = [""] * n_passages
        passages[gold_idx] = gold
        for j, hidx in enumerate(hard_idxs):
            h = HARD_TEMPLATES[j % len(HARD_TEMPLATES)].format(tag=tag)
            passages[hidx] = (h + " " + _filler(max(0, passage_words - len(h.split())), rng)).strip()
        for i in range(n_passages):
            if not passages[i]:
                passages[i] = _filler(passage_words, rng)

        items.append(BenchItem(id=qi, query=query, gold_value=value, tag=tag,
                               passages=passages, gold_idx=gold_idx, hard_idxs=hard_idxs))
    return items


# ── Metric ────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def exact_match(prediction: str, gold_value: str) -> bool:
    """EM = the gold value (case-insensitive) appears as a token in the output.
    Works for code/hex/alnum surface forms (normalize lowercases both sides)."""
    return gold_value.lower() in normalize(prediction).split()


# ── Adversarial-property diagnostic (optional, needs rank_bm25) ─────────────
def attach_bm25_ranks(items: list[BenchItem]) -> dict:
    """Rank each item's gold passage by BM25 vs the query; report the distribution.
    Proves the dataset is adversarial: gold should rank below #1 in ~all items."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return {"error": "rank_bm25 not installed; pip install rank_bm25"}
    ranks = []
    for it in items:
        bm = BM25Okapi([_words(p) for p in it.passages])
        scores = bm.get_scores(_words(it.query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        rank = order.index(it.gold_idx) + 1
        it.gold_bm25_rank = rank
        ranks.append(rank)
    n = len(ranks)
    return {
        "n": n,
        "gold_mean_rank": round(sum(ranks) / n, 3),
        "gold_is_top1": sum(1 for r in ranks if r == 1),
        "gold_in_top2": sum(1 for r in ranks if r <= 2),
        "adversarial": sum(1 for r in ranks if r > 1) >= 0.9 * n,
    }


# ── Prompt builder (model-agnostic default; audit can override) ─────────────
def build_prompt(item: BenchItem, instruction: str | None = None) -> str:
    instr = instruction or ("Answer the question using ONLY the context. "
                            "Give the numeric code only.")
    return (f"{instr}\n\nContext:\n{item.context()}\n\n"
            f"Question: {item.query}\nAnswer:")


def _main():
    ap = argparse.ArgumentParser(description="Adversarial answer-presence benchmark")
    ap.add_argument("--dump", type=str, default=None, help="write dataset JSONL here")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_passages", type=int, default=8)
    ap.add_argument("--n_hard", type=int, default=2)
    ap.add_argument("--passage_words", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer_kind", type=str, default="code", choices=["code", "hex", "alnum"])
    ap.add_argument("--diagnostic", action="store_true", help="report BM25 gold-rank")
    args = ap.parse_args()

    ds = build_dataset(args.n, args.n_passages, args.n_hard, args.passage_words,
                       args.seed, args.answer_kind)
    if args.diagnostic:
        diag = attach_bm25_ranks(ds)
        print("[adversarial-property diagnostic]", json.dumps(diag, indent=2))
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for it in ds:
                rec = asdict(it)
                rec["prompt"] = build_prompt(it)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[dump] wrote {len(ds)} items to {args.dump}")
    if not args.dump and not args.diagnostic:
        it = ds[0]
        print("Example item:\n", build_prompt(it))
        print("\ngold_value:", it.gold_value, "| gold_idx:", it.gold_idx,
              "| hard_idxs:", it.hard_idxs)


if __name__ == "__main__":
    _main()
