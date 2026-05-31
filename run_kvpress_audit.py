"""
run_kvpress_audit.py — Audit faithful SOTA KV-compression presses (NVIDIA kvpress)
on the adversarial answer-presence benchmark.

Hypothesis: query-aware / attention-based presses fail on the benchmark because
they attend to the query-echoing hard-negatives and evict the (paraphrased)
answer-bearing gold; the full-context ceiling answers correctly, so any drop is
attributable to eviction quality.

Runs each press at matched compression ratios vs the full-context ceiling on
Mistral-7B-Instruct (4-bit). Writes per-item JSONL + a press×ratio EM table.

Usage:  python run_kvpress_audit.py --n 40 --out kvpress_audit.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
import torch
import kvpress  # noqa: F401 — registers the "kv-press-text-generation" pipeline task
import answer_presence_bench as bench

INSTR = ("You are a precise assistant. Using ONLY the context, output the access "
         "code/key recorded for the unit. Answer with that value ONLY.")
def make_ctx(it): return INSTR + "\n\nContext:\n" + it.context()
def ask(it):      return f"What value was recorded as the access key/code for unit {it.tag}?"


def build_pipe(model: str, attn: str = "eager", four_bit: bool = True):
    from transformers import pipeline, BitsAndBytesConfig
    mk = dict(dtype=torch.float16, attn_implementation=attn)
    if four_bit:
        mk["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    return pipeline("kv-press-text-generation", model=model, device_map="auto", model_kwargs=mk)


def make_panel(r, include_wrappers: bool = False):
    """Core presses run cleanly on eager. Wrappers (AdaKV/ChunkKV/Critical — the
    DefensiveKV/ChunkKV lineage) are attempted only when include_wrappers=True
    (they need specific attn modes); failures are caught per-press."""
    from kvpress import (SnapKVPress, ExpectedAttentionPress, PyramidKVPress,
                         ObservedAttentionPress, KnormPress, StreamingLLMPress, TOVAPress)
    P = {
        "SnapKV":        SnapKVPress(compression_ratio=r),
        "ExpectedAttn":  ExpectedAttentionPress(compression_ratio=r),
        "PyramidKV":     PyramidKVPress(compression_ratio=r),
        "ObservedAttn":  ObservedAttentionPress(compression_ratio=r),
        "TOVA":          TOVAPress(compression_ratio=r),
        "StreamingLLM":  StreamingLLMPress(compression_ratio=r),
        "Knorm":         KnormPress(compression_ratio=r),
    }
    if include_wrappers:
        from kvpress import AdaKVPress, ChunkKVPress, CriticalKVPress
        def add(n, f):
            try: P[n] = f()
            except Exception as e: print(f"  [wrapper skip {n}] {str(e)[:70]}", flush=True)
        add("AdaKV-SnapKV",   lambda: AdaKVPress(SnapKVPress(compression_ratio=r)))
        add("ChunkKV-SnapKV", lambda: ChunkKVPress(SnapKVPress(compression_ratio=r)))
        add("CriticalKV-Snap",lambda: CriticalKVPress(SnapKVPress(compression_ratio=r)))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n_passages", type=int, default=8)
    ap.add_argument("--n_hard", type=int, default=2)
    ap.add_argument("--passage_words", type=int, default=80)
    ap.add_argument("--ratios", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer_kind", type=str, default="code", choices=["code","hex","alnum"])
    ap.add_argument("--attn", type=str, default="eager")
    ap.add_argument("--fp16", action="store_true", help="load in fp16 (no 4-bit) — for small models")
    ap.add_argument("--wrappers", action="store_true", help="also try AdaKV/ChunkKV/CriticalKV")
    ap.add_argument("--out", type=str, default="kvpress_audit.jsonl")
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]

    ds = bench.build_dataset(args.n, args.n_passages, args.n_hard, args.passage_words,
                             seed=args.seed, answer_kind=args.answer_kind)
    diag = bench.attach_bm25_ranks(ds)
    print(f"[bench] model={args.model} seed={args.seed} kind={args.answer_kind} n={len(ds)} "
          f"| adversarial: {json.dumps(diag)}", flush=True)

    pipe = build_pipe(args.model, attn=args.attn, four_bit=not args.fp16)
    print(f"[pipe] built | attn={pipe.model.config._attn_implementation}", flush=True)
    fout = open(args.out, "w", encoding="utf-8")
    table = {}   # (method, ratio) -> [hits, n]

    def run(method, ratio, press):
        hits = n = 0
        for it in ds:
            try:
                out = pipe(make_ctx(it), question=ask(it), press=press, max_new_tokens=10)
                ans = str(out["answer"] if isinstance(out, dict) else out)
                hit = bench.exact_match(ans, it.gold_value)
            except Exception as e:
                ans, hit = f"ERROR:{type(e).__name__}", False
            hits += int(hit); n += 1
            fout.write(json.dumps({"method": method, "ratio": ratio, "id": it.id,
                                   "gold": it.gold_value, "hit": bool(hit),
                                   "gold_bm25_rank": it.gold_bm25_rank, "ans": ans[:60]}) + "\n")
        fout.flush()
        table[(method, ratio)] = (hits, n)
        print(f"  [{method:<13} r={ratio}] EM={100*hits/n:5.1f}%  ({hits}/{n})", flush=True)

    # Ceiling (no compression) — one pass
    print("\n=== CEILING (full context, no compression) ===", flush=True)
    from kvpress import KnormPress
    run("CEILING", 0.0, KnormPress(compression_ratio=0.0))

    for r in ratios:
        print(f"\n=== compression_ratio = {r} (keep {100*(1-r):.0f}%) ===", flush=True)
        for method, press in make_panel(r, include_wrappers=args.wrappers).items():
            run(method, r, press)

    fout.close()
    # Final table — methods in first-seen order
    methods = []
    for (m, r) in table:
        if m != "CEILING" and m not in methods:
            methods.append(m)
    print("\n" + "#"*78 + "\n# ANSWER-PRESENCE AUDIT — EM% (rows=method, cols=compression)\n" + "#"*78, flush=True)
    ceil = table[("CEILING",0.0)]; print(f"  {'CEILING (full ctx)':<16} {100*ceil[0]/ceil[1]:5.1f}%", flush=True)
    header = "  " + f"{'method':<16}" + "".join(f" r={r:<5}" for r in ratios)
    print(header, flush=True)
    for m in methods:
        cells = []
        for r in ratios:
            h, n = table.get((m, r), (0, 0))
            cells.append(f"{100*h/n:5.1f}%" if n else "  -  ")
        print(f"  {m:<16}" + " ".join(f"{c:<7}" for c in cells), flush=True)
    print(f"\nfalsifier: ceiling≈100% but every press ≪ ceiling ⇒ SOTA KV-compression "
          f"fails to preserve answer-bearing content under misleading relevance.", flush=True)


if __name__ == "__main__":
    main()
