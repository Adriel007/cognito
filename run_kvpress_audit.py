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

INSTR = ("You are a precise assistant. Using ONLY the context, output the numeric "
         "code/key recorded for the unit. Answer with the number ONLY.")
def make_ctx(it): return INSTR + "\n\nContext:\n" + it.context()
def ask(it):      return f"What number was recorded as the access key/code for unit {it.tag}?"


def build_pipe():
    from transformers import pipeline, BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True)
    return pipeline("kv-press-text-generation", model="mistralai/Mistral-7B-Instruct-v0.3",
                    device_map="auto", model_kwargs=dict(quantization_config=qcfg,
                    dtype=torch.float16, attn_implementation="eager"))


def make_panel(r):
    """Presses that run cleanly on T4/4-bit/eager. Wrappers (AdaKV/ChunkKV/Critical)
    have attention-mode conflicts in this config and are recorded as N/A."""
    from kvpress import (SnapKVPress, ExpectedAttentionPress, PyramidKVPress,
                         ObservedAttentionPress, KnormPress, StreamingLLMPress, TOVAPress)
    return {
        "SnapKV":        SnapKVPress(compression_ratio=r),
        "ExpectedAttn":  ExpectedAttentionPress(compression_ratio=r),
        "PyramidKV":     PyramidKVPress(compression_ratio=r),
        "ObservedAttn":  ObservedAttentionPress(compression_ratio=r),
        "TOVA":          TOVAPress(compression_ratio=r),
        "StreamingLLM":  StreamingLLMPress(compression_ratio=r),
        "Knorm":         KnormPress(compression_ratio=r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n_passages", type=int, default=8)
    ap.add_argument("--n_hard", type=int, default=2)
    ap.add_argument("--passage_words", type=int, default=80)
    ap.add_argument("--ratios", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--out", type=str, default="kvpress_audit.jsonl")
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]

    ds = bench.build_dataset(args.n, args.n_passages, args.n_hard, args.passage_words, seed=0)
    diag = bench.attach_bm25_ranks(ds)
    print(f"[bench] n={len(ds)} | adversarial diagnostic: {json.dumps(diag)}", flush=True)

    pipe = build_pipe()
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
        for method, press in make_panel(r).items():
            run(method, r, press)

    fout.close()
    # Final table
    methods = ["SnapKV","ExpectedAttn","PyramidKV","ObservedAttn","TOVA","StreamingLLM","Knorm"]
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
