"""Measure closed-book vs open-book latency per query.

Outputs mean time-to-first-token (TTFT) and tokens/sec for:
  - closed-book: prompt only (no passages), 24 new tokens
  - open-book k=5: prompt + 5 passages (~350 extra tokens), 24 new tokens

Usage:
  python scripts/timing_benchmark.py --model #WORKSPACE/models/Qwen3-8B
  python scripts/timing_benchmark.py --model #WORKSPACE/models/Qwen3-32B --n 30
"""

from __future__ import annotations
import argparse, json, time
import numpy as np

SAMPLE_Q = [
    "What city celebrates the original Oktoberfest?",
    "Which vegetable is traditionally used in moussaka?",
    "Who wrote the novel Nineteen Eighty-Four?",
    "What is the chemical symbol for gold?",
    "Which planet is known as the Red Planet?",
]

SAMPLE_PASSAGE = (
    "This is a sample passage providing context for the question. "
    "It contains approximately seventy tokens of factual information that "
    "would typically be retrieved from a dense retrieval system such as DPR "
    "or bge-large-en-v1.5 in a retrieval-augmented generation pipeline. "
    "The passage is repeated to simulate a realistic open-book prompt length."
) * 2


def build_prompt(tok, q, passages=None, is_q3=False):
    ctx = ""
    if passages:
        ctx = (
            "\n\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
            + "\n\n"
        )
    content = (
        ctx
        + f"Answer the question with a short answer of a few words.\n\nQuestion: {q}"
    )
    if getattr(tok, "chat_template", None):
        kw = dict(tokenize=False, add_generation_prompt=True)
        if is_q3:
            kw["enable_thinking"] = False
        return tok.apply_chat_template([{"role": "user", "content": content}], **kw)
    return content + "\nAnswer:"


def time_generation(model, tok, prompts, n_new=24, warmup=3):
    import torch

    times, ntoks = [], []
    pad_id = tok.pad_token_id or tok.eos_token_id

    for i, prompt in enumerate(prompts):
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
            model.device
        )
        prompt_len = inp["input_ids"].shape[1]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp, do_sample=False, max_new_tokens=n_new, pad_token_id=pad_id
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t0
        gen_len = out.shape[1] - prompt_len
        if i >= warmup:
            times.append(elapsed)
            ntoks.append(gen_len)

    return np.mean(times), np.mean(ntoks), np.mean(ntoks) / np.mean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--n", type=int, default=50, help="total timed iterations (excl. warmup)"
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--k", type=int, default=5, help="passages for open-book")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(
        f"[TIMING] model={args.model} | n={args.n} | warmup={args.warmup} | k_open={args.k}"
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    is_q3 = "qwen3" in args.model.lower().replace("/", "")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[TIMING] model loaded | device={next(model.parameters()).device}")

    import torch

    if torch.cuda.is_available():
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        print(f"[GPU] {result.stdout.strip()}")

    questions = (SAMPLE_Q * ((args.n + args.warmup) // len(SAMPLE_Q) + 1))[
        : args.n + args.warmup
    ]
    passages = [SAMPLE_PASSAGE] * args.k

    cb_prompts = [build_prompt(tok, q, passages=None, is_q3=is_q3) for q in questions]
    ob_prompts = [
        build_prompt(tok, q, passages=passages, is_q3=is_q3) for q in questions
    ]

    cb_len = len(tok(cb_prompts[0])["input_ids"])
    ob_len = len(tok(ob_prompts[0])["input_ids"])
    print(
        f"[TIMING] prompt tokens: closed-book={cb_len}  open-book@k={args.k}: {ob_len}  "
        f"(passage overhead={ob_len - cb_len})"
    )

    print("[TIMING] running closed-book timing...")
    cb_t, cb_ntok, cb_tps = time_generation(model, tok, cb_prompts, warmup=args.warmup)

    print("[TIMING] running open-book timing...")
    ob_t, ob_ntok, ob_tps = time_generation(model, tok, ob_prompts, warmup=args.warmup)

    speedup = ob_t / cb_t
    print("\n========= TIMING RESULTS =========")
    print(
        f"  closed-book  : {cb_t * 1000:.1f} ms/query  |  {cb_tps:.1f} tok/s  |  prompt={cb_len} tok"
    )
    print(
        f"  open-book k={args.k}: {ob_t * 1000:.1f} ms/query  |  {ob_tps:.1f} tok/s  |  prompt={ob_len} tok"
    )
    print(f"  TTFT speedup (open/closed) : {speedup:.2f}x")
    print(f"  (closed-book is {speedup:.2f}x faster than open-book@k={args.k})")
    print("===================================")

    result = {
        "model": args.model,
        "n": args.n,
        "warmup": args.warmup,
        "k": args.k,
        "cb_ms": round(cb_t * 1000, 1),
        "ob_ms": round(ob_t * 1000, 1),
        "cb_tps": round(cb_tps, 1),
        "ob_tps": round(ob_tps, 1),
        "speedup": round(speedup, 2),
        "cb_prompt_tokens": cb_len,
        "ob_prompt_tokens": ob_len,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
