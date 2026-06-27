"""TARG-style prefix-logit gating baseline, evaluated under our protocol.

TARG (Wang et al., 2025, arXiv:2511.09803) decides retrieval from a no-context draft's
prefix-logit uncertainty, thresholding one of:
  - mean token entropy
  - top-1/top-2 logit margin   (most robust in their paper)
  - small-N variance across a few stochastic prefixes
We recompute these signals on the SAME questions as an existing per-query table, then compare
each signal's accuracy-vs-retrieval-rate frontier AUC against our calibrated gate's seq_logprob.

Because frontier AUC is rank-based, calibrating a single monotone signal does not change it; this
script therefore isolates *ranking quality* (seq_logprob vs TARG signals). Our calibration's
payoff is the graded budget, reported elsewhere.

Usage:
  python scripts/targ_baseline.py --table logs/triviaqa_rc_table.jsonl \
      --model #WORKSPACE/models/Qwen3-8B --n 600 --var_k 4
"""

from __future__ import annotations
import argparse, json
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def frontier_auc(skip_score, cb, ob):
    """skip_score high => skip first. Returns area under acc-vs-retrieval-rate curve."""
    n = len(cb)
    idx = np.argsort(-skip_score)
    pts = {1.0: float(ob.mean()), 0.0: float(cb.mean())}
    use_cb = np.zeros(n, bool)
    for k in range(1, n + 1):
        use_cb[idx[k - 1]] = True
        acc = (cb[use_cb].sum() + ob[~use_cb].sum()) / n
        rr = float((~use_cb).sum()) / n
        if acc > pts.get(rr, 0):
            pts[rr] = acc
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    return float(_trapz(ys, xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table",
        required=True,
        help="existing *_table.jsonl (reuse open/closed outcomes)",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument(
        "--var_k", type=int, default=4, help="stochastic prefixes for small-N variance"
    )
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument(
        "--dump",
        default="",
        help="dump per-query feature table (for the feature ablation)",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.table) if l.strip()][: args.n]
    print(f"[TARG] table={args.table} | n={len(rows)} | model={args.model}")
    cb = np.array([r["closed_correct"] for r in rows])
    ob = np.array([r.get("open_correct_k5", r.get("open_correct_k1", 0)) for r in rows])
    seqlp = np.array([r["seq_logprob"] for r in rows])  # our signal, from table

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    is_q3 = "qwen3" in args.model.lower().replace("/", "")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    def chat(q):
        c = f"Answer the question with a short answer of a few words, using only your own knowledge.\n\nQuestion: {q}"
        if getattr(tok, "chat_template", None):
            kw = dict(tokenize=False, add_generation_prompt=True)
            if is_q3:
                kw["enable_thinking"] = False
            return tok.apply_chat_template([{"role": "user", "content": c}], **kw)
        return c + "\nAnswer:"

    prompts = [chat(r["question"]) for r in rows]
    entropy, margin, min_lp, mean_lp = [], [], [], []
    for i in range(0, len(prompts), args.batch):
        ch = prompts[i : i + args.batch]
        inp = tok(
            ch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                do_sample=False,
                max_new_tokens=args.max_new,
                pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        plen = inp["input_ids"].shape[1]
        gen = out.sequences[:, plen:]
        logits = torch.stack(out.scores, 1).float()  # [B, T, V]
        logp = torch.log_softmax(logits, -1)
        probs = logp.exp()
        ent = -(probs * logp).sum(-1)  # [B, T] token entropy
        top2 = torch.topk(logp, 2, dim=-1).values  # [B, T, 2]
        marg = top2[..., 0] - top2[..., 1]  # [B, T] top1-top2 margin (logprob)
        for b in range(len(ch)):
            m = gen[b] != tok.pad_token_id
            tlen = int(m.sum().item()) or 1
            entropy.append(float(ent[b, :tlen].mean()))
            margin.append(float(marg[b, :tlen].mean()))
            # chosen-token logprobs -> mean (≈ seq_logprob sanity check) and min (weakest token)
            tok_lp = logp[b, torch.arange(tlen), gen[b, :tlen]]
            mean_lp.append(float(tok_lp.mean()))
            min_lp.append(float(tok_lp.min()))
        del inp, logits, logp, probs
    entropy = np.array(entropy)
    margin = np.array(margin)
    min_lp = np.array(min_lp)
    mean_lp = np.array(mean_lp)

    # small-N variance: variance of mean token entropy across var_k stochastic prefixes
    variance = np.zeros(len(rows))
    if args.var_k > 1:
        ent_samples = [[] for _ in rows]
        for i in range(0, len(prompts), args.batch):
            ch = prompts[i : i + args.batch]
            inp = tok(
                ch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    do_sample=True,
                    temperature=args.temp,
                    top_p=0.95,
                    num_return_sequences=args.var_k,
                    max_new_tokens=args.max_new,
                    pad_token_id=tok.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            plen = inp["input_ids"].shape[1]
            seqs = out.sequences[:, plen:]
            logp = torch.log_softmax(torch.stack(out.scores, 1).float(), -1)
            probs = logp.exp()
            ent = -(probs * logp).sum(-1)  # [B*var_k, T]
            for b in range(len(ch)):
                for s in range(args.var_k):
                    r = b * args.var_k + s
                    mm = seqs[r] != tok.pad_token_id
                    tlen = int(mm.sum().item()) or 1
                    ent_samples[i + b].append(float(ent[r, :tlen].mean()))
            del inp, logp, probs
        variance = np.array([np.var(e) if e else 0.0 for e in ent_samples])

    # frontier AUC: skip the most-confident first.
    # high seq_logprob => confident => skip.  high entropy/variance => unsure => retrieve (skip = -signal).
    # high margin => confident => skip.
    signals = {
        "ours: seq_logprob": seqlp,
        "TARG: top1-top2 margin": margin,
        "TARG: mean entropy": -entropy,
        "TARG: small-N variance": -variance,
    }
    from math import isnan

    print(
        f"\n=== frontier AUC (skip most-confident first) "
        f"| anchors: never={cb.mean():.3f}@0%, always={ob.mean():.3f}@100% ==="
    )
    rand = (cb.mean() + ob.mean()) / 2
    print(f"  {'signal':<26} {'frontierAUC':>11}")
    results = {}
    for name, s in signals.items():
        a = frontier_auc(s, cb, ob)
        results[name] = a
        print(f"  {name:<26} {a:>11.3f}")
    print(f"  {'random-skip':<26} {rand:>11.3f}")

    best_targ = max(results[k] for k in results if k.startswith("TARG"))
    ours = results["ours: seq_logprob"]
    print(f"\n--- VERDICT ---")
    print(
        f"  ours seq_logprob = {ours:.3f} | best TARG signal = {best_targ:.3f} | "
        f"Δ = {ours - best_targ:+.3f}"
    )
    if ours >= best_targ:
        print("  Our ranking signal >= best TARG prefix signal on the binary frontier.")
    else:
        print(
            "  A TARG prefix signal ranks better; our contribution rests on the graded budget"
            " + calibration + regime analysis (frontier AUC is rank-only)."
        )
    print(
        json.dumps(
            {
                "table": args.table,
                "model": args.model,
                "n": len(rows),
                **{
                    k.replace(": ", "_").replace(" ", "_").replace("-", ""): round(v, 4)
                    for k, v in results.items()
                },
            }
        )
    )

    if args.dump:
        with open(args.dump, "w") as f:
            for i, r in enumerate(rows):
                f.write(
                    json.dumps(
                        {
                            "question": r["question"],
                            "closed_correct": int(cb[i]),
                            "open_correct_k5": int(ob[i]),
                            "seq_logprob": float(seqlp[i]),
                            "mean_token_logprob": float(mean_lp[i]),
                            "min_token_logprob": float(min_lp[i]),
                            "mean_entropy": float(entropy[i]),
                            "top1_top2_margin": float(margin[i]),
                            "prefix_variance": float(variance[i]),
                        }
                    )
                    + "\n"
                )
        print(f"[TARG] dumped feature table -> {args.dump}")


if __name__ == "__main__":
    main()
