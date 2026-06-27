"""Build results_manifest.csv with per-config frontier AUC and bootstrap CI."""

from __future__ import annotations
import json, os, csv
import numpy as np

from _tables import resolve

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
HERE = os.path.dirname(__file__)
RNG = np.random.default_rng(0)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pc = np.array([r["p_correct"] for r in rows])
    cb = np.array([r["closed_correct"] for r in rows])
    ob = np.array([r.get("open_correct_k5", r.get("open_correct_k1", 0)) for r in rows])
    return pc, cb, ob


def frontier_auc(pc, cb, ob, order=None):
    """Frontier AUC: skip queries with highest `order` (default pc) first."""
    n = len(pc)
    sc = pc if order is None else order
    idx = np.argsort(-sc)
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


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s)
    tp = 0
    acc = 0
    for i in order:
        if y[i]:
            tp += 1
        else:
            acc += tp
    return acc / (n_pos * n_neg)


def metrics(pc, cb, ob):
    gate = frontier_auc(pc, cb, ob)
    oracle = frontier_auc(
        pc, cb, ob, order=(cb - ob)
    )  # skip where retrieval least helps
    rand = (float(cb.mean()) + float(ob.mean())) / 2
    nv = (gate - rand) / (oracle - rand) if oracle > rand else float("nan")
    flip = float(((cb == 1) & (ob == 0)).sum() / max((cb == 1).sum(), 1))
    return dict(
        cb=float(cb.mean()),
        ob1=float("nan"),
        ob5=float(ob.mean()),
        auroc=auroc(cb, pc),
        gate_auc=gate,
        oracle_auc=oracle,
        random_auc=rand,
        norm_value=nv,
        flip_rate=flip,
    )


def bootstrap_ci(pc, cb, ob, B=1000):
    n = len(pc)
    gates, navs = [], []
    rand = (cb.mean() + ob.mean()) / 2
    oracle = frontier_auc(pc, cb, ob, order=(cb - ob))
    for _ in range(B):
        s = RNG.integers(0, n, n)
        g = frontier_auc(pc[s], cb[s], ob[s])
        gates.append(g)
        r = (cb[s].mean() + ob[s].mean()) / 2
        o = frontier_auc(pc[s], cb[s], ob[s], order=(cb[s] - ob[s]))
        navs.append((g - r) / (o - r) if o > r else np.nan)
    return (
        np.percentile(gates, 2.5),
        np.percentile(gates, 97.5),
        np.nanpercentile(navs, 2.5),
        np.nanpercentile(navs, 97.5),
    )


FILES = [
    (
        "triviaqa_rc_qwen1p7b_table.jsonl",
        "TriviaQA-rc",
        "Qwen3-1.7B",
        "1.7B",
        "bge-large",
    ),
    ("triviaqa_rc_table.jsonl", "TriviaQA-rc", "Qwen3-8B", "8B", "bge-large"),
    ("triviaqa_rc_bgesmall_table.jsonl", "TriviaQA-rc", "Qwen3-8B", "8B", "bge-small"),
    (
        "triviaqa_rc_32b_n600_table.jsonl",
        "TriviaQA-rc",
        "Qwen3-32B",
        "32B",
        "bge-large",
    ),
    (
        "triviaqa_rc_qwen35_9b_table.jsonl",
        "TriviaQA-rc",
        "Qwen3.5-9B",
        "9B",
        "bge-large",
    ),
    ("triviaqa_rc_llama_table.jsonl", "TriviaQA-rc", "Llama-3.1-8B", "8B", "bge-large"),
    ("nq_dpr_table.jsonl", "NQ-DPR", "Qwen3-8B", "8B", "DPR"),
    ("nq_dpr_32b_table.jsonl", "NQ-DPR", "Qwen3-32B", "32B", "DPR"),
    ("msmarco_table.jsonl", "MS-MARCO", "Qwen3-8B", "8B", "passage-pool"),
    ("popqa_table.jsonl", "PopQA", "Qwen3-8B", "8B", "n/a (probe)"),
    ("hotpotqa_table.jsonl", "HotpotQA", "Qwen3-8B", "8B", "passage-pool"),
    (
        "triviaqa_rc_shared_table.jsonl",
        "TriviaQA-rc",
        "Qwen3-8B",
        "8B",
        "shared-corpus",
    ),
    ("nq_dpr_shared_table.jsonl", "NQ-DPR", "Qwen3-8B", "8B", "shared-corpus"),
    ("msmarco_shared_table.jsonl", "MS-MARCO", "Qwen3-8B", "8B", "shared-corpus"),
    # 2k-scale tables
    (
        "triviaqa_rc_2k_table.jsonl",
        "TriviaQA-rc",
        "Qwen3-8B",
        "8B",
        "bge-large (n=2000)",
    ),
    ("nq_dpr_2k_table.jsonl", "NQ-DPR", "Qwen3-8B", "8B", "DPR (n=2000)"),
    ("msmarco_2k_table.jsonl", "MS-MARCO", "Qwen3-8B", "8B", "passage-pool (n=2000)"),
]

FIELDS = [
    "dataset",
    "model",
    "model_size",
    "retriever",
    "n",
    "jsonl_table",
    "closed_book_acc",
    "open_book_acc_k5",
    "gate_auroc",
    "frontier_auc_raw",
    "frontier_auc_random",
    "frontier_auc_oracle",
    "norm_value",
    "noise_flip_rate",
    "gate_auc_ci_low",
    "gate_auc_ci_high",
    "norm_value_ci_low",
    "norm_value_ci_high",
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default 1000; use 200 for smoke tests)",
    )
    args = parser.parse_args()
    B = args.bootstrap

    out = os.path.join(HERE, "results_manifest.csv")
    rows_out = []
    for fname, ds, model, size, retr in FILES:
        path = resolve(fname)
        if not os.path.exists(path):
            print(f"MISSING {fname}")
            continue
        pc, cb, ob = load(path)
        m = metrics(pc, cb, ob)
        gl, gh, nl, nh = bootstrap_ci(pc, cb, ob, B=B)
        rows_out.append(
            {
                "dataset": ds,
                "model": model,
                "model_size": size,
                "retriever": retr,
                "n": len(pc),
                "jsonl_table": fname,
                "closed_book_acc": round(m["cb"], 4),
                "open_book_acc_k5": round(m["ob5"], 4),
                "gate_auroc": round(m["auroc"], 4),
                "frontier_auc_raw": round(m["gate_auc"], 4),
                "frontier_auc_random": round(m["random_auc"], 4),
                "frontier_auc_oracle": round(m["oracle_auc"], 4),
                "norm_value": round(m["norm_value"], 4),
                "noise_flip_rate": round(m["flip_rate"], 4),
                "gate_auc_ci_low": round(gl, 4),
                "gate_auc_ci_high": round(gh, 4),
                "norm_value_ci_low": round(nl, 4),
                "norm_value_ci_high": round(nh, 4),
            }
        )
        print(
            f"{ds:<12} {model:<13} {retr:<12} CB={m['cb']:.3f} gateAUC={m['gate_auc']:.3f}"
            f" [{gl:.3f},{gh:.3f}]  NV={m['norm_value']:.3f} [{nl:.3f},{nh:.3f}]"
            f"  flip={m['flip_rate']:.1%}"
        )
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
