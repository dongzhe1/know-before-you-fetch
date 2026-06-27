"""Compute expected per-query latency from measured timing data."""

from __future__ import annotations
import json, os, csv
import numpy as np
from _tables import table_path, data_path
from experiment_registry import get_experiment


def _load_timing():
    """Read measured timing from timing_raw.csv and retriever_benchmark.csv."""
    import pandas as pd

    timing_path = data_path("timing_raw.csv")
    retr_path = data_path("retriever_benchmark.csv")
    result = {}
    if os.path.exists(timing_path):
        df = pd.read_csv(timing_path)
        for model in df["model"].unique():
            md = df[df["model"] == model]
            cb_row = md[(md["mode"] == "closed-book") & (md["k"] == 1)]
            ob1_row = md[(md["mode"] == "open-book") & (md["k"] == 1)]
            ob5_row = md[(md["mode"] == "open-book") & (md["k"] == 5)]
            c_cb = float(cb_row["ttft_ms_mean"].iloc[0]) if len(cb_row) else None
            c_ob_k1 = float(ob1_row["ttft_ms_mean"].iloc[0]) if len(ob1_row) else None
            c_ob_k5 = float(ob5_row["ttft_ms_mean"].iloc[0]) if len(ob5_row) else None
            c_ret_k1, c_ret_k5 = 5.0, 5.0
            if os.path.exists(retr_path):
                rdf = pd.read_csv(retr_path)
                r1 = rdf[(rdf["batch_size"] == 1) & (rdf["k"] == 1)]
                r5 = rdf[(rdf["batch_size"] == 1) & (rdf["k"] == 5)]
                if len(r1):
                    c_ret_k1 = float(r1["total_ms_mean"].iloc[0])
                if len(r5):
                    c_ret_k5 = float(r5["total_ms_mean"].iloc[0])
            result[(model,)] = {
                "c_cb": c_cb,
                "c_ob_k1": c_ob_k1,
                "c_ob_k5": c_ob_k5,
                "c_ret_k1": c_ret_k1,
                "c_ret_k5": c_ret_k5,
            }
    return result


TIMING = _load_timing()

CONFIGS = [
    ("triviaqa_qwen8b_bgelarge", "triviaqa_rc_ksweep_table.jsonl"),
    ("triviaqa_qwen32b_bgelarge", "triviaqa_rc_ksweep_32b_table.jsonl"),
]


def choose_thresholds(p_cal, cb, ob5, ob1):
    """Select binary and graded operating points at match-always accuracy."""
    n = len(cb)
    always = ob5.mean()

    # Binary: find tau where acc >= always
    idx = np.argsort(-p_cal)
    best_k = 0
    for k in range(n + 1):
        skip = np.zeros(n, bool)
        skip[idx[:k]] = True
        if np.where(skip, cb, ob5).mean() >= always:
            best_k = k
    tau_bin = p_cal[idx[best_k - 1]] if best_k > 0 else 1.0
    skip_bin = p_cal >= tau_bin

    # Graded: match-always with best split
    # simplified: use coarse grid
    best_acc, best_ab = 0.0, (0, 0)
    step = max(1, n // 50)
    for a in range(0, n + 1, step):
        for b in range(a, n + 1, step):
            skip_g = np.zeros(n, bool)
            skip_g[idx[:a]] = True
            k1_g = np.zeros(n, bool)
            k1_g[idx[a:b]] = True
            k5_g = np.zeros(n, bool)
            k5_g[idx[b:]] = True
            acc = (cb[skip_g].sum() + ob1[k1_g].sum() + ob5[k5_g].sum()) / n
            if acc >= always and acc > best_acc:
                best_acc = acc
                best_ab = (a, b)

    a, b = best_ab
    skip_g = np.zeros(n, bool)
    skip_g[idx[:a]] = True
    k1_g = np.zeros(n, bool)
    k1_g[idx[a:b]] = True
    k5_g = np.zeros(n, bool)
    k5_g[idx[b:]] = True

    return {
        "binary": {
            "skip_rate": float(skip_bin.mean()),
            "k5_rate": float((~skip_bin).mean()),
            "k1_rate": 0.0,
            "accuracy": float(np.where(skip_bin, cb, ob5).mean()),
        },
        "graded": {
            "skip_rate": float(skip_g.mean()),
            "k1_rate": float(k1_g.mean()),
            "k5_rate": float(k5_g.mean()),
            "accuracy": float(
                (cb[skip_g].sum() + ob1[k1_g].sum() + ob5[k5_g].sum()) / n
            ),
        },
    }


def main():
    out_csv = data_path("cost_metrics.csv")
    w = csv.writer(open(str(out_csv), "w", newline=""))
    w.writerow(
        [
            "dataset",
            "model",
            "policy",
            "skip_rate",
            "k1_rate",
            "k5_rate",
            "retrieval_call_rate",
            "full_context_rate",
            "passage_budget_rate",
            "closed_ms",
            "open_ms_k1",
            "open_ms_k5",
            "retrieval_ms",
            "expected_total_ms",
            "expected_context_tokens",
            "accuracy",
        ]
    )

    for reg_key, ksw_fname in CONFIGS:
        exp = get_experiment(reg_key)
        if exp is None:
            continue
        ksw_path = table_path(ksw_fname)
        rows = [json.loads(l) for l in open(ksw_path)]
        n = len(rows)
        cb = np.array([r["closed_correct"] for r in rows], dtype=float)
        ob5 = np.array([r["open_correct_k5"] for r in rows], dtype=float)
        ob1 = np.array([r["open_correct_k1"] for r in rows], dtype=float)
        p_cal = np.array([float(r.get("p_correct", 0.5)) for r in rows])

        timing = TIMING.get((exp["model"],), {})
        c_cb = timing.get("c_cb")
        c_ob_k1 = timing.get("c_ob_k1")
        c_ob_k5 = timing.get("c_ob_k5", c_cb)
        c_ret_k1 = timing.get("c_ret_k1", 5.0)
        c_ret_k5 = timing.get("c_ret_k5", 5.0)

        rates = choose_thresholds(p_cal, cb, ob5, ob1)

        for policy in ["binary", "graded"]:
            r = rates[policy]
            ret_call_rate = r["k1_rate"] + r["k5_rate"]
            passage_budget = (r["k1_rate"] + 5 * r["k5_rate"]) / 5

            # Expected latency
            if c_cb is not None:
                if c_ob_k1 is not None:
                    exp_ms = (
                        c_cb
                        + r["k1_rate"] * (c_ret_k1 + c_ob_k1)
                        + r["k5_rate"] * (c_ret_k5 + c_ob_k5)
                    )
                else:
                    exp_ms = c_cb + ret_call_rate * (c_ret_k5 + c_ob_k5)
            else:
                exp_ms = "NA"

            ctx_tokens = r["k1_rate"] * 200 + r["k5_rate"] * 650
            c_ret_display = c_ret_k5  # display k=5 retrieval cost as representative

            w.writerow(
                [
                    exp["dataset"],
                    exp["model"],
                    policy,
                    round(r["skip_rate"], 4),
                    round(r["k1_rate"], 4),
                    round(r["k5_rate"], 4),
                    round(ret_call_rate, 4),
                    round(r["k5_rate"], 4),
                    round(passage_budget, 4),
                    c_cb if c_cb else "NA",
                    c_ob_k1 if c_ob_k1 else "NA",
                    c_ob_k5 if c_ob_k5 else "NA",
                    round(c_ret_display, 1),
                    round(exp_ms, 1) if isinstance(exp_ms, (int, float)) else exp_ms,
                    round(ctx_tokens, 0),
                    round(r["accuracy"], 4),
                ]
            )

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
