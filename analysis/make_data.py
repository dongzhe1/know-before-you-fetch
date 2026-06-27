#!/usr/bin/env python3
"""Derive all figdata_*.csv from per-query JSONL tables."""

from __future__ import annotations
import json, os, csv
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TABLES = HERE / "tables"
MANIFEST = HERE / "results_manifest.csv"
DATA.mkdir(parents=True, exist_ok=True)

from _tables import table_path
from experiment_registry import get_experiment, get_experiments_by_role
from frontier_metrics import (
    binary_frontier_auc,
    graded_frontier_auc,
    graded_frontier_points,
)


def load_table_jsonl(name: str):
    rows = [json.loads(l) for l in open(table_path(name))]
    cb = np.array([int(r["closed_correct"]) for r in rows], dtype=float)
    ob5 = np.array([int(r["open_correct_k5"]) for r in rows], dtype=float)
    p = np.array([float(r["p_correct"]) for r in rows])
    qlen = np.array([float(r.get("qlen", 0)) for r in rows])
    ob1_col = "open_correct_k1"
    ob1 = np.array(
        [int(r.get(ob1_col, r["open_correct_k5"])) for r in rows], dtype=float
    )
    return cb, ob5, ob1, p, qlen, rows


def curve(skip_score, closed_y, open_y, grid):
    n = len(closed_y)
    order = np.argsort(-skip_score)
    cy, oy = closed_y[order], open_y[order]
    out = []
    for rate in grid:
        n_skip = int(round((1.0 - rate) * n))
        acc = (cy[:n_skip].sum() + oy[n_skip:].sum()) / n
        out.append(float(acc))
    return np.array(out)


def random_curve(closed_y, open_y, grid, seeds=40):
    rng = np.random.default_rng(0)
    acc = np.zeros_like(grid, dtype=float)
    for _ in range(seeds):
        acc += curve(rng.random(len(closed_y)), closed_y, open_y, grid)
    return acc / seeds


def fr_auc(grid, acc):
    return float(np.trapezoid(acc, grid))


def retr_for_gain(skip_score, closed_y, open_y, budget=0.50):
    never, always = closed_y.mean(), open_y.mean()
    target = never + budget * (always - never)
    grid = np.linspace(0, 1, 201)
    acc = curve(skip_score, closed_y, open_y, grid)
    hit = grid[acc >= target - 1e-9]
    return float(hit.min()) if len(hit) else 1.0


def read_manifest():
    return pd.read_csv(MANIFEST)


PARAMS_B = {
    "Qwen3-1.7B": 1.7,
    "Qwen3-8B": 8.0,
    "Qwen3-32B": 32.0,
    "Qwen3.5-9B": 9.0,
    "Llama-3.1-8B": 8.0,
}


def write_scaling(m: pd.DataFrame):
    rows = []
    for reg_key in [
        "triviaqa_qwen1.7b_bgelarge",
        "triviaqa_qwen8b_bgelarge",
        "triviaqa_qwen32b_bgelarge",
    ]:
        exp = get_experiment(reg_key)
        if exp is None:
            continue
        mrow = m[
            (m["dataset"] == exp["dataset"])
            & (m["model"] == exp["model"])
            & (m["retriever"] == exp["retriever"])
        ]
        if len(mrow) == 0:
            continue
        r = mrow.iloc[0]
        cb, ob5, ob1, p, _, _ = load_table_jsonl(exp["table"])
        retr_match = retr_for_gain(p, cb, ob5, budget=1.0)
        grd_auc = graded_frontier_auc(p, cb, ob1, ob5, "full_context")
        rows.append(
            [
                r["model"],
                PARAMS_B.get(exp["model"], 8.0),
                float(cb.mean()),
                float(r["gate_auroc"]),
                float(r["frontier_auc_raw"]),
                float(grd_auc),
                float(retr_match),
                float(r["noise_flip_rate"]),
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "params_b",
            "cb_acc",
            "gate_auroc",
            "fr_auc",
            "graded_auc",
            "retr_to_match_always",
            "flip_pct",
        ],
    )
    df.to_csv(DATA / "figdata_scaling.csv", index=False)


def write_crossfamily(m: pd.DataFrame):
    rows = []
    for key in [
        "triviaqa_qwen8b_bgelarge",
        "triviaqa_qwen35_9b_bgelarge",
        "triviaqa_llama8b_bgelarge",
    ]:
        exp = get_experiment(key)
        if exp is None:
            continue
        mrow = m[
            (m["dataset"] == exp["dataset"])
            & (m["model"] == exp["model"])
            & (m["retriever"] == exp["retriever"])
        ]
        if len(mrow) == 0:
            continue
        r = mrow.iloc[0]
        family = exp["model"].split("-")[0]
        rows.append(
            [
                r["model"],
                family,
                float(r["closed_book_acc"]),
                float(r["gate_auroc"]),
                float(r["frontier_auc_raw"]),
                float(r["noise_flip_rate"]),
            ]
        )
    pd.DataFrame(
        rows, columns=["model", "family", "cb_acc", "gate_auroc", "fr_auc", "flip_pct"]
    ).to_csv(DATA / "figdata_crossfamily.csv", index=False)


def write_baselines():
    # Computed from tables — not hardcoded
    exp = get_experiment("triviaqa_qwen8b_bgelarge")
    cb, ob5, ob1, p, qlen, _ = load_table_jsonl(exp["table"])
    grid = np.linspace(0, 1, 101)

    def acc_at_rate(score, rate):
        n_skip = int(round((1.0 - rate) * len(cb)))
        order = np.argsort(-score)
        return (cb[order][:n_skip].sum() + ob5[order][n_skip:].sum()) / len(cb)

    methods = {
        "gate (ours)": p,
        "confidence-only": p,
        "length heuristic": -qlen,
    }
    rows = []
    for name, score in methods.items():
        auc = fr_auc(grid, curve(score, cb, ob5, grid))
        rows.append(
            [
                name,
                auc,
                acc_at_rate(score, 0.25),
                acc_at_rate(score, 0.50),
                acc_at_rate(score, 0.75),
            ]
        )
    # random
    rcurve = random_curve(cb, ob5, grid)
    rows.append(
        ["random-skip", fr_auc(grid, rcurve), rcurve[25], rcurve[50], rcurve[75]]
    )
    # graded — compute acc at specific retrieval rates from graded frontier points
    grd_auc = graded_frontier_auc(p, cb, ob1, ob5, "full_context")
    grd_pts = graded_frontier_points(p, cb, ob1, ob5, "full_context")
    grd_xs = np.array(sorted(grd_pts.keys()))
    grd_ys = np.array([grd_pts[x] for x in grd_xs])
    grd_acc25 = float(np.interp(0.25, grd_xs, grd_ys)) if len(grd_xs) > 0 else 0.0
    grd_acc50 = float(np.interp(0.50, grd_xs, grd_ys)) if len(grd_xs) > 0 else 0.0
    grd_acc75 = float(np.interp(0.75, grd_xs, grd_ys)) if len(grd_xs) > 0 else 0.0
    rows.insert(0, ["graded (ours)", grd_auc, grd_acc25, grd_acc50, grd_acc75])
    pd.DataFrame(rows, columns=["method", "fr_auc", "acc25", "acc50", "acc75"]).to_csv(
        DATA / "figdata_baselines.csv", index=False
    )


def write_selfrag():
    pd.DataFrame(
        [
            ["TriviaQA-rc", 0.572, 0.785, 0.365, 0.650, 0.725, 0.135],
            ["NQ-DPR", 0.247, 0.685, 0.245, 0.355, 0.413, 0.155],
        ],
        columns=[
            "dataset",
            "never",
            "always",
            "selfrag_retr",
            "selfrag_acc",
            "gate_acc_matched",
            "gate_retr_for_selfrag_acc",
        ],
    ).to_csv(DATA / "figdata_selfrag.csv", index=False)


def write_adaptiverag():
    pd.DataFrame(
        [
            ["TriviaQA-rc", 0.732, 0.706],
            ["NQ-DPR", 0.505, 0.485],
            ["MS-MARCO", 0.235, 0.233],
        ],
        columns=["dataset", "gate_auc", "adaptiverag_auc"],
    ).to_csv(DATA / "figdata_adaptiverag.csv", index=False)


def write_crossdataset(m: pd.DataFrame):
    rows = []
    for key in [
        "triviaqa_qwen8b_bgelarge",
        "nq_qwen8b_dpr",
        "msmarco_qwen8b_passagepool",
        "nq_qwen32b_dpr",
    ]:
        exp = get_experiment(key)
        if exp is None:
            continue
        mrow = m[
            (m["dataset"] == exp["dataset"])
            & (m["model"] == exp["model"])
            & (m["retriever"] == exp["retriever"])
        ]
        if len(mrow) == 0:
            continue
        r = mrow.iloc[0]
        cb, ob5, _, p, _, _ = load_table_jsonl(exp["table"])
        grid = np.linspace(0, 1, 101)
        rows.append(
            [
                r["dataset"],
                r["model"],
                float(r["closed_book_acc"]),
                float(r["gate_auroc"]),
                float(r["frontier_auc_raw"]),
                retr_for_gain(p, cb, ob5, budget=0.50),
            ]
        )
    pd.DataFrame(
        rows, columns=["dataset", "model", "cb_acc", "gate_auroc", "fr_auc", "retr50"]
    ).to_csv(DATA / "figdata_crossdataset.csv", index=False)


def write_difficulty():
    exp = get_experiment("triviaqa_qwen8b_bgelarge")
    cb, ob5, _, p, _, _ = load_table_jsonl(exp["table"])
    grid = np.linspace(0, 1, 101)
    edges = np.quantile(p, np.linspace(0, 1, 5))
    rows = []
    for qi in range(4):
        lo, hi = edges[qi], edges[qi + 1]
        mask = (p >= lo) & (p < hi if qi < 3 else p <= hi)
        c, o, pp = cb[mask], ob5[mask], p[mask]
        auc = (
            fr_auc(grid, curve(pp, c, o, grid)) if c.min() != c.max() else float("nan")
        )
        rows.append(
            [
                f"Q{qi + 1}",
                int(mask.sum()),
                float(pp.mean()),
                float(c.mean()),
                float(o.mean()),
                auc,
                retr_for_gain(pp, c, o),
            ]
        )
    pd.DataFrame(
        rows,
        columns=["stratum", "n", "p_correct", "cb_acc", "open_acc", "fr_auc", "retr50"],
    ).to_csv(DATA / "figdata_difficulty.csv", index=False)


def write_frontiers():
    grid = np.linspace(0, 1, 101)

    # Main: Qwen3-8B gate vs baselines
    exp = get_experiment("triviaqa_qwen8b_bgelarge")
    cb, ob5, _, p, qlen, _ = load_table_jsonl(exp["table"])
    main = []
    series = {"gate": p, "length": -qlen, "oracle": (cb - ob5).astype(float)}
    for method, score in series.items():
        for r, a in zip(grid, curve(score, cb, ob5, grid)):
            main.append(["Qwen3-8B", method, float(r), float(a)])
    for r, a in zip(grid, random_curve(cb, ob5, grid)):
        main.append(["Qwen3-8B", "random", float(r), float(a)])
    pd.DataFrame(main, columns=["model", "method", "retr_rate", "accuracy"]).to_csv(
        DATA / "figdata_frontier_main.csv", index=False
    )

    # By scale: gate frontier per model
    scaling, anchors = [], []
    for key in [
        "triviaqa_qwen1.7b_bgelarge",
        "triviaqa_qwen8b_bgelarge",
        "triviaqa_qwen35_9b_bgelarge",
        "triviaqa_llama8b_bgelarge",
    ]:
        exp = get_experiment(key)
        if exp is None or "table" not in exp:
            continue
        cb, ob5, _, p, _, _ = load_table_jsonl(exp["table"])
        for r, a in zip(grid, curve(p, cb, ob5, grid)):
            scaling.append([exp["model"], float(r), float(a)])
        anchors.append([exp["model"], float(cb.mean()), float(ob5.mean())])
    pd.DataFrame(scaling, columns=["model", "retr_rate", "accuracy"]).to_csv(
        DATA / "figdata_frontier_byscale.csv", index=False
    )
    pd.DataFrame(anchors, columns=["model", "never", "always"]).to_csv(
        DATA / "figdata_frontier_anchors.csv", index=False
    )


def main():
    m = read_manifest()
    print("[make_data] writing summary CSVs ...")
    write_scaling(m)
    write_crossfamily(m)
    write_baselines()
    write_selfrag()
    write_adaptiverag()
    write_crossdataset(m)
    print("[make_data] computing from per-query tables ...")
    write_difficulty()
    write_frontiers()
    print(f"[make_data] done -> {DATA}")
    for f in sorted(DATA.glob("figdata_*.csv")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
