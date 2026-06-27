"""2k scale-up summary: binary and graded AUC under all three cost axes."""

from __future__ import annotations
import json, os, csv
import numpy as np

from _tables import table_path, data_path
from frontier_metrics import (
    binary_frontier_auc,
    graded_frontier_auc,
    bootstrap_ci,
)

TABLES_2K = [
    ("triviaqa_rc_2k_table.jsonl", "TriviaQA-rc"),
    ("nq_dpr_2k_table.jsonl", "NQ-DPR"),
    ("msmarco_2k_table.jsonl", "MS-MARCO"),
]

AXES = ["full_context", "retrieval_call", "passage_budget"]


def auroc(y, s):
    y = np.asarray(y, dtype=float)
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


def main():
    out = data_path("figdata_scaleup.csv")
    rows = []

    for tbl_name, ds_label in TABLES_2K:
        p = table_path(tbl_name)
        if not os.path.exists(p):
            print(f"MISSING {tbl_name}")
            continue

        data = [json.loads(l) for l in open(p)]
        n = len(data)
        cb = np.array([r["closed_correct"] for r in data], dtype=float)
        ob5 = np.array([r["open_correct_k5"] for r in data], dtype=float)
        ob1 = np.array(
            [r.get("open_correct_k1", r["open_correct_k5"]) for r in data], dtype=float
        )
        p_cal = np.array([r["p_correct"] for r in data])

        bin_auc = binary_frontier_auc(p_cal, cb, ob5)
        gate_auroc = auroc(cb, p_cal)
        flip = float(((cb == 1) & (ob5 == 0)).sum() / max((cb == 1).sum(), 1))

        row = {
            "dataset": ds_label,
            "n": n,
            "cb_acc": round(float(cb.mean()), 4),
            "ob5_acc": round(float(ob5.mean()), 4),
            "gate_auroc": round(gate_auroc, 4),
            "binary_auc": round(bin_auc, 4),
            "harm_rate": round(flip, 4),
        }
        for axis in AXES:
            grd = graded_frontier_auc(p_cal, cb, ob1, ob5, axis)
            row[f"graded_auc_{axis}"] = round(grd, 4)

        # Normalized value
        rand_auc = (float(cb.mean()) + float(ob5.mean())) / 2
        oracle_auc = binary_frontier_auc((cb - ob5).astype(float), cb, ob5)
        nv = (
            (bin_auc - rand_auc) / (oracle_auc - rand_auc)
            if oracle_auc > rand_auc
            else float("nan")
        )
        row["norm_value"] = round(nv, 4)

        rows.append(row)
        print(
            f"{ds_label:12s} n={n:5d}  CB={cb.mean():.3f}  OB@5={ob5.mean():.3f}  "
            f"bin={bin_auc:.4f}  grd_fc={row['graded_auc_full_context']:.4f}  "
            f"grd_rc={row['graded_auc_retrieval_call']:.4f}  "
            f"grd_pb={row['graded_auc_passage_budget']:.4f}"
        )

    cols = [
        "dataset",
        "n",
        "cb_acc",
        "ob5_acc",
        "gate_auroc",
        "binary_auc",
        "graded_auc_full_context",
        "graded_auc_retrieval_call",
        "graded_auc_passage_budget",
        "norm_value",
        "harm_rate",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
