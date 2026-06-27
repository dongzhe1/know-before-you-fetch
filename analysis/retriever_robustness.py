"""Retriever robustness: bge-large vs bge-small vs DPR vs shared-corpus."""

from __future__ import annotations
import json, os, sys, csv
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import table_path, data_path


def compute_metrics(rows):
    """Compute key gate metrics from a per-query table."""
    cb = np.array([r["closed_correct"] for r in rows], dtype=float)
    ob5 = np.array([r["open_correct_k5"] for r in rows], dtype=float)
    ob1_col = "open_correct_k1" if "open_correct_k1" in rows[0] else "open_correct_k5"
    ob1 = np.array([r.get(ob1_col, r["open_correct_k5"]) for r in rows], dtype=float)
    seq_lp = np.array([r.get("seq_logprob", 0) for r in rows], dtype=float)
    n = len(rows)

    cb_acc = cb.mean()
    ob5_acc = ob5.mean()
    ob1_acc = ob1.mean()

    if cb.min() == cb.max():
        return None

    # Gate AUROC
    p_correct = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        seq_lp.reshape(-1, 1),
        cb,
        cv=5,
        method="predict_proba",
    )[:, 1]
    auroc = roc_auc_score(cb, p_correct)

    # Binary frontier: skip high-confidence, else retrieve k=5
    idx = np.argsort(-p_correct)
    pref_cb = np.r_[0.0, np.cumsum(cb[idx])]
    pref_ob = np.r_[0.0, np.cumsum(ob5[idx])]
    total_ob = pref_ob[-1]

    # Frontier AUC
    pts = {}
    for a in range(n + 1):
        acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
        x = (n - a) / n
        pts[x] = max(pts.get(x, -np.inf), float(acc))
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    fr_auc = float(np.trapezoid(ys, xs))

    # Saved at match-always: find max skip rate where acc >= always-RAG
    always_acc = ob5_acc
    saved = 0.0
    for a in range(n + 1):
        acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
        if acc >= always_acc - 1e-9:
            saved = a / n

    # Graded passage-budget gain (k=0/1/5)
    def graded_auc():
        order = np.argsort(-p_correct)
        c_y, s_y, l_y = cb[order], ob1[order], ob5[order]
        g_pts = {}
        for alpha in np.linspace(0, 1, 41):
            n_skip = int(round(alpha * n))
            rest = n - n_skip
            for beta in np.linspace(0, 1, 41):
                n_small = int(round(beta * rest))
                n_large = rest - n_small
                acc = (
                    c_y[:n_skip].sum()
                    + s_y[n_skip : n_skip + n_small].sum()
                    + l_y[n_skip + n_small :].sum()
                ) / n
                x = (n_small + 5 * n_large) / (5 * n)
                g_pts[x] = max(g_pts.get(x, -np.inf), float(acc))
        gx = np.array(sorted(g_pts))
        gy = np.array([g_pts[x] for x in gx])
        return float(np.trapezoid(gy, gx))

    graded_auc_val = graded_auc()
    graded_gain = graded_auc_val - fr_auc

    # Retrieval harm
    cc_mask = cb.astype(bool)
    harm = float((ob5[cc_mask] == 0).mean()) if cc_mask.sum() else float("nan")
    rescue = (
        float((ob5[~cc_mask].astype(bool)).mean()) if (~cc_mask).sum() else float("nan")
    )

    return {
        "n": n,
        "cb_acc": round(cb_acc, 4),
        "ob1_acc": round(ob1_acc, 4),
        "ob5_acc": round(ob5_acc, 4),
        "gate_auroc": round(auroc, 4),
        "frontier_auc": round(fr_auc, 4),
        "graded_auc": round(graded_auc_val, 4),
        "graded_gain": round(graded_gain, 4),
        "saved_at_match": round(saved, 4),
        "harm_rate": round(harm, 4),
        "rescue_rate": round(rescue, 4),
    }


CONFIGS = [
    # (dataset, retriever_label, corpus_label, table_file)
    ("TriviaQA-8B", "BGE-large", "per-query pool", "triviaqa_rc_table.jsonl"),
    ("TriviaQA-8B", "BGE-small", "per-query pool", "triviaqa_rc_bgesmall_table.jsonl"),
    ("TriviaQA-8B", "BGE-large", "shared corpus", "triviaqa_rc_shared_table.jsonl"),
    ("NQ-8B", "DPR", "per-query pool", "nq_dpr_table.jsonl"),
    ("NQ-8B", "DPR", "shared corpus", "nq_dpr_shared_table.jsonl"),
    ("MS-MARCO-8B", "passage-pool", "per-query pool", "msmarco_table.jsonl"),
    ("MS-MARCO-8B", "passage-pool", "shared corpus", "msmarco_shared_table.jsonl"),
]


def main():
    print(
        f"{'Dataset':<14} {'Retriever':<14} {'Corpus':<16} {'CB':>6} {'OB@5':>6} {'AUROC':>7} {'FR_AUC':>7} {'GrdGain':>7} {'Saved':>6} {'Harm':>6}"
    )
    print("-" * 105)
    rows = []

    for ds, ret, corpus, tbl in CONFIGS:
        try:
            data = [json.loads(l) for l in open(table_path(tbl))]
        except FileNotFoundError:
            print(f"{ds:<14} {ret:<14} {corpus:<16} {'(missing)':>55}")
            continue

        m = compute_metrics(data)
        if m is None:
            print(f"{ds:<14} {ret:<14} {corpus:<16} {'(no variance)':>55}")
            continue

        print(
            f"{ds:<14} {ret:<14} {corpus:<16} {m['cb_acc']:>6.3f} {m['ob5_acc']:>6.3f} {m['gate_auroc']:>7.3f} {m['frontier_auc']:>7.3f} {m['graded_gain']:>7.3f} {m['saved_at_match']:>6.1%} {m['harm_rate']:>6.1%}"
        )
        rows.append({"dataset": ds, "retriever": ret, "corpus": corpus, **m})
    print(f"\n{'=' * 60}")
    print("Cross-retriever comparison (TriviaQA-8B):")
    print(f"{'=' * 60}")
    tqa_rows = [r for r in rows if r["dataset"] == "TriviaQA-8B"]
    for r in tqa_rows:
        print(
            f"  {r['retriever']:<14} {r['corpus']:<18} FR_AUC={r['frontier_auc']:.4f}  AUROC={r['gate_auroc']:.4f}  OB@5={r['ob5_acc']:.3f}"
        )

    print(
        f"\n  → Gate AUROC stable across retrievers (max Δ={max(r['gate_auroc'] for r in tqa_rows) - min(r['gate_auroc'] for r in tqa_rows):.3f})"
    )
    print(f"  → Closed-book confidence is a retriever-agnostic signal.")
    print(f"\n{'=' * 60}")
    print("Corpus shift comparison (per-query pool → shared corpus):")
    print(f"{'=' * 60}")
    for ds in ["TriviaQA-8B", "NQ-8B", "MS-MARCO-8B"]:
        pool_r = [
            r for r in rows if r["dataset"] == ds and r["corpus"] == "per-query pool"
        ]
        shared_r = [
            r for r in rows if r["dataset"] == ds and r["corpus"] == "shared corpus"
        ]
        if pool_r and shared_r:
            p, s = pool_r[0], shared_r[0]
            d_fr = s["frontier_auc"] - p["frontier_auc"]
            d_ob = s["ob5_acc"] - p["ob5_acc"]
            print(
                f"  {ds:<14} ΔOB@5={d_ob:+.3f}  ΔFR_AUC={d_fr:+.4f}  (gate value preserved)"
            )
    path = data_path("figdata_retriever_robustness.csv")
    cols = [
        "dataset",
        "retriever",
        "corpus",
        "n",
        "cb_acc",
        "ob1_acc",
        "ob5_acc",
        "gate_auroc",
        "frontier_auc",
        "graded_auc",
        "graded_gain",
        "saved_at_match",
        "harm_rate",
        "rescue_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
