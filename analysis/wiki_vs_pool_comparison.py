"""Wikipedia-scale retrieval vs per-question-pool comparison."""

from __future__ import annotations
import json, os, sys, csv
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import table_path, data_path

COMPARISONS = [
    # (dataset, pool_table, wiki_table)
    ("TriviaQA-8B", "triviaqa_rc_table.jsonl", "triviaqa_rc_wiki_table.jsonl"),
    ("NQ-8B", "nq_dpr_table.jsonl", "nq_dpr_wiki_table.jsonl"),
    ("MS-MARCO-8B", "msmarco_table.jsonl", "msmarco_wiki_table.jsonl"),
]


# Try logs/ fallback for wiki tables
def find_table(name):
    try:
        return table_path(name)
    except FileNotFoundError:
        fallback = os.path.join(HERE, "..", "logs", name)
        if os.path.exists(fallback):
            return fallback
        raise


def compute_metrics(rows):
    """Compute key metrics from a per-query table."""
    cb = np.array([r["closed_correct"] for r in rows], dtype=float)
    ob5 = np.array([r["open_correct_k5"] for r in rows], dtype=float)
    ob1 = np.array(
        [r.get("open_correct_k1", r.get("open_correct_k5")) for r in rows], dtype=float
    )
    seq_lp = np.array([r.get("seq_logprob", 0) for r in rows], dtype=float)
    n = len(rows)

    # Basic accuracies
    cb_acc = cb.mean()
    ob5_acc = ob5.mean()
    ob1_acc = ob1.mean()

    # Gate AUROC (OOF logistic)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if cb.min() != cb.max():
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
        best_acc_at_ret = {}
        for ret_rate in [0.25, 0.50, 0.75]:
            a = int((1 - ret_rate) * n)
            acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
            best_acc_at_ret[f"acc@{int(ret_rate * 100)}%"] = acc
        # Frontier AUC (trapezoidal)
        pts = {}
        for a in range(n + 1):
            acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
            x = (n - a) / n
            pts[x] = max(pts.get(x, -np.inf), float(acc))
        xs = np.array(sorted(pts))
        ys = np.array([pts[x] for x in xs])
        fr_auc = float(np.trapezoid(ys, xs))
        # Retrieval saved at match-always: find max skip where acc >= always-RAG
        always_acc = ob5_acc
        saved = 0.0
        for a in range(n + 1):
            acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
            if acc >= always_acc - 1e-9:
                saved = a / n
    else:
        auroc = float("nan")
        fr_auc = float("nan")
        saved = float("nan")
        best_acc_at_ret = {}

    # Retrieval harm
    cc_mask = cb.astype(bool)
    harm_rate = float((ob5[cc_mask] == 0).mean()) if cc_mask.sum() else float("nan")
    rescue_rate = (
        float((ob5[~cc_mask].astype(bool)).mean()) if (~cc_mask).sum() else float("nan")
    )

    return {
        "n": n,
        "cb_acc": cb_acc,
        "ob1_acc": ob1_acc,
        "ob5_acc": ob5_acc,
        "gate_auroc": auroc,
        "frontier_auc": fr_auc,
        "harm_rate": harm_rate,
        "rescue_rate": rescue_rate,
        "saved_at_match": saved,
        **best_acc_at_ret,
    }


def main():
    rows = []
    print(
        f"{'Dataset':<16} {'Mode':<10} {'CB':>6} {'OB@1':>6} {'OB@5':>6} {'AUROC':>7} {'FR_AUC':>7} {'Saved':>6} {'Harm':>6}"
    )
    print("-" * 85)

    for ds_label, pool_tbl, wiki_tbl in COMPARISONS:
        pool_metrics = None
        wiki_metrics = None

        try:
            pool_data = [json.loads(l) for l in open(find_table(pool_tbl))]
            pool_metrics = compute_metrics(pool_data)
            print(
                f"{ds_label:<16} {'pool':<10} {pool_metrics['cb_acc']:>6.3f} {pool_metrics['ob1_acc']:>6.3f} {pool_metrics['ob5_acc']:>6.3f} {pool_metrics['gate_auroc']:>7.3f} {pool_metrics['frontier_auc']:>7.3f} {pool_metrics['saved_at_match']:>6.1%} {pool_metrics['harm_rate']:>6.1%}"
            )
        except FileNotFoundError:
            print(f"{ds_label:<16} {'pool':<10} {'(missing)':>50}")

        try:
            wiki_data = [json.loads(l) for l in open(find_table(wiki_tbl))]
            wiki_metrics = compute_metrics(wiki_data)
            print(
                f"{ds_label:<16} {'wiki':<10} {wiki_metrics['cb_acc']:>6.3f} {wiki_metrics['ob1_acc']:>6.3f} {wiki_metrics['ob5_acc']:>6.3f} {wiki_metrics['gate_auroc']:>7.3f} {wiki_metrics['frontier_auc']:>7.3f} {wiki_metrics['saved_at_match']:>6.1%} {wiki_metrics['harm_rate']:>6.1%}"
            )
        except FileNotFoundError:
            print(
                f"{ds_label:<16} {'wiki':<10} {'(not yet run — submit wiki job)':>50}"
            )

        if pool_metrics and wiki_metrics:
            delta_fr = wiki_metrics["frontier_auc"] - pool_metrics["frontier_auc"]
            delta_ob = wiki_metrics["ob5_acc"] - pool_metrics["ob5_acc"]
            print(
                f"{'':<16} {'Δ (wiki−pool)':<10} {'':>6} {'':>6} {delta_ob:>+6.3f} {'':>7} {delta_fr:>+7.3f}"
            )
        print()

        if pool_metrics:
            for mode, m in [("pool", pool_metrics), ("wiki", wiki_metrics)]:
                if m is None:
                    continue
                rows.append(
                    {
                        "dataset": ds_label,
                        "retrieval_mode": mode,
                        **{
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in m.items()
                        },
                    }
                )

    if rows:
        path = data_path("figdata_wiki_vs_pool.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
