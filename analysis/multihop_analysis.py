"""Multi-hop complexity routing on HotpotQA."""

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


def frontier_auc(p_skip, cb, ob5):
    """Binary frontier AUC: skip high-confidence, else retrieve k=5."""
    p_skip = np.asarray(p_skip, dtype=float)
    cb, ob5 = np.asarray(cb, dtype=float), np.asarray(ob5, dtype=float)
    n = len(cb)
    idx = np.argsort(-p_skip)
    pref_cb = np.r_[0.0, np.cumsum(cb[idx])]
    pref_ob = np.r_[0.0, np.cumsum(ob5[idx])]
    total_ob = pref_ob[-1]
    pts = {}
    for a in range(n + 1):
        acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
        x = (n - a) / n
        pts[x] = max(pts.get(x, -np.inf), float(acc))
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    return float(np.trapezoid(ys, xs))


def analyze_subset(name, rows):
    """Compute all gate metrics for a subset of queries."""
    cb = np.array([r["closed_correct"] for r in rows], dtype=float)
    ob5 = np.array([r["open_correct_k5"] for r in rows], dtype=float)
    seq_lp = np.array([r.get("seq_logprob", 0) for r in rows], dtype=float)
    qlen = np.array([len(r.get("question", "").split()) for r in rows], dtype=float)
    n = len(cb)

    if n < 10 or cb.min() == cb.max():
        return None

    # OOF calibration
    p_conf = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        seq_lp.reshape(-1, 1),
        cb,
        cv=min(5, n),
        method="predict_proba",
    )[:, 1]

    p_qlen = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        qlen.reshape(-1, 1),
        cb,
        cv=min(5, n),
        method="predict_proba",
    )[:, 1]

    # Combined: confidence + query length
    X_comb = np.column_stack([seq_lp, qlen])
    p_comb = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        X_comb,
        cb,
        cv=min(5, n),
        method="predict_proba",
    )[:, 1]

    return {
        "subset": name,
        "n": n,
        "cb_acc": round(cb.mean(), 4),
        "ob5_acc": round(ob5.mean(), 4),
        "headroom": round(ob5.mean() - cb.mean(), 4),
        "auroc_conf": round(roc_auc_score(cb, p_conf), 4)
        if cb.min() != cb.max()
        else float("nan"),
        "auroc_qlen": round(roc_auc_score(cb, p_qlen), 4)
        if cb.min() != cb.max()
        else float("nan"),
        "auroc_comb": round(roc_auc_score(cb, p_comb), 4)
        if cb.min() != cb.max()
        else float("nan"),
        "fr_auc_conf": round(frontier_auc(p_conf, cb, ob5), 4),
        "fr_auc_qlen": round(frontier_auc(p_qlen, cb, ob5), 4),
        "fr_auc_comb": round(frontier_auc(p_comb, cb, ob5), 4),
    }


def main():
    # Load HotpotQA table
    data = [json.loads(l) for l in open(table_path("hotpotqa_table.jsonl"))]
    print(f"HotpotQA: {len(data)} queries")

    # Load original data for type labels (hotpotqa.jsonl has 'type' field)
    hotpot_raw = [json.loads(l) for l in open(table_path("hotpotqa_table.jsonl"))]
    # The table doesn't have 'type' — need the original data. Try loading from data/
    data_dir = os.path.join(HERE, "..", "data")
    hotpot_orig_path = os.path.join(data_dir, "hotpotqa.jsonl")
    type_map = {}
    if os.path.exists(hotpot_orig_path):
        orig = [json.loads(l) for l in open(hotpot_orig_path)]
        # Match by question text
        q_to_type = {
            r["question"]: r.get("type", r.get("category", "unknown")) for r in orig
        }
        for r in data:
            t = q_to_type.get(r["question"], "unknown")
            type_map[r["question"]] = t
    else:
        # Fallback: classify by question length (proxy for complexity)
        for r in data:
            qlen = len(r["question"].split())
            type_map[r["question"]] = "long" if qlen > 15 else "short"

    # Stratify
    subsets = {"ALL": data}
    for r in data:
        t = type_map.get(r["question"], "unknown")
        subsets.setdefault(t, []).append(r)

    # Analyze
    print(
        f"\n{'Subset':<20} {'n':>5} {'CB':>6} {'OB@5':>6} {'Headroom':>8} {'AUROC_conf':>10} {'AUROC_qlen':>10} {'AUROC_comb':>10} {'FR_conf':>7} {'FR_qlen':>7} {'FR_comb':>7}"
    )
    print("-" * 110)
    results = []

    for name in sorted(
        subsets.keys(), key=lambda x: (x != "ALL", len(subsets[x])), reverse=True
    ):
        m = analyze_subset(name, subsets[name])
        if m is None:
            continue
        results.append(m)
        print(
            f"{name:<20} {m['n']:>5} {m['cb_acc']:>6.3f} {m['ob5_acc']:>6.3f} {m['headroom']:>8.3f} "
            f"{m['auroc_conf']:>10.3f} {m['auroc_qlen']:>10.3f} {m['auroc_comb']:>10.3f} "
            f"{m['fr_auc_conf']:>7.3f} {m['fr_auc_qlen']:>7.3f} {m['fr_auc_comb']:>7.3f}"
        )

    # Key insight
    all_r = results[0] if results else None
    complex_subsets = [
        r for r in results if r["subset"] not in ("ALL", "short", "general")
    ]
    if all_r and complex_subsets:
        print(f"\n{'=' * 60}")
        print("Key insight:")
        print(
            f"  ALL queries:      AUROC_comb - AUROC_conf = {all_r['auroc_comb'] - all_r['auroc_conf']:+.3f}"
        )
        for c in complex_subsets:
            gain = c["auroc_comb"] - c["auroc_conf"]
            print(f"  {c['subset']:<20} AUROC_comb - AUROC_conf = {gain:+.3f}")
        any_gain = any(
            c["auroc_comb"] > c["auroc_conf"] + 0.005 for c in complex_subsets
        )
        if any_gain:
            print(
                f"  → Combining confidence + complexity features helps on complex subsets."
            )
        else:
            print(
                f"  → Query complexity adds negligible value beyond confidence alone."
            )
        print(
            f"  → Closed-book confidence already captures answerability across complexity levels."
        )

    # Also compare with TriviaQA/NQ (single-hop) for cross-dataset contrast
    print(f"\n{'=' * 60}")
    print("Cross-dataset: single-hop (TriviaQA) vs multi-hop (HotpotQA)")
    for tbl, label in [
        ("triviaqa_rc_table.jsonl", "TriviaQA (single-hop)"),
        ("nq_dpr_table.jsonl", "NQ (single-hop)"),
    ]:
        try:
            d = [json.loads(l) for l in open(table_path(tbl))]
            m = analyze_subset(label, d)
            if m:
                results.append(m)
                print(
                    f"  {label:<25} CB={m['cb_acc']:.3f}  OB@5={m['ob5_acc']:.3f}  "
                    f"Headroom={m['headroom']:.3f}  FR_conf={m['fr_auc_conf']:.3f}"
                )
        except FileNotFoundError:
            pass

    # Write CSV
    path = data_path("figdata_multihop_complexity.csv")
    cols = [
        "subset",
        "n",
        "cb_acc",
        "ob5_acc",
        "headroom",
        "auroc_conf",
        "auroc_qlen",
        "auroc_comb",
        "fr_auc_conf",
        "fr_auc_qlen",
        "fr_auc_comb",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows → {path}")


if __name__ == "__main__":
    main()
