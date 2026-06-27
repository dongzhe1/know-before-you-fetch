"""Single-feature AUROC and frontier AUC ablation."""

from __future__ import annotations
import json, os, csv
import numpy as np

from _tables import resolve

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# higher value => more confident => skip.  flip sign for uncertainty features.
FEATURES = {
    "seq_logprob": (+1, "seq_logprob"),
    "mean_token_logprob": (+1, "mean_token_logprob"),
    "min_token_logprob": (+1, "min_token_logprob"),
    "top1_top2_margin": (+1, "top1_top2_margin"),
    "mean_entropy": (-1, "mean_entropy"),
    "prefix_variance": (-1, "prefix_variance"),
}
DATASETS = ["triviaqa_rc", "nq_dpr", "msmarco"]


def auroc(y, s):
    y = np.asarray(y)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(-s)
    tp = 0
    acc = 0
    for i in order:
        tp += y[i]
        acc += tp * (1 - y[i])
    return acc / (npos * nneg)


def frontier_auc(score, cb, ob):
    n = len(cb)
    idx = np.argsort(-score)
    pts = {1.0: float(ob.mean()), 0.0: float(cb.mean())}
    use = np.zeros(n, bool)
    for k in range(1, n + 1):
        use[idx[k - 1]] = True
        rr = float((~use).sum()) / n
        pts[rr] = max(pts.get(rr, 0), (cb[use].sum() + ob[~use].sum()) / n)
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    return float(_trapz(ys, xs))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, "figdata_feature_ablation.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["dataset", "feature", "auroc", "frontier_auc"])
    any_found = False
    for ds in DATASETS:
        fp = resolve(f"{ds}_targ_features.jsonl")
        if not os.path.exists(fp):
            print(f"MISSING {ds}_targ_features.jsonl (run scripts/targ_baseline.py first)")
            continue
        any_found = True
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        cb = np.array([r["closed_correct"] for r in rows])
        ob = np.array([r.get("open_correct_k5", 0) for r in rows])
        print(f"\n{ds}  (n={len(rows)})")
        print(f"  {'feature':<22}{'AUROC':>8}{'frAUC':>8}")
        cols = []
        for name, (sgn, key) in FEATURES.items():
            if key not in rows[0]:
                continue
            raw = np.array([r[key] for r in rows], dtype=float)
            if np.isfinite(raw).sum() < 10 or np.nanstd(raw) < 1e-9:
                print(
                    f"  {name:<22}  SKIP (degenerate: finite={np.isfinite(raw).sum()}, std={np.nanstd(raw):.2e})"
                )
                continue
            s = sgn * raw
            s = np.nan_to_num(
                s,
                nan=np.nanmedian(s) if np.isfinite(np.nanmedian(s)) else 0.0,
                posinf=0.0,
                neginf=0.0,
            )
            a = auroc(cb, s)
            fa = frontier_auc(s, cb, ob)
            print(f"  {name:<22}{a:>8.3f}{fa:>8.3f}")
            w.writerow([ds, name, round(a, 4), round(fa, 4)])
            if np.std(s) > 1e-9:
                cols.append(s)
        # fused (all non-degenerate features, OOF logistic)
        X = np.column_stack(cols)
        prob = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            X,
            cb,
            cv=5,
            method="predict_proba",
        )[:, 1]
        a = auroc(cb, prob)
        fa = frontier_auc(prob, cb, ob)
        print(f"  {'fused (all, OOF)':<22}{a:>8.3f}{fa:>8.3f}")
        w.writerow([ds, "fused_all_oof", round(a, 4), round(fa, 4)])
    out.close()
    if any_found:
        print(f"\nwrote {DATA}/figdata_feature_ablation.csv")
    else:
        print(
            "\nNo feature tables yet — run `python scripts/targ_baseline.py`, then sync"
            " logs/*_targ_features.jsonl and re-run this script."
        )


if __name__ == "__main__":
    main()
