"""Compare logistic OOF, isotonic, and temperature scaling."""

from __future__ import annotations
import json, os, sys
import numpy as np


from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import resolve

DATASETS = [
    ("triviaqa_rc_table.jsonl", "TriviaQA-8B"),
    ("nq_dpr_table.jsonl", "NQ-8B"),
    ("msmarco_table.jsonl", "MS-MARCO-8B"),
    ("popqa_table.jsonl", "PopQA-8B"),
]


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.sum() * abs(probs[mask].mean() - labels[mask].mean())
    return ece_val / len(probs)


def temperature_scale_oof(logits, labels, cv=5):
    """OOF temperature scaling: fit T on train fold, apply on val fold."""
    n = len(logits)
    idx = np.arange(n)
    folds = np.array_split(idx, cv)
    probs = np.zeros(n)
    for i in range(cv):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(cv) if j != i])
        # fit T: minimize NLL on train
        from scipy.optimize import minimize_scalar

        def nll(T):
            p = 1 / (1 + np.exp(-logits[train] / max(T, 1e-6)))
            return -np.mean(
                labels[train] * np.log(p + 1e-9)
                + (1 - labels[train]) * np.log(1 - p + 1e-9)
            )

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T = res.x
        probs[val] = 1 / (1 + np.exp(-logits[val] / T))
    return probs


def main():
    print(f"{'Dataset':<18} {'Method':<22} {'AUROC':>7} {'ECE':>7}")
    print("-" * 60)
    rows = []
    for fname, label in DATASETS:
        p = resolve(fname)
        if not os.path.exists(p):
            print(f"MISSING {fname}")
            continue
        data = [json.loads(l) for l in open(p)]
        cb = np.array([r["closed_correct"] for r in data], float)
        lp = np.array([r["seq_logprob"] for r in data])

        methods = {}

        # 1. Logistic regression (OOF, standard)
        lr_probs = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            lp.reshape(-1, 1),
            cb,
            cv=5,
            method="predict_proba",
        )[:, 1]
        methods["Logistic (OOF)"] = lr_probs

        # 2. Isotonic regression (OOF)
        from sklearn.model_selection import KFold

        iso_probs = np.zeros(len(cb))
        kf = KFold(n_splits=5, shuffle=False)
        for train_idx, val_idx in kf.split(lp):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(lp[train_idx], cb[train_idx])
            iso_probs[val_idx] = iso.predict(lp[val_idx])
        methods["Isotonic (OOF)"] = iso_probs

        # 3. Temperature scaling (OOF)
        ts_probs = temperature_scale_oof(lp, cb)
        methods["Temperature (OOF)"] = ts_probs

        # 4. Raw seq_logprob (no calibration — ranking baseline)
        # normalize to [0,1] for fair ECE
        lp_norm = (lp - lp.min()) / (lp.max() - lp.min() + 1e-9)
        methods["Raw logprob (norm)"] = lp_norm

        for mname, probs in methods.items():
            auroc = roc_auc_score(cb, probs)
            e = ece(probs, cb)
            print(f"{label:<18} {mname:<22} {auroc:>7.4f} {e:>7.4f}")
            rows.append(dict(dataset=label, method=mname, auroc=auroc, ece=e))
        print()

    # Summary: which method wins on AUROC vs ECE?
    import csv

    out_path = os.path.join(HERE, "data", "figdata_calibration_methods.csv")
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "auroc", "ece"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
