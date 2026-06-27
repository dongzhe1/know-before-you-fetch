"""Calibration sample efficiency: sweep 20 → 500 labeled examples."""

from __future__ import annotations
import json, os, sys, csv
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import table_path, data_path

CONFIGS = [
    ("triviaqa_rc_table.jsonl", "TriviaQA-8B"),
    ("nq_dpr_table.jsonl", "NQ-8B"),
    ("msmarco_table.jsonl", "MS-MARCO-8B"),
]

SIZES = [20, 40, 60, 80, 100, 150, 200, 300, 400, 500]
N_TRIALS = 20  # random trials per size
RNG = np.random.default_rng(42)


def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        e += mask.sum() * abs(probs[mask].mean() - labels[mask].mean())
    return e / len(probs)


def trial(X, y, train_size, test_idx, train_pool):
    """One random trial: sample train_size from train_pool, calibrate, evaluate on test_idx."""
    chosen = RNG.choice(train_pool, train_size, replace=False)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X[chosen].reshape(-1, 1), y[chosen])
    probs = clf.predict_proba(X[test_idx].reshape(-1, 1))[:, 1]
    cal_probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return {
        "auroc": roc_auc_score(y[test_idx], probs)
        if y[test_idx].min() != y[test_idx].max()
        else 0.5,
        "ece": ece(probs, y[test_idx]),
        "brier": brier_score_loss(y[test_idx], probs),
        "nll": log_loss(y[test_idx], cal_probs),
    }


def main():
    all_rows = []

    for tbl, label in CONFIGS:
        data = [json.loads(l) for l in open(table_path(tbl))]
        cb = np.array([r["closed_correct"] for r in data], dtype=float)
        seq_lp = np.array([r["seq_logprob"] for r in data], dtype=float)
        n = len(cb)

        # Hold out 30% as fixed test set
        all_idx = np.arange(n)
        RNG.shuffle(all_idx)
        n_test = int(n * 0.3)
        test_idx = all_idx[:n_test]
        train_pool = all_idx[n_test:]

        print(f"\n{label} (n={n}, test={n_test}, train_pool={len(train_pool)}):")
        print(f"  {'Size':>5}  {'AUROC':>7}  {'ECE':>7}  {'Brier':>7}  {'NLL':>7}")
        print(f"  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}")

        for size in SIZES:
            if size > len(train_pool):
                continue
            metrics = {"auroc": [], "ece": [], "brier": [], "nll": []}
            for _ in range(N_TRIALS):
                m = trial(seq_lp, cb, size, test_idx, train_pool)
                for k, v in m.items():
                    metrics[k].append(v)

            mu = {k: np.mean(v) for k, v in metrics.items()}
            std = {k: np.std(v) for k, v in metrics.items()}
            all_rows.append(
                {
                    "dataset": label,
                    "calibration_size": size,
                    "n_trials": N_TRIALS,
                    "auroc_mean": round(mu["auroc"], 4),
                    "auroc_std": round(std["auroc"], 4),
                    "ece_mean": round(mu["ece"], 4),
                    "ece_std": round(std["ece"], 4),
                    "brier_mean": round(mu["brier"], 4),
                    "brier_std": round(std["brier"], 4),
                    "nll_mean": round(mu["nll"], 4),
                    "nll_std": round(std["nll"], 4),
                }
            )
            print(
                f"  {size:>5}  {mu['auroc']:>7.3f}  {mu['ece']:>7.4f}  {mu['brier']:>7.4f}  {mu['nll']:>7.4f}"
            )

        # Find "good enough" size: where ECE stabilizes within 10% of final value
        ds_rows = [r for r in all_rows if r["dataset"] == label]
        final_ece = ds_rows[-1]["ece_mean"] if ds_rows else 0
        good_idx = next(
            (i for i, r in enumerate(ds_rows) if r["ece_mean"] <= final_ece * 1.1), None
        )
        if good_idx is not None:
            print(
                f"  → ECE stabilizes at calibration_size ≈ {ds_rows[good_idx]['calibration_size']}"
            )

    # Write CSV
    path = data_path("figdata_calibration_size_sensitivity.csv")
    cols = [
        "dataset",
        "calibration_size",
        "n_trials",
        "auroc_mean",
        "auroc_std",
        "ece_mean",
        "ece_std",
        "brier_mean",
        "brier_std",
        "nll_mean",
        "nll_std",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {path}")


if __name__ == "__main__":
    main()
