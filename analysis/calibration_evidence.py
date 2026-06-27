"""ECE, Brier score, NLL, and reliability diagram for all signals."""

from __future__ import annotations
import json, os, sys, csv
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import table_path, data_path

CONFIGS = [
    # (dataset_label, main_table, targ_table, model_label)
    (
        "TriviaQA-8B",
        "triviaqa_rc_table.jsonl",
        "triviaqa_rc_targ_features.jsonl",
        "Qwen3-8B",
    ),
    ("NQ-8B", "nq_dpr_table.jsonl", "nq_dpr_targ_features.jsonl", "Qwen3-8B"),
    ("MS-MARCO-8B", "msmarco_table.jsonl", "msmarco_targ_features.jsonl", "Qwen3-8B"),
    (
        "TriviaQA-32B",
        "triviaqa_rc_32b_n600_table.jsonl",
        "triviaqa_rc_targ_features_32b.jsonl",
        "Qwen3-32B",
    ),
]

N_BINS = 10
N_FOLDS = 5


def ece(probs, labels, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        e += mask.sum() * abs(probs[mask].mean() - labels[mask].mean())
    return e / len(probs)


def oof_calibrate(features, labels, cv=N_FOLDS):
    """OOF logistic calibration. Returns calibrated probabilities."""
    X = (
        np.asarray(features, dtype=float).reshape(-1, 1)
        if features.ndim == 1
        else features
    )
    return cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        X,
        labels,
        cv=cv,
        method="predict_proba",
    )[:, 1]


def normalize_01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.full_like(x, 0.5)
    return (x - mn) / (mx - mn)


def reliability_curve(probs, labels, n_bins=N_BINS):
    """Return (bin_centers, bin_accuracies, bin_counts) for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        cnt = mask.sum()
        if cnt > 0:
            centers.append((lo + hi) / 2)
            accs.append(labels[mask].mean())
            counts.append(cnt)
    return np.array(centers), np.array(accs), np.array(counts)


def eval_signal(name, raw_values, labels, cal_probs=None):
    """Return dict with AUROC, ECE, Brier, NLL for raw and calibrated versions."""
    raw_norm = normalize_01(raw_values)
    results = {
        "signal": name,
        "auroc_raw": roc_auc_score(labels, raw_norm),
        "ece_raw": ece(raw_norm, labels),
        "brier_raw": brier_score_loss(labels, raw_norm),
        "nll_raw": log_loss(labels, np.clip(raw_norm, 1e-9, 1 - 1e-9)),
    }
    if cal_probs is not None:
        cal = np.clip(cal_probs, 1e-9, 1 - 1e-9)
        results.update(
            {
                "auroc_cal": roc_auc_score(labels, cal),
                "ece_cal": ece(cal, labels),
                "brier_cal": brier_score_loss(labels, cal),
                "nll_cal": log_loss(labels, cal),
            }
        )
    return results


def main():
    evidence_rows = []
    reliability_rows = []

    for ds_label, main_tbl, targ_tbl, model in CONFIGS:
        # Load main table
        main_data = [json.loads(l) for l in open(table_path(main_tbl))]
        cb = np.array([r["closed_correct"] for r in main_data], dtype=float)
        seq_lp = np.array(
            [r.get("seq_logprob", r.get("seq_logprob", 0)) for r in main_data]
        )

        # Load TARG table
        targ_data = [json.loads(l) for l in open(table_path(targ_tbl))]
        # Align by row order (both tables have 600 rows in same order)
        entropy = np.array([r.get("mean_entropy", 0) for r in targ_data], dtype=float)
        margin = np.array(
            [r.get("top1_top2_margin", 0) for r in targ_data], dtype=float
        )
        variance = np.array(
            [r.get("prefix_variance", 0) for r in targ_data], dtype=float
        )

        # Ensure same length (use min)
        n = min(len(cb), len(seq_lp), len(entropy))
        cb = cb[:n]
        seq_lp = seq_lp[:n]
        entropy = entropy[:n]
        margin = margin[:n]
        variance = variance[:n]

        # Handle NaN: skip signals that are entirely NaN (e.g., prefix_variance not computed)
        def is_valid(arr):
            return not np.all(np.isnan(arr))

        valid_signals = {
            "seq_logprob": seq_lp,
            "entropy": entropy,
            "margin": margin,
            "variance": variance,
        }
        if not is_valid(variance):
            print(f"  [SKIP] TARG_variance is all-NaN — skipping")
        if not is_valid(margin):
            print(f"  [SKIP] TARG_margin is all-NaN — skipping")

        print(f"\n{'=' * 60}")
        print(f"{ds_label} ({model}) | n={n} | CB acc={cb.mean():.3f}")
        print(f"{'=' * 60}")
        print(
            f"{'Signal':<28} {'AUROC_raw':>9} {'ECE_raw':>8} {'Brier_raw':>9} {'NLL_raw':>8} | {'AUROC_cal':>9} {'ECE_cal':>8} {'Brier_cal':>9} {'NLL_cal':>8}"
        )
        print("-" * 110)
        cal_lp = oof_calibrate(seq_lp, cb)
        r = eval_signal("seq_logprob", seq_lp, cb, cal_lp)
        r.update({"dataset": ds_label, "model": model})
        evidence_rows.append(r)
        print(
            f"{'seq_logprob':<28} {r['auroc_raw']:>9.4f} {r['ece_raw']:>8.4f} {r['brier_raw']:>9.4f} {r['nll_raw']:>8.4f} | {r.get('auroc_cal', 0):>9.4f} {r.get('ece_cal', 0):>8.4f} {r.get('brier_cal', 0):>9.4f} {r.get('nll_cal', 0):>8.4f}"
        )

        # Reliability data for calibrated seq_logprob
        centers, accs, counts = reliability_curve(cal_lp, cb)
        for c, a, cnt in zip(centers, accs, counts):
            reliability_rows.append(
                {
                    "dataset": ds_label,
                    "model": model,
                    "signal": "seq_logprob_calibrated",
                    "bin_center": round(float(c), 3),
                    "accuracy": round(float(a), 4),
                    "count": int(cnt),
                }
            )
        cal_ent = oof_calibrate(entropy, cb)
        r = eval_signal(
            "TARG_entropy", -entropy, cb, cal_ent
        )  # negate: lower entropy = more confident
        r.update({"dataset": ds_label, "model": model})
        evidence_rows.append(r)
        print(
            f"{'TARG_entropy':<28} {r['auroc_raw']:>9.4f} {r['ece_raw']:>8.4f} {r['brier_raw']:>9.4f} {r['nll_raw']:>8.4f} | {r.get('auroc_cal', 0):>9.4f} {r.get('ece_cal', 0):>8.4f} {r.get('brier_cal', 0):>9.4f} {r.get('nll_cal', 0):>8.4f}"
        )

        centers, accs, counts = reliability_curve(cal_ent, cb)
        for c, a, cnt in zip(centers, accs, counts):
            reliability_rows.append(
                {
                    "dataset": ds_label,
                    "model": model,
                    "signal": "TARG_entropy_calibrated",
                    "bin_center": round(float(c), 3),
                    "accuracy": round(float(a), 4),
                    "count": int(cnt),
                }
            )
        if is_valid(margin):
            cal_mar = oof_calibrate(margin, cb)
            r = eval_signal("TARG_margin", margin, cb, cal_mar)
            r.update({"dataset": ds_label, "model": model})
            evidence_rows.append(r)
            print(
                f"{'TARG_margin':<28} {r['auroc_raw']:>9.4f} {r['ece_raw']:>8.4f} {r['brier_raw']:>9.4f} {r['nll_raw']:>8.4f} | {r.get('auroc_cal', 0):>9.4f} {r.get('ece_cal', 0):>8.4f} {r.get('brier_cal', 0):>9.4f} {r.get('nll_cal', 0):>8.4f}"
            )
        if is_valid(variance):
            var_vals = np.where(np.isnan(variance), np.nanmean(variance), variance)
            cal_var = oof_calibrate(-var_vals, cb)
            r = eval_signal("TARG_variance", -var_vals, cb, cal_var)
            r.update({"dataset": ds_label, "model": model})
            evidence_rows.append(r)
            print(
                f"{'TARG_variance':<28} {r['auroc_raw']:>9.4f} {r['ece_raw']:>8.4f} {r['brier_raw']:>9.4f} {r['nll_raw']:>8.4f} | {r.get('auroc_cal', 0):>9.4f} {r.get('ece_cal', 0):>8.4f} {r.get('brier_cal', 0):>9.4f} {r.get('nll_cal', 0):>8.4f}"
            )
        else:
            print(f"{'TARG_variance':<28} {'(all NaN — skipped)':>60}")
        fuse_components = [seq_lp, -entropy]
        if is_valid(margin):
            fuse_components.append(margin)
        if is_valid(variance):
            fuse_components.append(np.where(np.isnan(variance), 0, -variance))
        X_fused = np.column_stack(fuse_components)
        cal_fused = oof_calibrate(X_fused, cb)
        r = eval_signal("fused_all", cal_fused, cb, cal_fused)
        r["auroc_raw"] = roc_auc_score(cb, cal_fused)
        r.update({"dataset": ds_label, "model": model})
        evidence_rows.append(r)
        print(
            f"{'fused_all':<28} {r['auroc_raw']:>9.4f} {r['ece_raw']:>8.4f} {r['brier_raw']:>9.4f} {r['nll_raw']:>8.4f}"
        )
        print()
    evidence_cols = [
        "dataset",
        "model",
        "signal",
        "auroc_raw",
        "ece_raw",
        "brier_raw",
        "nll_raw",
        "auroc_cal",
        "ece_cal",
        "brier_cal",
        "nll_cal",
    ]
    path1 = data_path("figdata_calibration_evidence.csv")
    with open(path1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=evidence_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(evidence_rows)
    print(f"Wrote {len(evidence_rows)} rows → {path1}")

    rel_cols = ["dataset", "model", "signal", "bin_center", "accuracy", "count"]
    path2 = data_path("figdata_reliability.csv")
    with open(path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rel_cols)
        w.writeheader()
        w.writerows(reliability_rows)
    print(f"Wrote {len(reliability_rows)} rows → {path2}")


if __name__ == "__main__":
    main()
