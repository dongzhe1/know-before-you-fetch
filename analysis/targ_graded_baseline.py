"""TARG signals + our calibration framework: graded vs binary frontier."""

from __future__ import annotations
import json, os, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _tables import table_path, data_path
from experiment_registry import get_experiment
from frontier_metrics import (
    binary_frontier_auc,
    graded_frontier_auc,
    paired_bootstrap_graded_vs_binary,
    bootstrap_ci,
)

HERE = os.path.dirname(__file__)
RNG = np.random.default_rng(42)

_ALL_SIGNALS = [
    ("seq_logprob", "Ours (seq-logprob)"),
    ("mean_entropy", "TARG entropy"),
    ("top1_top2_margin", "TARG margin"),
    ("prefix_variance", "TARG variance"),
]

AXES = ["full_context", "retrieval_call", "passage_budget"]

# Configs: (registry_key, targ_table_file, ksweep_table_file)
CONFIGS = [
    (
        "triviaqa_qwen8b_bgelarge",
        "triviaqa_rc_targ_features.jsonl",
        "triviaqa_rc_ksweep_table.jsonl",
    ),
    (
        "triviaqa_qwen32b_bgelarge",
        "triviaqa_rc_targ_features_32b.jsonl",
        "triviaqa_rc_ksweep_32b_table.jsonl",
    ),
]


def calibrate_oof(signal, y):
    ok = np.isfinite(signal) & ~np.isnan(signal)
    p = np.full(len(signal), 0.5)
    if ok.sum() < 10:
        return p
    X = signal[ok].reshape(-1, 1)
    p[ok] = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        X,
        y[ok],
        cv=5,
        method="predict_proba",
    )[:, 1]
    return p


def main():
    out_csv = data_path("figdata_targ_graded.csv")
    w = csv.writer(open(str(out_csv), "w", newline=""))
    w.writerow(
        [
            "dataset",
            "model",
            "signal",
            "policy",
            "axis",
            "auc",
            "ci_low",
            "ci_high",
            "delta_vs_binary",
            "delta_ci_low",
            "delta_ci_high",
        ]
    )

    print(
        f"{'Dataset':<16} {'Signal':<22} {'Axis':<18} {'Binary':>8} {'Graded':>8} "
        f"{'Δ':>8}  {'Δ 95% CI':>22}"
    )
    print("-" * 120)

    for reg_key, targ_fname, ksw_fname in CONFIGS:
        exp = get_experiment(reg_key)
        if exp is None:
            print(f"MISSING registry key: {reg_key}")
            continue

        label = f"{exp['dataset']} ({exp['model']})"
        targ_path = table_path(targ_fname)
        ksw_path = table_path(ksw_fname)

        targ_rows = [json.loads(l) for l in open(targ_path)]
        ksw_rows = [json.loads(l) for l in open(ksw_path)]
        n = len(targ_rows)

        cb = np.array([r["closed_correct"] for r in ksw_rows], dtype=float)
        ob5 = np.array([r["open_correct_k5"] for r in ksw_rows], dtype=float)
        ob1 = np.array([r["open_correct_k1"] for r in ksw_rows], dtype=float)

        print(f"\n{label} (n={n})")
        print(
            f"  Anchors: CB={cb.mean():.3f}  OB@1={ob1.mean():.3f}  OB@5={ob5.mean():.3f}"
        )

        for sig_key, sig_name in _ALL_SIGNALS:
            raw = np.array(
                [targ_rows[i].get(sig_key, np.nan) for i in range(n)], dtype=float
            )
            if np.isfinite(raw).sum() < 10 or np.nanstd(raw) < 1e-9:
                print(
                    f"  SKIP {sig_name}: degenerate (finite={np.isfinite(raw).sum()}, std={np.nanstd(raw):.2e})"
                )
                continue
            if sig_key in ("mean_entropy", "prefix_variance"):
                raw = -raw  # higher = less certain → negate

            p_cal = calibrate_oof(raw, cb)

            # Binary AUC (same on all axes for binary — only k=5, no k=1)
            bin_auc = binary_frontier_auc(p_cal, cb, ob5)
            bin_lo, bin_hi = bootstrap_ci(binary_frontier_auc, (p_cal, cb, ob5), B=500)

            # Write binary row (retrieval_call axis as canonical)
            w.writerow(
                [
                    exp["dataset"],
                    exp["model"],
                    sig_name,
                    "binary",
                    "retrieval_call",
                    round(bin_auc, 4),
                    round(bin_lo, 4),
                    round(bin_hi, 4),
                    "",
                    "",
                    "",
                ]
            )

            # Graded on each axis
            first_sig = True
            for axis in AXES:
                grd_auc = graded_frontier_auc(p_cal, cb, ob1, ob5, axis)
                grd_lo, grd_hi = bootstrap_ci(
                    graded_frontier_auc, (p_cal, cb, ob1, ob5), B=500, axis=axis
                )
                delta, d_lo, d_hi = paired_bootstrap_graded_vs_binary(
                    p_cal, cb, ob1, ob5, axis=axis, B=500
                )

                w.writerow(
                    [
                        exp["dataset"],
                        exp["model"],
                        sig_name,
                        "graded",
                        axis,
                        round(grd_auc, 4),
                        round(grd_lo, 4),
                        round(grd_hi, 4),
                        round(delta, 4),
                        round(d_lo, 4),
                        round(d_hi, 4),
                    ]
                )

                if first_sig:
                    print(
                        f"  {sig_name:<22} {'binary':>8} {'—':>18} {bin_auc:>8.4f} {'—':>8}"
                    )
                    first_sig = False
                sig_str = " " * 22
                verdict = "★" if d_lo > 0 else ("tie" if d_lo <= 0 <= d_hi else "")
                print(
                    f"  {sig_str} {'graded':>8} {axis:<18} {grd_auc:>8.4f} {delta:>+8.4f}  [{d_lo:+.4f}, {d_hi:+.4f}] {verdict}"
                )

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
