"""Held-out deployment: 5-fold CV, threshold on val, evaluate on test."""

from __future__ import annotations
import json, os, sys, csv
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import table_path, data_path

CONFIGS = [
    ("TriviaQA-8B", "triviaqa_rc_table.jsonl", "triviaqa_rc_targ_features.jsonl"),
    ("NQ-8B", "nq_dpr_table.jsonl", "nq_dpr_targ_features.jsonl"),
    ("MS-MARCO-8B", "msmarco_table.jsonl", "msmarco_targ_features.jsonl"),
]

N_OUTER = 5
LAMBDA_VALS = [0.0, 0.05, 0.10, 0.20, 0.30]
RNG = np.random.default_rng(42)


def calibrate(train_X, train_y, test_X):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(train_X.reshape(-1, 1), train_y)
    return clf.predict_proba(test_X.reshape(-1, 1))[:, 1]


def select_threshold_match_always(p_correct, cb, ob5, n_steps=200):
    always_acc = ob5.mean()
    idx = np.argsort(-p_correct)
    best_skip, best_tau = 0, 1.0
    for i in range(0, len(cb) + 1, max(1, len(cb) // n_steps)):
        skip_mask = np.zeros(len(cb), bool)
        skip_mask[idx[:i]] = True
        if np.where(skip_mask, cb, ob5).mean() >= always_acc - 1e-9:
            if i > best_skip:
                best_skip = i
                best_tau = p_correct[idx[i - 1]] if i > 0 else 1.01
    return best_tau, best_skip / len(cb)


def select_threshold_fixed_retrieval(p_correct, target_ret_rate):
    idx_sorted = np.argsort(-p_correct)
    target_skip = int((1 - target_ret_rate) * len(p_correct))
    if target_skip <= 0:
        return 1.01, 1.0
    if target_skip >= len(p_correct):
        return -0.01, 0.0
    return p_correct[idx_sorted[target_skip - 1]], target_ret_rate


def select_threshold_max_utility(p_correct, cb, ob5, lambda_r, n_steps=200):
    idx = np.argsort(-p_correct)
    best_util, best_skip, best_tau = -np.inf, 0, 1.0
    for i in range(0, len(cb) + 1, max(1, len(cb) // n_steps)):
        skip_mask = np.zeros(len(cb), bool)
        skip_mask[idx[:i]] = True
        acc = np.where(skip_mask, cb, ob5).mean()
        util = acc - lambda_r * (1 - skip_mask.mean())
        if util > best_util:
            best_util, best_skip = util, i
            best_tau = p_correct[idx[i - 1]] if i > 0 else 1.01
    return best_tau, 1 - (best_skip / len(cb))


def oracle_upper_bound(cb, ob5):
    return {
        "accuracy": float(np.maximum(cb, ob5).mean()),
        "retrieval_rate": float((ob5 > cb).mean()),
        "skip_rate": float((cb >= ob5).mean()),
    }


def evaluate_policy(p_correct, tau_skip, cb, ob5):
    skip = p_correct >= tau_skip
    return {
        "skip_rate": float(skip.mean()),
        "retrieval_rate": float(1 - skip.mean()),
        "accuracy": float(np.where(skip, cb, ob5).mean()),
        "always_acc": float(ob5.mean()),
    }


def run_deployment(signal_name, raw_values, cb, ob5):
    """Run 5-fold CV deployment protocol for one signal. Returns list of result dicts."""
    n = len(cb)
    try:
        skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
        splits = list(skf.split(raw_values, cb.astype(int)))
    except ValueError:
        splits = list(
            KFold(n_splits=N_OUTER, shuffle=True, random_state=42).split(raw_values)
        )

    fold_results = {
        p: [] for p in ["match_always", "fixed_retrieval_50", "max_utility", "oracle"]
    }

    for train_val_idx, test_idx in splits:
        n_tv = len(train_val_idx)
        perm = RNG.permutation(train_val_idx)
        n_train = int(n_tv * 0.75)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        # Calibrate — same train-only calibrator for val and test
        # so threshold selected on p_val lives on the same scale as p_test
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(raw_values[train_idx].reshape(-1, 1), cb[train_idx])
        p_val = clf.predict_proba(raw_values[val_idx].reshape(-1, 1))[:, 1]
        p_test = clf.predict_proba(raw_values[test_idx].reshape(-1, 1))[:, 1]

        cb_v, ob5_v = cb[val_idx], ob5[val_idx]
        cb_t, ob5_t = cb[test_idx], ob5[test_idx]

        # Oracle
        r = oracle_upper_bound(cb_t, ob5_t)
        r["tau_skip"] = float("nan")
        r["fold"] = fold_results["oracle"].__len__()
        fold_results["oracle"].append(r)

        # match_always
        tau, _ = select_threshold_match_always(p_val, cb_v, ob5_v)
        r = evaluate_policy(p_test, tau, cb_t, ob5_t)
        r["tau_skip"] = float(tau)
        r["fold"] = len(fold_results["match_always"])
        fold_results["match_always"].append(r)

        # fixed_retrieval_50
        tau, _ = select_threshold_fixed_retrieval(p_val, 0.50)
        r = evaluate_policy(p_test, tau, cb_t, ob5_t)
        r["tau_skip"] = float(tau)
        r["fold"] = len(fold_results["fixed_retrieval_50"])
        fold_results["fixed_retrieval_50"].append(r)

        # max_utility
        best_util, best_lambda, best_tau = -np.inf, 0.0, 1.0
        for lam in LAMBDA_VALS:
            tau_cand, _ = select_threshold_max_utility(p_val, cb_v, ob5_v, lam)
            rc = evaluate_policy(p_val, tau_cand, cb_v, ob5_v)
            if rc["accuracy"] - lam * rc["retrieval_rate"] > best_util:
                best_util = rc["accuracy"] - lam * rc["retrieval_rate"]
                best_lambda, best_tau = lam, tau_cand
        r = evaluate_policy(p_test, best_tau, cb_t, ob5_t)
        r["tau_skip"] = float(best_tau)
        r["lambda_r"] = float(best_lambda)
        r["fold"] = len(fold_results["max_utility"])
        fold_results["max_utility"].append(r)

    # Aggregate
    rows = []
    for policy in ["match_always", "fixed_retrieval_50", "max_utility", "oracle"]:
        results = fold_results[policy]
        for key in ["accuracy", "retrieval_rate", "skip_rate", "tau_skip"]:
            vals = [r[key] for r in results if not np.isnan(r.get(key, np.nan))]
            if not vals:
                continue
            mu, std = np.mean(vals), np.std(vals)
            row = {
                "signal": signal_name,
                "policy": policy,
                "threshold_source": "validation"
                if policy != "oracle"
                else "oracle_test_labels",
                "metric": key,
                "mean": round(float(mu), 4),
                "std": round(float(std), 4),
                "n_folds": len(results),
            }
            if policy == "max_utility":
                row["lambda_r_mean"] = round(
                    float(np.mean([r.get("lambda_r", 0) for r in results])), 3
                )
            rows.append(row)
    return rows


def main():
    all_rows = []

    for ds_label, main_tbl, targ_tbl in CONFIGS:
        # Load main table
        main_data = [json.loads(l) for l in open(table_path(main_tbl))]
        cb = np.array([r["closed_correct"] for r in main_data], dtype=float)
        ob5 = np.array([r["open_correct_k5"] for r in main_data], dtype=float)
        seq_lp = np.array([r["seq_logprob"] for r in main_data], dtype=float)

        # Load TARG table
        targ_data = [json.loads(l) for l in open(table_path(targ_tbl))]
        entropy = np.array([r.get("mean_entropy", 0) for r in targ_data], dtype=float)
        margin = np.array(
            [r.get("top1_top2_margin", 0) for r in targ_data], dtype=float
        )
        n = min(len(cb), len(seq_lp), len(entropy))
        cb = cb[:n]
        ob5 = ob5[:n]
        seq_lp = seq_lp[:n]
        entropy = entropy[:n]
        margin = margin[:n]

        # Handle NaN in margin
        if np.any(np.isnan(margin)):
            margin = np.where(np.isnan(margin), np.nanmean(margin), margin)

        print(f"\n{'=' * 70}")
        print(f"{ds_label} | n={n} | CB={cb.mean():.3f} | OB@5={ob5.mean():.3f}")
        print(f"{'=' * 70}")
        rows = run_deployment("seq_logprob (ours)", seq_lp, cb, ob5)
        for r in rows:
            r["dataset"] = ds_label
        all_rows.extend(rows)
        rows = run_deployment("TARG entropy", -entropy, cb, ob5)
        for r in rows:
            r["dataset"] = ds_label
        all_rows.extend(rows)
        rows = run_deployment("TARG margin", margin, cb, ob5)
        for r in rows:
            r["dataset"] = ds_label
        all_rows.extend(rows)

        # Print comparison
        print(
            f"\n{'Signal':<22} {'Policy':<22} {'Acc':>7} {'RetRate':>7} {'OracleGap':>9}"
        )
        print("-" * 70)
        for sig in ["seq_logprob (ours)", "TARG entropy", "TARG margin"]:
            sig_rows = [
                r
                for r in all_rows
                if r.get("dataset") == ds_label
                and r.get("signal") == sig
                and r["metric"] == "accuracy"
                and r["policy"] != "oracle"
            ]
            oracle_rows = [
                r
                for r in all_rows
                if r.get("dataset") == ds_label
                and r.get("signal") == sig
                and r["policy"] == "oracle"
                and r["metric"] == "accuracy"
            ]
            oracle_acc = oracle_rows[0]["mean"] if oracle_rows else None
            for sr in sig_rows:
                policy = sr["policy"]
                ret_r = [
                    r
                    for r in all_rows
                    if r.get("dataset") == ds_label
                    and r.get("signal") == sig
                    and r["policy"] == policy
                    and r["metric"] == "retrieval_rate"
                ]
                ret_s = f"{ret_r[0]['mean']:.3f}" if ret_r else "N/A"
                gap = f"{oracle_acc - sr['mean']:.4f}" if oracle_acc else "N/A"
                print(f"{sig:<22} {policy:<22} {sr['mean']:>7.4f} {ret_s:>7} {gap:>9}")
    cols = [
        "dataset",
        "signal",
        "policy",
        "threshold_source",
        "metric",
        "mean",
        "std",
        "n_folds",
        "lambda_r_mean",
    ]
    path = data_path("figdata_deployment.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {path}")


if __name__ == "__main__":
    main()
