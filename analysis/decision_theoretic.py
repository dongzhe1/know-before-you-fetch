"""Decision-theoretic policy: argmax_a U(a|x) with held-out lambda."""

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
    ("TriviaQA-8B", "triviaqa_rc_table.jsonl"),
    ("NQ-8B", "nq_dpr_table.jsonl"),
    ("MS-MARCO-8B", "msmarco_table.jsonl"),
]

N_OUTER = 5
RNG = np.random.default_rng(42)

# Cost parameters (from timing_raw.csv, H100 bf16)
# 8B: c_CB=65ms, c_OB1=61ms, c_OB5=80ms
# 32B: c_CB=162ms, c_OB1=257ms, c_OB5=319ms
COST_8B = {"cb": 65.0, "ob1": 61.4, "ob5": 80.0, "abstain_penalty": 0.0}
COST_32B = {"cb": 162.0, "ob1": 257.0, "ob5": 319.0, "abstain_penalty": 0.0}

# Utility λ sweep range
LAMBDA_CALL = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]  # cost per retrieval call
LAMBDA_TOK = [0.0, 0.001, 0.005, 0.01]  # cost per 100 input tokens
GAMMA_ABSTAIN = [0.0, 0.05, 0.10]  # abstain penalty (opportunity cost)


def calibrate(train_X, train_y, test_X):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(train_X.reshape(-1, 1), train_y)
    return clf.predict_proba(test_X.reshape(-1, 1))[:, 1]


def decision_theoretic_policy(p_cb, p_ob1, p_ob5, costs, lam_call, lam_tok, gamma):
    """
    For each query, compute utility of each action and select argmax.
    p_cb:  P(closed-book correct | x)
    p_ob1: P(open-book@k=1 correct | x)
    p_ob5: P(open-book@k=5 correct | x)

    U(CB)   = p_cb  - lam_call*0 - lam_tok*(cost["cb"]/100)
    U(OB1)  = p_ob1 - lam_call*1 - lam_tok*(cost["ob1"]/100)
    U(OB5)  = p_ob5 - lam_call*1 - lam_tok*(cost["ob5"]/100)
    U(ABS)  = 0     - lam_call*0 - lam_tok*0           - gamma
    """
    n = len(p_cb)
    u_cb = p_cb - lam_tok * (costs["cb"] / 100)
    u_ob1 = p_ob1 - lam_call - lam_tok * (costs["ob1"] / 100)
    u_ob5 = p_ob5 - lam_call - lam_tok * (costs["ob5"] / 100)
    u_abs = np.full(n, -gamma)

    U = np.column_stack([u_cb, u_ob1, u_ob5, u_abs])
    actions = np.argmax(U, axis=1)  # 0=CB, 1=OB1, 2=OB5, 3=ABSTAIN

    return actions


def evaluate_actions(actions, cb, ob1, ob5):
    """Compute realized accuracy and cost metrics for selected actions."""
    n = len(actions)
    # Realized accuracy: use ground-truth outcome for the chosen action
    realized_correct = np.where(
        actions == 0, cb, np.where(actions == 1, ob1, np.where(actions == 2, ob5, 0))
    )  # abstain = always wrong (conservative)
    acc = realized_correct.mean()

    ret_rate = ((actions == 1) | (actions == 2)).mean()
    k1_rate = (actions == 1).mean()
    k5_rate = (actions == 2).mean()
    abs_rate = (actions == 3).mean()
    cb_rate = (actions == 0).mean()

    return {
        "accuracy": float(acc),
        "retrieval_rate": float(ret_rate),
        "k1_rate": float(k1_rate),
        "k5_rate": float(k5_rate),
        "abstain_rate": float(abs_rate),
        "closed_rate": float(cb_rate),
    }


def main():
    all_rows = []

    for ds_label, tbl_name in CONFIGS:
        data = [json.loads(l) for l in open(table_path(tbl_name))]
        cb = np.array([r["closed_correct"] for r in data], dtype=float)
        ob5 = np.array([r["open_correct_k5"] for r in data], dtype=float)
        ob1_col = (
            "open_correct_k1" if "open_correct_k1" in data[0] else "open_correct_k5"
        )
        ob1 = np.array(
            [r.get(ob1_col, r["open_correct_k5"]) for r in data], dtype=float
        )
        seq_lp = np.array([r["seq_logprob"] for r in data], dtype=float)
        costs = COST_8B if "8B" in ds_label else COST_32B
        n = len(cb)

        print(f"\n{'=' * 70}")
        print(
            f"{ds_label} | n={n} | CB={cb.mean():.3f} | OB@1={ob1.mean():.3f} | OB@5={ob5.mean():.3f}"
        )
        print(f"Costs: CB={costs['cb']}ms  OB1={costs['ob1']}ms  OB5={costs['ob5']}ms")
        print(f"{'=' * 70}")

        # Stratified outer CV
        try:
            skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
            splits = list(skf.split(seq_lp, cb.astype(int)))
        except ValueError:
            splits = list(
                KFold(n_splits=N_OUTER, shuffle=True, random_state=42).split(seq_lp)
            )

        best_configs = []  # track best (λ_call, λ_tok, γ) per fold

        for fold_i, (train_val_idx, test_idx) in enumerate(splits):
            n_tv = len(train_val_idx)
            perm = RNG.permutation(train_val_idx)
            n_train = int(n_tv * 0.75)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            # P(CB correct | seq_logprob)
            p_cb_val = calibrate(seq_lp[train_idx], cb[train_idx], seq_lp[val_idx])
            p_cb_test = calibrate(
                np.concatenate([seq_lp[train_idx], seq_lp[val_idx]]),
                np.concatenate([cb[train_idx], cb[val_idx]]),
                seq_lp[test_idx],
            )

            # P(OB@1 correct | seq_logprob) — same signal, different outcome
            p_ob1_val = calibrate(seq_lp[train_idx], ob1[train_idx], seq_lp[val_idx])
            p_ob1_test = calibrate(
                np.concatenate([seq_lp[train_idx], seq_lp[val_idx]]),
                np.concatenate([ob1[train_idx], ob1[val_idx]]),
                seq_lp[test_idx],
            )

            # P(OB@5 correct | seq_logprob)
            p_ob5_val = calibrate(seq_lp[train_idx], ob5[train_idx], seq_lp[val_idx])
            p_ob5_test = calibrate(
                np.concatenate([seq_lp[train_idx], seq_lp[val_idx]]),
                np.concatenate([ob5[train_idx], ob5[val_idx]]),
                seq_lp[test_idx],
            )

            cb_v, ob1_v, ob5_v = cb[val_idx], ob1[val_idx], ob5[val_idx]
            cb_t, ob1_t, ob5_t = cb[test_idx], ob1[test_idx], ob5[test_idx]
            best_util = -np.inf
            best_params = (0.0, 0.0, 0.0)
            for lam_c in LAMBDA_CALL:
                for lam_t in LAMBDA_TOK:
                    for gam in GAMMA_ABSTAIN:
                        acts = decision_theoretic_policy(
                            p_cb_val, p_ob1_val, p_ob5_val, costs, lam_c, lam_t, gam
                        )
                        ev = evaluate_actions(acts, cb_v, ob1_v, ob5_v)
                        utility = (
                            ev["accuracy"]
                            - lam_c * ev["retrieval_rate"]
                            - lam_t
                            * (
                                ev["k1_rate"] * costs["ob1"]
                                + ev["k5_rate"] * costs["ob5"]
                            )
                            / 100
                            - gam * ev["abstain_rate"]
                        )
                        if utility > best_util:
                            best_util = utility
                            best_params = (lam_c, lam_t, gam)
            best_configs.append(best_params)
            acts_test = decision_theoretic_policy(
                p_cb_test, p_ob1_test, p_ob5_test, costs, *best_params
            )
            ev_test = evaluate_actions(acts_test, cb_t, ob1_t, ob5_t)
            ev_test["fold"] = fold_i
            ev_test["lambda_call"] = best_params[0]
            ev_test["lambda_tok"] = best_params[1]
            ev_test["gamma_abstain"] = best_params[2]
            all_rows.append(
                {
                    "dataset": ds_label,
                    "fold": fold_i,
                    **{
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in ev_test.items()
                    },
                    **{
                        k: round(v, 4)
                        for k, v in zip(
                            ["lambda_call", "lambda_tok", "gamma_abstain"], best_params
                        )
                    },
                }
            )
        ds_rows = [r for r in all_rows if r["dataset"] == ds_label]
        print(
            f"\n  {'Policy':<25} {'Acc':>7} {'Ret':>7} {'K1':>7} {'K5':>7} {'Abs':>7} {'λ_call':>7} {'λ_tok':>7}"
        )
        print(
            f"  {'-' * 25} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}"
        )
        keys = [
            "accuracy",
            "retrieval_rate",
            "k1_rate",
            "k5_rate",
            "abstain_rate",
            "lambda_call",
            "lambda_tok",
        ]
        means = {k: np.mean([r[k] for r in ds_rows]) for k in keys}
        print(
            f"  {'decision-theoretic':<25} {means['accuracy']:>7.4f} {means['retrieval_rate']:>7.3f} "
            f"{means['k1_rate']:>7.3f} {means['k5_rate']:>7.3f} {means['abstain_rate']:>7.3f} "
            f"{means['lambda_call']:>7.3f} {means['lambda_tok']:>7.3f}"
        )

        # Also compute oracle (upper bound using true outcomes)
        oracle_acc = float(np.maximum(np.maximum(cb, ob1), ob5).mean())
        print(f"  {'oracle upper bound':<25} {oracle_acc:>7.4f}")
        print(f"  {'always-RAG (k=5)':<25} {ob5.mean():>7.4f}")
        print(f"  {'never-RAG (CB)':<25} {cb.mean():>7.4f}")
    path = data_path("figdata_decision_theoretic.csv")
    cols = [
        "dataset",
        "fold",
        "accuracy",
        "retrieval_rate",
        "k1_rate",
        "k5_rate",
        "abstain_rate",
        "closed_rate",
        "lambda_call",
        "lambda_tok",
        "gamma_abstain",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {path}")


if __name__ == "__main__":
    main()
