"""Three-action utility sweep: skip / retrieve / abstain."""

from __future__ import annotations
import json, os, sys, csv
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _tables import resolve

DATASETS = [
    ("triviaqa_rc_openconf_table.jsonl", "TriviaQA-8B"),
    ("nq_dpr_openconf_table.jsonl", "NQ-8B"),
    ("msmarco_openconf_table.jsonl", "MS-MARCO-8B"),
]
KFIELD = "open_seq_logprob_k5"

# Break-even cost ratios from timing benchmark (c_CB / (c_ret + c_OB)):
# ρ* = 1.62 for 8B (gate saves only if retrieval costs > 1.62× generation)
# In practice with cheap retrieval, λ_r is small — we sweep it.
LAMBDA_R_VALS = [0.0, 0.05, 0.10, 0.20, 0.30]  # retrieval cost weight
LAMBDA_A_VALS = [0.0, 0.05, 0.10, 0.20]  # abstain cost (opportunity cost)


def calibrate_oof(X, y):
    return cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        X.reshape(-1, 1),
        y,
        cv=5,
        method="predict_proba",
    )[:, 1]


def three_action_sweep(p_skip, p_open, ans_correct, cb, ob, lambda_r, lambda_a):
    """Sweep τ_skip and τ_abstain; return best utility and its operating point."""
    n = len(cb)
    taus_skip = np.percentile(p_skip, np.arange(0, 101, 5))
    taus_abs = np.percentile(p_skip, np.arange(0, 101, 5))  # same signal, different cut

    best = dict(utility=-999)
    for tau_s in taus_skip:
        # gate: skip if p_skip >= tau_s, retrieve otherwise
        skip_mask = p_skip >= tau_s
        chosen_correct = np.where(skip_mask, cb, ob)
        chosen_conf = np.where(skip_mask, p_skip, p_open)
        ret_rate = 1 - skip_mask.mean()

        for tau_a in taus_abs:
            # abstain: refuse if chosen_conf < tau_a
            abstain_mask = chosen_conf < tau_a
            answered_correct = chosen_correct[~abstain_mask]
            abs_rate = abstain_mask.mean()
            if abs_rate >= 1.0:
                continue

            acc = answered_correct.mean() if len(answered_correct) > 0 else 0.0
            # coverage-weighted: we care about accuracy on answered queries;
            # abstaining costs opportunity (lambda_a per abstained query)
            utility = acc * (1 - abs_rate) - lambda_r * ret_rate - lambda_a * abs_rate

            if utility > best["utility"]:
                best = dict(
                    utility=utility,
                    acc=acc,
                    ret_rate=ret_rate,
                    abs_rate=abs_rate,
                    tau_s=float(tau_s),
                    tau_a=float(tau_a),
                    full_acc=chosen_correct.mean(),
                )
    return best


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out_rows = []

    for fname, label in DATASETS:
        p = resolve(fname)
        if not os.path.exists(p):
            print(f"MISSING {fname}")
            continue
        data = [json.loads(l) for l in open(p)]
        if KFIELD not in data[0]:
            print(f"{fname}: no {KFIELD}")
            continue

        cb = np.array([r["closed_correct"] for r in data], float)
        ob = np.array([r["open_correct_k5"] for r in data], float)
        lp = np.array([r["seq_logprob"] for r in data])
        olp = np.array([r[KFIELD] for r in data])

        p_skip = calibrate_oof(lp, cb)
        p_open = calibrate_oof(olp, ob)

        baselines = {
            "always-retrieve": dict(acc=ob.mean(), ret_rate=1.0, abs_rate=0.0),
            "always-skip": dict(acc=cb.mean(), ret_rate=0.0, abs_rate=0.0),
        }

        print(f"\n=== {label} ===")
        print(
            f"  Baseline always-retrieve acc={ob.mean():.3f}  always-skip acc={cb.mean():.3f}"
        )
        print(
            f"  {'λ_r':>5} {'λ_a':>5} | {'Utility':>8} {'Acc':>7} {'RetRate':>8} {'AbsRate':>8} {'FullAcc':>8}"
        )
        print("  " + "-" * 60)

        for lr in LAMBDA_R_VALS:
            for la in LAMBDA_A_VALS:
                res = three_action_sweep(
                    p_skip, p_open, np.where(p_skip >= 0.5, cb, ob), cb, ob, lr, la
                )
                print(
                    f"  {lr:>5.2f} {la:>5.2f} | {res['utility']:>8.4f} {res['acc']:>7.3f} "
                    f"{res['ret_rate']:>8.3f} {res['abs_rate']:>8.3f} {res['full_acc']:>8.3f}"
                )
                out_rows.append(dict(dataset=label, lambda_r=lr, lambda_a=la, **res))

    with open(
        os.path.join(HERE, "data", "figdata_unified_decision.csv"), "w", newline=""
    ) as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(f"\nwrote paper/data/figdata_unified_decision.csv")


if __name__ == "__main__":
    main()
