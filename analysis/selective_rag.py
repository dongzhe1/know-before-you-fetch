"""Selective RAG: proper chosen-answer confidence vs naive closed-book abstention."""

from __future__ import annotations
import json, os, csv
import numpy as np
from _tables import resolve

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
DATASETS = [
    ("triviaqa_rc_openconf_table.jsonl", "TriviaQA-8B"),
    ("nq_dpr_openconf_table.jsonl", "NQ-8B"),
    ("msmarco_openconf_table.jsonl", "MS-MARCO-8B"),
]
KFIELD = "open_seq_logprob_k5"


def aurc(conf, correct):
    """Area under risk-coverage curve; answer most-confident first. Lower is better."""
    order = np.argsort(-conf)
    c = correct[order]
    n = len(c)
    covs = np.arange(1, n + 1) / n
    risks = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return float(np.trapezoid(risks, covs)), covs, risks


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, "figdata_selective.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["dataset", "method", "coverage", "risk"])
    any_found = False
    for fname, label in DATASETS:
        p = resolve(fname)
        if not os.path.exists(p):
            print(f"MISSING {fname} (run frontier probe with OPENCONF=1, TAG=openconf)")
            continue
        rows = [json.loads(l) for l in open(p)]
        if KFIELD not in rows[0]:
            print(f"{fname}: no {KFIELD} (re-run with --dump_open_conf)")
            continue
        any_found = True
        cb = np.array([r["closed_correct"] for r in rows])
        ob = np.array([r["open_correct_k5"] for r in rows])
        p_closed = np.array([r["p_correct"] for r in rows])
        oseqlp = np.array([r[KFIELD] for r in rows])
        # calibrate open-book confidence OOF
        p_open = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            oseqlp.reshape(-1, 1),
            ob,
            cv=5,
            method="predict_proba",
        )[:, 1]
        # gate decision: skip (answer closed) if p_closed >= tau*, else retrieve (answer open)
        # tau* = threshold matching always-RAG accuracy (same as main paper)
        idx = np.argsort(-p_closed)
        always = ob.mean()
        best_k = 0
        for k in range(len(rows) + 1):
            skip = np.zeros(len(rows), bool)
            skip[idx[:k]] = True
            if np.where(skip, cb, ob).mean() >= always:
                best_k = k
        skip = np.zeros(len(rows), bool)
        skip[idx[:best_k]] = True
        ans_correct = np.where(skip, cb, ob)
        chosen_conf = np.where(
            skip, p_closed, p_open
        )  # proper: confidence of the chosen answer
        rng = np.random.default_rng(0)

        a_proper, cov, risk_p = aurc(chosen_conf, ans_correct)
        a_naive, _, risk_n = aurc(p_closed, ans_correct)
        a_rand = float(np.trapezoid(np.full_like(cov, 1 - ans_correct.mean()), cov))
        print(f"\n{label}: full-coverage acc={ans_correct.mean():.3f}")
        print(
            f"  AURC proper(chosen-conf)={a_proper:.4f}  naive(closed-conf)={a_naive:.4f}  "
            f"random={a_rand:.4f}"
        )
        verdict = (
            "BETTER than naive & random"
            if a_proper < min(a_naive, a_rand)
            else ("beats random only" if a_proper < a_rand else "no gain")
        )
        print(f"  -> {verdict}")
        # selling point: accuracy on the answered 90%/80%
        for c in (0.9, 0.8):
            k = int(round(c * len(rows)))
            ordr = np.argsort(-chosen_conf)[:k]
            print(
                f"  coverage {c:.0%}: acc@answered = {ans_correct[ordr].mean():.3f} "
                f"(vs full {ans_correct.mean():.3f})"
            )
        stride = max(1, len(cov) // 20)
        for cc, rr in zip(cov[::stride], risk_p[::stride]):
            w.writerow([label, "proper", round(float(cc), 4), round(float(rr), 4)])
        _, _, risk_n2 = aurc(p_closed, ans_correct)
        for cc, rr in zip(cov[::stride], risk_n2[::stride]):
            w.writerow([label, "naive", round(float(cc), 4), round(float(rr), 4)])
        rand_risk = 1 - ans_correct.mean()
        for cc in cov[::stride]:
            w.writerow([label, "random", round(float(cc), 4), round(rand_risk, 4)])
    out.close()

    # Write summary CSV with verdicts — compute directly from full-resolution data
    import csv as csv2

    summary_path = os.path.join(DATA, "selective_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        sw = csv2.writer(sf)
        sw.writerow(
            [
                "dataset",
                "method",
                "aurc",
                "full_acc",
                "acc_at_80pct",
                "proper_vs_naive",
                "proper_vs_random",
                "naive_vs_random",
            ]
        )
        for fname, label in DATASETS:
            ds_rows = [
                r
                for r in csv.DictReader(
                    open(os.path.join(DATA, "figdata_selective.csv"))
                )
                if r["dataset"] == label
            ]

            def aurc_from_rows(method):
                mr = [
                    (float(r["coverage"]), float(r["risk"]))
                    for r in ds_rows
                    if r["method"] == method
                ]
                if not mr:
                    return float("nan")
                covs = np.array([c for c, _ in sorted(mr)])
                risks = np.array([r for _, r in sorted(mr)])
                return float(np.trapezoid(risks, covs))

            ap = aurc_from_rows("proper")
            an = aurc_from_rows("naive")
            ar = aurc_from_rows("random")
            for method, aurc_val in [("proper", ap), ("naive", an), ("random", ar)]:
                pvn = "better" if ap < an else ("worse" if ap > an else "tie")
                pvr = "better" if ap < ar else ("worse" if ap > ar else "tie")
                nvr = "better" if an < ar else ("worse" if an > ar else "tie")
                sw.writerow([label, method, round(aurc_val, 4), "", "", pvn, pvr, nvr])
    print(f"wrote {summary_path}")
    if any_found:
        print(f"wrote {DATA}/figdata_selective.csv")
    else:
        print("\nNo open-conf tables yet.")


if __name__ == "__main__":
    main()
