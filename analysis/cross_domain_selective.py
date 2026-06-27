"""Cross-domain selective RAG: source calibrator applied to target data."""

from __future__ import annotations
import json, os, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _tables import table_path, data_path

RNG = np.random.default_rng(42)
KFIELD = "open_seq_logprob_k5"

OPENCONF_TABLES = {
    "TriviaQA-8B": "triviaqa_rc_openconf_table.jsonl",
    "NQ-8B": "nq_dpr_openconf_table.jsonl",
    "MS-MARCO-8B": "msmarco_openconf_table.jsonl",
}


def load_openconf(name: str):
    fname = OPENCONF_TABLES.get(name)
    if not fname:
        return None
    path = table_path(fname)
    rows = [json.loads(l) for l in open(path)]
    cb = np.array([r["closed_correct"] for r in rows], dtype=float)
    ob = np.array([r["open_correct_k5"] for r in rows], dtype=float)
    slp = np.array([r.get("seq_logprob", 0) for r in rows], dtype=float)
    oslp = np.array([r.get(KFIELD, 0) for r in rows], dtype=float)
    return cb, ob, slp, oslp, len(rows)


def calibrate(signal, y, cv=5):
    ok = np.isfinite(signal) & ~np.isnan(signal)
    p = np.full(len(signal), 0.5)
    if ok.sum() < 10:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(signal[ok].reshape(-1, 1), y[ok])
        return clf, p
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p[ok] = cross_val_predict(
        clf, signal[ok].reshape(-1, 1), y[ok], cv=cv, method="predict_proba"
    )[:, 1]
    clf.fit(signal[ok].reshape(-1, 1), y[ok])
    return clf, p


def calibrate_transfer(clf, signal):
    ok = np.isfinite(signal) & ~np.isnan(signal)
    p = np.full(len(signal), 0.5)
    if ok.sum() > 0:
        try:
            p[ok] = clf.predict_proba(signal[ok].reshape(-1, 1))[:, 1]
        except Exception:
            pass
    return p


def aurc(conf, correct):
    order = np.argsort(-conf)
    c = correct[order]
    n = len(c)
    covs = np.arange(1, n + 1) / n
    risks = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return float(np.trapezoid(risks, covs)), covs, risks


def choose_tau_on_source(p_closed_src, cb_src, ob_src, policy="match_always"):
    """Select skip threshold tau on SOURCE data.
    Returns tau value. Does NOT access target data.
    """
    n = len(cb_src)
    if policy == "match_always":
        always = ob_src.mean()
        idx = np.argsort(-p_closed_src)
        best_k = 0
        for k in range(n + 1):
            skip = np.zeros(n, bool)
            skip[idx[:k]] = True
            if np.where(skip, cb_src, ob_src).mean() >= always:
                best_k = k
        if best_k == 0:
            return 1.0  # skip nothing
        return float(p_closed_src[idx[best_k - 1]])
    elif policy == "fixed_skip_30pct":
        return float(np.quantile(p_closed_src, 0.70))  # skip top 30%
    elif policy == "fixed_skip_50pct":
        return float(np.quantile(p_closed_src, 0.50))
    else:
        raise ValueError(f"Unknown policy: {policy}")


def evaluate_with_tau(cb, ob, p_closed, p_open, tau_skip, threshold_protocol=""):
    """Evaluate selective RAG given a pre-selected tau (from source or oracle).
    Uses target labels ONLY for evaluation, NOT for threshold selection.
    """
    n = len(cb)
    skip = p_closed >= tau_skip
    ans_correct = np.where(skip, cb, ob)
    chosen_conf = np.where(skip, p_closed, p_open)
    ap, _, _ = aurc(chosen_conf, ans_correct)
    an, _, _ = aurc(p_closed, ans_correct)
    ar = float(1 - ans_correct.mean())
    k80 = int(round(0.8 * n))
    ordr = np.argsort(-chosen_conf)[:k80]
    acc80 = ans_correct[ordr].mean()
    return {
        "tau_skip": tau_skip,
        "skip_rate": float(skip.mean()),
        "retrieval_rate": float(1 - skip.mean()),
        "full_acc": float(ans_correct.mean()),
        "aurc_proper": ap,
        "aurc_naive": an,
        "aurc_random": ar,
        "acc_at_80pct": acc80,
        "threshold_protocol": threshold_protocol,
    }


def main():
    out_csv = data_path("figdata_cross_selective.csv")
    w = csv.writer(open(str(out_csv), "w", newline=""))
    w.writerow(
        [
            "source",
            "target",
            "threshold_protocol",
            "method",
            "tau_skip",
            "skip_rate",
            "retrieval_rate",
            "aurc",
            "acc_at_80pct",
            "full_acc",
        ]
    )

    print("=" * 90)
    print("CROSS-DOMAIN SELECTIVE RAG — with source-threshold transfer")
    print("=" * 90)

    print("\n--- In-domain OOF baselines ---")
    in_domain = {}
    for ds_name in OPENCONF_TABLES:
        loaded = load_openconf(ds_name)
        if loaded is None:
            print(f"  MISSING {ds_name}")
            continue
        cb, ob, slp, oslp, n = loaded
        clf_cb, p_closed_id = calibrate(slp, cb)
        clf_ob, p_open_id = calibrate(oslp, ob)

        # Oracle threshold (uses same-domain labels — this IS valid in-domain)
        tau_oracle = choose_tau_on_source(p_closed_id, cb, ob, "match_always")
        res = evaluate_with_tau(
            cb, ob, p_closed_id, p_open_id, tau_oracle, "in_domain_oof_oracle"
        )
        print(
            f"\n{ds_name} (in-domain oracle): proper={res['aurc_proper']:.4f} "
            f"naive={res['aurc_naive']:.4f} random={res['aurc_random']:.4f}"
        )

        in_domain[ds_name] = {
            "cb": cb,
            "ob": ob,
            "slp": slp,
            "oslp": oslp,
            "clf_cb": clf_cb,
            "clf_ob": clf_ob,
            "p_closed_id": p_closed_id,
            "p_open_id": p_open_id,
            **res,
        }
        for m, aurc_val in [
            ("proper", res["aurc_proper"]),
            ("naive", res["aurc_naive"]),
            ("random", res["aurc_random"]),
        ]:
            w.writerow(
                [
                    ds_name,
                    ds_name,
                    "in_domain_oof_oracle",
                    m,
                    round(tau_oracle, 4),
                    round(res["skip_rate"], 4),
                    round(res["retrieval_rate"], 4),
                    round(aurc_val, 4),
                    round(res["acc_at_80pct"], 4),
                    round(res["full_acc"], 4),
                ]
            )

    print("\n--- Cross-domain: SOURCE-THRESHOLD transfer ---")
    pairs = [
        ("TriviaQA-8B", "NQ-8B"),
        ("NQ-8B", "TriviaQA-8B"),
        ("TriviaQA-8B", "MS-MARCO-8B"),
        ("NQ-8B", "MS-MARCO-8B"),
    ]

    for src, tgt in pairs:
        if src not in in_domain or tgt not in in_domain:
            continue
        src_d = in_domain[src]
        tgt_d = in_domain[tgt]

        # Select tau on SOURCE ONLY
        tau_src = choose_tau_on_source(
            src_d["p_closed_id"], src_d["cb"], src_d["ob"], "match_always"
        )

        # Apply source calibrators to target data
        p_closed_x = calibrate_transfer(src_d["clf_cb"], tgt_d["slp"])
        p_open_x = calibrate_transfer(src_d["clf_ob"], tgt_d["oslp"])

        # Evaluate on target with source-selected tau (NO target labels for threshold)
        res_x = evaluate_with_tau(
            tgt_d["cb"],
            tgt_d["ob"],
            p_closed_x,
            p_open_x,
            tau_src,
            "source_threshold_match_always",
        )

        # Oracle upper bound: select tau on target labels (old approach, for reference)
        tau_oracle_tgt = choose_tau_on_source(
            p_closed_x, tgt_d["cb"], tgt_d["ob"], "match_always"
        )
        res_oracle = evaluate_with_tau(
            tgt_d["cb"],
            tgt_d["ob"],
            p_closed_x,
            p_open_x,
            tau_oracle_tgt,
            "oracle_target_threshold",
        )

        print(f"\n{src} → {tgt}:")
        print(
            f"  SOURCE-THRESHOLD (no target labels): proper={res_x['aurc_proper']:.4f} "
            f"naive={res_x['aurc_naive']:.4f}"
        )
        print(
            f"  ORACLE-TARGET (upper bound):          proper={res_oracle['aurc_proper']:.4f} "
            f"naive={res_oracle['aurc_naive']:.4f}"
        )
        print(f"  Target in-domain proper (reference):   {tgt_d['aurc_proper']:.4f}")
        better = "YES ★" if res_x["aurc_proper"] < tgt_d["aurc_naive"] else "no"
        print(f"  Source-threshold proper < target naive? {better}")

        for proto, res in [
            ("source_threshold_match_always", res_x),
            ("oracle_target_threshold", res_oracle),
        ]:
            for m, aurc_val in [
                ("proper", res["aurc_proper"]),
                ("naive", res["aurc_naive"]),
            ]:
                w.writerow(
                    [
                        src,
                        tgt,
                        proto,
                        m,
                        round(
                            tau_src if proto.startswith("source") else tau_oracle_tgt, 4
                        ),
                        round(res["skip_rate"], 4),
                        round(res["retrieval_rate"], 4),
                        round(aurc_val, 4),
                        round(res["acc_at_80pct"], 4),
                        round(res["full_acc"], 4),
                    ]
                )

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
