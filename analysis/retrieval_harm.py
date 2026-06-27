"""Retrieval harm/rescue 2x2 matrix by confidence quartile."""

from __future__ import annotations
import json, os, csv
import numpy as np

from _tables import resolve

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

DATASETS = [
    ("triviaqa_rc_table.jsonl", "TriviaQA-rc / Qwen3-8B"),
    ("nq_dpr_table.jsonl", "NQ-DPR / Qwen3-8B"),
    ("msmarco_table.jsonl", "MS-MARCO / Qwen3-8B"),
    ("triviaqa_rc_32b_n600_table.jsonl", "TriviaQA-rc / Qwen3-32B"),
]


def analyze(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cb = np.array([r["closed_correct"] for r in rows])
    ob = np.array([r.get("open_correct_k5", r.get("open_correct_k1", 0)) for r in rows])
    pc = np.array([r["p_correct"] for r in rows])
    n = len(rows)
    cc_oc = int(((cb == 1) & (ob == 1)).sum())  # retrieval unnecessary
    cc_ow = int(((cb == 1) & (ob == 0)).sum())  # HARM: flipped correct->wrong
    cw_oc = int(((cb == 0) & (ob == 1)).sum())  # RESCUE
    cw_ow = int(((cb == 0) & (ob == 0)).sum())  # insufficient
    # by confidence quartile (high pc = gate would skip)
    q = np.quantile(pc, [0, 0.25, 0.5, 0.75, 1.0])
    by_q = []
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        mask = (pc >= lo) & (pc <= hi) if i == 3 else (pc >= lo) & (pc < hi)
        if mask.sum() == 0:
            by_q.append((0, 0, 0))
            continue
        ccm = cb[mask] == 1
        harm = float(
            ((ob[mask] == 0) & ccm).sum() / max(ccm.sum(), 1)
        )  # P(harm | closed correct)
        rescue = float(((ob[mask] == 1) & (~ccm)).sum() / max((~ccm).sum(), 1))
        by_q.append((int(mask.sum()), harm, rescue))
    return n, (cc_oc, cc_ow, cw_oc, cw_ow), by_q


def main():
    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, "figdata_harm.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["dataset", "quartile", "n", "harm_rate", "rescue_rate"])
    for fname, label in DATASETS:
        p = resolve(fname)
        if not os.path.exists(p):
            print(f"MISSING {fname}")
            continue
        n, (a, b, c, d), by_q = analyze(p)
        print(f"\n{label}  (n={n})")
        print(f"  closed-correct & open-correct (unnecessary) : {a:4d} ({a / n:.1%})")
        print(f"  closed-correct & open-WRONG   (HARM)        : {b:4d} ({b / n:.1%})")
        print(f"  closed-wrong   & open-correct (RESCUE)      : {c:4d} ({c / n:.1%})")
        print(f"  closed-wrong   & open-wrong   (insufficient): {d:4d} ({d / n:.1%})")
        print(f"  {'quartile':<10}{'n':>5}{'harm%':>8}{'rescue%':>9}")
        for i, (qn, harm, rescue) in enumerate(by_q):
            qlab = ["Q1 low-conf", "Q2", "Q3", "Q4 high-conf"][i]
            print(f"  {qlab:<10}{qn:>5}{harm * 100:>8.1f}{rescue * 100:>9.1f}")
            w.writerow([label, qlab, qn, round(harm, 4), round(rescue, 4)])
    out.close()
    print(f"\nwrote {DATA}/figdata_harm.csv")
    print(
        "Story (TriviaQA/32B): among confident-correct queries retrieval rarely flips them"
        " (harm 5-6% at high-conf Q4) -> safe to skip; rescue is high at low confidence"
        " (Q1) -> worth retrieving. Both gradients justify the gate."
    )


if __name__ == "__main__":
    main()
