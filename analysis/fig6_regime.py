"""Figure 6: cost regime map with break-even curves."""

from __future__ import annotations
import json, os, csv
import numpy as np
import matplotlib.pyplot as plt
from _tables import resolve
from figure_style import set_style, style_axes, save, SCALE_COLORS

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

# measured generation latency (ms): (c_cb, c_ob) from timing_benchmark.py on H100.
# 1.7B: linearly scaled from 8B by parameter ratio (no direct measurement).
GEN = {
    "Qwen3-1.7B": (13.7, 17.0),
    "Qwen3-8B": (64.6, 80.0),
    "Qwen3-32B": (161.1, 319.0),
}
CFG = [
    ("Qwen3-1.7B", "triviaqa_rc_qwen1p7b_table.jsonl"),
    ("Qwen3-8B", "triviaqa_rc_table.jsonl"),
    ("Qwen3-32B", "triviaqa_rc_32b_n600_table.jsonl"),
]


def skip_at_match(path):
    R = [json.loads(l) for l in open(path)]
    pc = np.array([r["p_correct"] for r in R])
    cb = np.array([r["closed_correct"] for r in R])
    ob = np.array([r.get("open_correct_k5", r.get("open_correct_k1", 0)) for r in R])
    n = len(R)
    always = ob.mean()
    idx = np.argsort(-pc)
    best = 0.0
    for k in range(n + 1):
        skip = np.zeros(n, bool)
        skip[idx[:k]] = True
        if np.where(skip, cb, ob).mean() >= always:
            best = k / n
    return best, float(cb.mean())


def main():
    set_style()
    os.makedirs(DATA, exist_ok=True)
    rows = []
    for model, fname in CFG:
        p = resolve(fname)
        if not os.path.exists(p):
            print(f"MISSING {fname}")
            continue
        sk, cbacc = skip_at_match(p)
        ccb, cob = GEN[model]
        # rho* : min retrieval/generation ratio for the gate to save total cost
        rho_star = (ccb / (sk * cob) - 1) if sk > 0 else float("inf")
        rows.append((model, cbacc, sk, ccb, cob, rho_star))
        print(
            f"{model:<12} CB={cbacc:.3f} skip@match={sk:.2f}  c_cb/c_ob={ccb / cob:.2f}  "
            f"rho*={rho_star:.2f}"
        )

    with open(os.path.join(DATA, "figdata_regime.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cb_acc", "skip_at_match", "c_cb", "c_ob", "rho_star"])
        for r in rows:
            w.writerow(
                [
                    r[0],
                    round(r[1], 4),
                    round(r[2], 4),
                    r[3],
                    r[4],
                    round(r[5], 4) if np.isfinite(r[5]) else "inf",
                ]
            )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    rho = np.logspace(-1.3, 1.3, 200)  # 0.05 .. 20
    colors = {"Qwen3-1.7B": "#9ecae1", "Qwen3-8B": "#3182bd", "Qwen3-32B": "#08519c"}

    # Shade the "gate saves" region (above the highest curve) and "costs more" region
    all_thr = np.zeros((len(rows), len(rho)))
    for i, (model, cbacc, sk, ccb, cob, rho_star) in enumerate(rows):
        all_thr[i] = ccb / ((1 + rho) * cob)
    lo_thr = np.min(all_thr, axis=0)  # easiest to beat (lowest threshold)
    hi_thr = np.max(all_thr, axis=0)  # hardest to beat

    # Above hi_thr: saves for ALL models (green)
    ax.fill_between(rho, hi_thr, 1.0, alpha=0.12, color="#2ca02c")
    ax.text(
        0.15,
        0.88,
        "gate SAVES\n(for all models)",
        fontsize=7.5,
        color="#1a7a1a",
        transform=ax.transAxes,
    )
    # Below lo_thr: costs more for ALL models (red)
    ax.fill_between(rho, 0, lo_thr, alpha=0.10, color="#d62728")
    ax.text(
        0.30,
        0.12,
        "gate COSTS MORE\n(for all models)",
        fontsize=7.5,
        color="#aa2222",
        transform=ax.transAxes,
    )

    for model, cbacc, sk, ccb, cob, rho_star in rows:
        thr = ccb / ((1 + rho) * cob)
        ax.plot(
            rho, thr, "-", color=colors.get(model, "#888"), lw=2.0, label=f"{model}"
        )
        # operating point: at default rho=0.1 (cheap retrieval)
        ax.scatter(
            [0.1],
            [sk],
            color=colors.get(model, "#888"),
            s=100,
            edgecolor="black",
            zorder=5,
            marker="o",
            linewidths=1.2,
        )
        ax.annotate(
            f"{model}\nskip={sk * 100:.0f}%",
            (0.1, sk),
            textcoords="offset points",
            xytext=(14, 0),
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"retrieval/generation cost ratio  $\rho = c_{\mathrm{ret}}/c_{\mathrm{ob}}$",
        fontsize=11,
    )
    ax.set_ylabel("skip rate (at matched accuracy)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Gate saves total cost only above its break-even curve", fontsize=11, loc="left"
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, title="Break-even curve")
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig6_regime")
    print(f"\nwrote {DATA}/figdata_regime.csv + figures/fig6_regime.pdf")


if __name__ == "__main__":
    main()
