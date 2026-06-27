"""Figure 9: entity popularity vs closed-book correctness (PopQA)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from _tables import table_path
from figure_style import set_style, style_axes, save


def main():
    rows = [json.loads(l) for l in open(table_path("popqa_table.jsonl")) if l.strip()]
    pop = np.array([r["popularity"] for r in rows])
    pc = np.array([r["p_correct"] for r in rows])
    cc = np.array([r["closed_correct"] for r in rows])

    # Use log10(1+popularity) to avoid symlog negative ticks
    x = np.log10(1 + pop)

    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    wrong = cc == 0
    right = cc == 1
    ax.scatter(
        x[wrong],
        pc[wrong],
        c="#d62728",
        alpha=0.7,
        s=18,
        linewidths=0.4,
        edgecolors="white",
        label="Closed-book wrong",
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        x[right],
        pc[right],
        c="#2ca02c",
        alpha=0.85,
        s=20,
        linewidths=0.4,
        edgecolors="white",
        label="Closed-book correct",
        rasterized=True,
        zorder=3,
    )

    # Quartile lines
    for q in [25, 50, 75]:
        ax.axvline(np.percentile(x, q), color="#aaaaaa", lw=0.7, ls="--")

    ax.set_xlabel("Entity popularity  $\\log_{10}(1+\\mathrm{PageViews})$", fontsize=9)
    ax.set_ylabel("Gate confidence  $P(\\mathrm{correct})$", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    style_axes(ax)
    ax.tick_params(axis="both", labelsize=8)

    # Annotation — bottom-left
    ax.text(
        0.68,
        0.85,
        f"AUROC = 0.826\n$\\rho_{{pop,conf}}$ = 0.082  ($p$=0.009)\n$n$ = {len(rows)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    fig.tight_layout(pad=0.4)
    save(fig, "fig9_motivation")
    print("saved figures/fig9_motivation.pdf")


if __name__ == "__main__":
    main()
