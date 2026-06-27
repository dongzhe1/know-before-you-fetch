#!/usr/bin/env python3
"""Figure 2: gate quality and skip rate vs model scale."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import DATA, SCALE_COLORS, save, set_style, style_axes


def main() -> None:
    set_style()
    df = pd.read_csv(DATA / "figdata_scaling.csv")
    df["params_b"] = df["params_b"].astype(float)
    df = df.sort_values("params_b")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.0))
    x = [float(v) for v in df["params_b"]]

    # (a) the three quality signals vs scale (log-x)
    axA.plot(
        x, df["cb_acc"], "-o", color="#888888", lw=2, ms=8, label="Closed-book accuracy"
    )
    axA.plot(x, df["gate_auroc"], "-s", color="#4575b4", lw=2, ms=8, label="Gate AUROC")
    axA.plot(x, df["fr_auc"], "-^", color="#d73027", lw=2.4, ms=9, label="Frontier AUC")
    axA.set_xscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels([f"{p:g}B" for p in x])
    axA.xaxis.set_minor_locator(plt.NullLocator())
    axA.set_xlabel("Model size (parameters, log scale)", fontsize=13, labelpad=8)
    axA.set_ylabel("Score", fontsize=13, labelpad=8)
    axA.set_ylim(0.2, 0.9)
    axA.set_title(
        "a  Stronger model → more reliable gate", fontsize=14, loc="left", pad=10
    )
    style_axes(axA)
    axA.legend(frameon=False, fontsize=11, loc="lower right")

    # (b) retrieval the gate can skip while matching always-retrieve
    savings = (1.0 - df["retr_to_match_always"]) * 100
    colors = [SCALE_COLORS.get(m, "#d73027") for m in df["model"]]
    bars = axB.bar(
        df["model"], savings, color=colors, edgecolor="black", linewidth=1.0, width=0.62
    )
    for b, v in zip(bars, savings):
        axB.text(
            b.get_x() + b.get_width() / 2,
            v + 1.5,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    axB.set_ylabel("Retrieval skipped at equal accuracy (%)", fontsize=13, labelpad=8)
    axB.set_ylim(0, 72)
    axB.set_title("b  … and a larger free-skip budget", fontsize=14, loc="left", pad=10)
    style_axes(axB)

    fig.tight_layout(pad=0.8)
    save(fig, "fig2_scaling")


if __name__ == "__main__":
    main()
