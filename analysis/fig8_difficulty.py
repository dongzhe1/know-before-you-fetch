#!/usr/bin/env python3
"""Figure 8: difficulty stratification by confidence quartile."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import DATA, save, set_style, style_axes


def main() -> None:
    set_style()
    df = pd.read_csv(DATA / "figdata_difficulty.csv")
    df["headroom"] = df["open_acc"] - df["cb_acc"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(df))
    w = 0.38
    labels = [f"{q}\n(p̄={p:.2f})" for q, p in zip(df["stratum"], df["p_correct"])]

    # (a) closed-book vs open-book accuracy per confidence stratum
    axA.bar(
        x - w / 2,
        df["cb_acc"],
        w,
        color="#888888",
        edgecolor="black",
        linewidth=1.0,
        label="Closed-book",
    )
    axA.bar(
        x + w / 2,
        df["open_acc"],
        w,
        color="#4575b4",
        edgecolor="black",
        linewidth=1.0,
        label="Open-book (k=5)",
    )
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=10.5)
    axA.set_ylabel("Answer accuracy", fontsize=13, labelpad=8)
    axA.set_ylim(0, 1.0)
    axA.set_title(
        "a  Closed- vs open-book by gate-confidence quartile",
        fontsize=14,
        loc="left",
        pad=10,
    )
    style_axes(axA)
    axA.legend(frameon=False, fontsize=11, loc="upper left")

    # (b) retrieval headroom (open − closed) — vanishes at the confident end
    colors = ["#d73027"] * len(df)
    bars = axB.bar(
        x, df["headroom"], color=colors, edgecolor="black", linewidth=1.0, width=0.6
    )
    for bar, v in zip(bars, df["headroom"]):
        axB.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.012 if v >= 0 else -0.028),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=11,
            fontweight="bold",
        )
    axB.axhline(0, color="black", lw=1.0)
    axB.set_xticks(x)
    axB.set_xticklabels([q for q in df["stratum"]], fontsize=11)
    axB.set_xlabel("Gate-confidence quartile (low → high)", fontsize=12, labelpad=8)
    axB.set_ylabel("Retrieval headroom  (open − closed)", fontsize=13, labelpad=8)
    axB.set_ylim(-0.08, max(df["headroom"]) + 0.08)
    axB.set_title(
        "b  Retrieval value → 0 where the gate is confident",
        fontsize=14,
        loc="left",
        pad=10,
    )
    style_axes(axB)

    fig.tight_layout()
    save(fig, "fig8_difficulty")


if __name__ == "__main__":
    main()
