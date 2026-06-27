#!/usr/bin/env python3
"""Figure 1: cost-quality frontier curves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import DATA, METHOD_COLORS, SCALE_COLORS, save, set_style, style_axes

METHOD_STYLE = {  # label, color, linestyle, z
    "oracle": ("Oracle ceiling", METHOD_COLORS["oracle"], (0, (4, 2)), 2),
    "gate": ("Confidence gate (ours)", METHOD_COLORS["gate"], "-", 5),
    "length": ("Length heuristic", METHOD_COLORS["length"], "-", 3),
    "random": ("Random-skip", METHOD_COLORS["random"], "-", 3),
}


def main() -> None:
    set_style()
    main_df = pd.read_csv(DATA / "figdata_frontier_main.csv")
    byscale = pd.read_csv(DATA / "figdata_frontier_byscale.csv")
    anchors = pd.read_csv(DATA / "figdata_frontier_anchors.csv").set_index("model")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))

    # (a) gate vs baselines on Qwen3-8B
    for method in ["oracle", "length", "random", "gate"]:
        d = main_df[main_df["method"] == method].sort_values("retr_rate")
        lbl, col, ls, z = METHOD_STYLE[method]
        lw = 2.8 if method == "gate" else 1.8
        axA.plot(
            d["retr_rate"] * 100,
            d["accuracy"],
            linestyle=ls,
            color=col,
            lw=lw,
            label=lbl,
            zorder=z,
        )
    never = anchors.loc["Qwen3-8B", "never"]
    always = anchors.loc["Qwen3-8B", "always"]
    axA.axhline(always, color="#222222", lw=1.0, ls=":", zorder=1)
    axA.text(2, always + 0.004, "always-retrieve", fontsize=10, color="#222222")
    axA.axhline(never, color="#888888", lw=1.0, ls=":", zorder=1)
    axA.text(
        2, never + 0.004, "never-retrieve (closed-book)", fontsize=10, color="#888888"
    )
    axA.set_xlabel("Retrieval rate (%)", fontsize=13, labelpad=8)
    axA.set_ylabel("Answer accuracy", fontsize=13, labelpad=8)
    axA.set_xlim(0, 100)
    axA.set_title(
        "a  Gate dominates baselines (Qwen3-8B)", fontsize=14, loc="left", pad=10
    )
    style_axes(axA)
    axA.legend(frameon=False, fontsize=11, loc="lower right")

    # (b) gate frontier per model — scaling / cross-family shift
    order = ["Qwen3-1.7B", "Qwen3-8B", "Qwen3.5-9B", "Llama-3.1-8B"]
    for model in order:
        d = byscale[byscale["model"] == model].sort_values("retr_rate")
        if d.empty:
            continue
        axB.plot(
            d["retr_rate"] * 100,
            d["accuracy"],
            "-",
            color=SCALE_COLORS[model],
            lw=2.4,
            label=model,
        )
    axB.set_xlabel("Retrieval rate (%)", fontsize=13, labelpad=8)
    axB.set_ylabel("Answer accuracy", fontsize=13, labelpad=8)
    axB.set_xlim(0, 100)
    axB.set_title(
        "b  Gate frontier shifts up-left with scale / family",
        fontsize=14,
        loc="left",
        pad=10,
    )
    style_axes(axB)
    axB.legend(
        frameon=False, fontsize=11, loc="lower right", title="closed-book → open-book"
    )

    fig.tight_layout()
    save(fig, "fig1_frontier")


if __name__ == "__main__":
    main()
