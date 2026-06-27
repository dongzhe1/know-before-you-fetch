#!/usr/bin/env python3
"""Figure 5: gate vs Self-RAG and Adaptive-RAG baselines."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import METHOD_COLORS, DATA, save, set_style, style_axes


def main() -> None:
    set_style()
    sr = pd.read_csv(DATA / "figdata_selfrag.csv")
    ar = pd.read_csv(DATA / "figdata_adaptiverag.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))

    # (a) Self-RAG vs gate at matched retrieval rate
    x = np.arange(len(sr))
    w = 0.36
    b1 = axA.bar(
        x - w / 2,
        sr["selfrag_acc"],
        w,
        color=METHOD_COLORS["self-rag"],
        edgecolor="black",
        linewidth=1.0,
        label="Self-RAG (fine-tuned 7B)",
    )
    b2 = axA.bar(
        x + w / 2,
        sr["gate_acc_matched"],
        w,
        color=METHOD_COLORS["gate"],
        edgecolor="black",
        linewidth=1.0,
        label="Gate (ours), matched retrieval",
    )
    for bars, col in [(b1, sr["selfrag_acc"]), (b2, sr["gate_acc_matched"])]:
        for bar, v in zip(bars, col):
            axA.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.006,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    # delta annotations
    for xi, (_, r) in zip(x, sr.iterrows()):
        d = r["gate_acc_matched"] - r["selfrag_acc"]
        axA.text(
            xi,
            max(r["selfrag_acc"], r["gate_acc_matched"]) + 0.045,
            f"+{d * 100:.1f} pt",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=METHOD_COLORS["gate"],
        )
    axA.set_xticks(x)
    axA.set_xticklabels(
        [
            f"{d}\n(Self-RAG retr={r:.0%})"
            for d, r in zip(sr["dataset"], sr["selfrag_retr"])
        ],
        fontsize=11,
    )
    axA.set_ylabel("Answer accuracy", fontsize=13, labelpad=8)
    axA.set_ylim(0, 0.9)
    axA.set_title(
        "a  Decision signal vs Self-RAG trigger (matched reader)",
        fontsize=14,
        loc="left",
        pad=10,
    )
    style_axes(axA)
    axA.legend(frameon=False, fontsize=10.5, loc="upper right")

    # (b) Adaptive-RAG vs gate (frontier AUC) across datasets
    x = np.arange(len(ar))
    w = 0.36
    axB.bar(
        x - w / 2,
        ar["adaptiverag_auc"],
        w,
        color=METHOD_COLORS["adaptive-rag"],
        edgecolor="black",
        linewidth=1.0,
        label="Adaptive-RAG (bge + logistic)",
    )
    axB.bar(
        x + w / 2,
        ar["gate_auc"],
        w,
        color=METHOD_COLORS["gate"],
        edgecolor="black",
        linewidth=1.0,
        label="Gate (ours)",
    )
    for xi, (_, r) in zip(x, ar.iterrows()):
        d = r["gate_auc"] - r["adaptiverag_auc"]
        axB.text(
            xi,
            max(r["gate_auc"], r["adaptiverag_auc"]) + 0.012,
            f"+{d:.3f}",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=METHOD_COLORS["gate"],
        )
    axB.set_xticks(x)
    axB.set_xticklabels(ar["dataset"], fontsize=11)
    axB.set_ylabel("Frontier AUC", fontsize=13, labelpad=8)
    axB.set_ylim(0, 0.82)
    axB.set_title(
        "b  Gate vs Adaptive-RAG-inspired query classifier",
        fontsize=14,
        loc="left",
        pad=10,
    )
    style_axes(axB)
    axB.legend(frameon=False, fontsize=10.5, loc="upper right")

    fig.tight_layout()
    save(fig, "fig5_baselines")


if __name__ == "__main__":
    main()
