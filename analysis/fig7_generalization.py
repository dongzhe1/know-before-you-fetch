#!/usr/bin/env python3
"""Figure 7: cross-family and cross-dataset generalization."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import DATASET_COLORS, FAMILY_COLORS, save, set_style, style_axes

HERE = os.path.dirname(__file__)
MANIFEST = os.path.join(HERE, "results_manifest.csv")
FAMILY = {"Qwen3-8B": "Qwen3", "Qwen3.5-9B": "Qwen3.5", "Llama-3.1-8B": "Llama"}


def main() -> None:
    set_style()
    m = pd.read_csv(MANIFEST)
    big = m[m["retriever"] == "bge-large"].copy()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))

    # (a) cross-family normalized value at ~8-9B (bge-large TriviaQA rows)
    fam = (
        big[big["model"].isin(FAMILY)]
        .drop_duplicates("model")
        .set_index("model")
        .loc[list(FAMILY)]
    )
    colors = [FAMILY_COLORS.get(FAMILY[mm], "#d73027") for mm in fam.index]
    yerr = np.vstack(
        [
            fam["norm_value"] - fam["norm_value_ci_low"],
            fam["norm_value_ci_high"] - fam["norm_value"],
        ]
    )
    bars = axA.bar(
        fam.index,
        fam["norm_value"],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        width=0.6,
        yerr=yerr,
        capsize=5,
        error_kw=dict(elinewidth=1.0),
    )
    for bar, nv, au in zip(bars, fam["norm_value"], fam["gate_auroc"]):
        axA.text(
            bar.get_x() + bar.get_width() / 2,
            fam["norm_value_ci_high"].max() + 0.03,
            f"NV {nv:.2f}\nAUROC {au:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axA.set_ylabel("Normalized gate value", fontsize=13, labelpad=8)
    axA.set_ylim(0, max(0.75, fam["norm_value_ci_high"].max() + 0.18))
    axA.set_xticks(range(len(fam)))
    axA.set_xticklabels(fam.index, fontsize=11)
    axA.set_title(
        "a  Holds across model families (~8–9B)", fontsize=14, loc="left", pad=10
    )
    style_axes(axA)

    # (b) normalized value vs closed-book accuracy, with CIs
    # Use canonical retriever per dataset; exclude shared-corpus duplicates.
    # Also exclude n=600 when n=2000 is available (dedup same model×dataset)
    pts = m[
        m["dataset"].isin(["TriviaQA-rc", "NQ-DPR", "MS-MARCO"])
        & (~m["retriever"].isin(["bge-small", "shared-corpus"]))
    ].copy()
    # Keep largest n per (model, dataset) — prefer 2k over 600
    pts["_rank"] = pts.groupby(["model", "dataset"])["n"].rank("dense", ascending=False)
    pts = pts[pts["_rank"] == 1].drop(columns=["_rank"])

    # assign markers by model family / size
    def marker_for(row):
        if "32B" in row["model"]:
            return "s"  # square = 32B
        if "1.7B" in row["model"]:
            return "^"  # triangle = 1.7B
        if "Llama" in row["model"]:
            return "P"  # plus = Llama
        if "Qwen3.5" in row["model"]:
            return "X"  # X = Qwen3.5
        return "o"  # circle = Qwen3-8B

    for _, r in pts.iterrows():
        col = DATASET_COLORS.get(r["dataset"], "#d73027")
        mk = marker_for(r)
        axB.errorbar(
            r["closed_book_acc"],
            r["norm_value"],
            yerr=[
                [r["norm_value"] - r["norm_value_ci_low"]],
                [r["norm_value_ci_high"] - r["norm_value"]],
            ],
            fmt=mk,
            ms=10,
            color=col,
            mec="black",
            mew=1.0,
            ecolor="#888888",
            elinewidth=1.0,
            capsize=4,
            zorder=5,
        )
    # Dataset-color legend — bottom-right corner
    from matplotlib.lines import Line2D

    ds_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=c,
            mec="black",
            mew=1.0,
            ms=8,
            linestyle="none",
            label=lbl,
        )
        for lbl, c in [
            ("TriviaQA-rc", DATASET_COLORS["TriviaQA-rc"]),
            ("NQ-DPR", DATASET_COLORS["NQ-DPR"]),
            ("MS-MARCO", DATASET_COLORS["MS-MARCO"]),
        ]
    ]
    leg1 = axB.legend(
        handles=ds_handles,
        fontsize=8,
        framealpha=0.85,
        edgecolor="#ccc",
        loc="lower right",
        title="Dataset",
        bbox_to_anchor=(1.0, 0.0),
    )
    # Model-marker legend — to the LEFT of Dataset, same y-level
    mk_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="#555",
            mec="black",
            mew=0.8,
            ms=7,
            linestyle="none",
            label=lbl,
        )
        for m, lbl in [
            ("o", "Qwen3-8B"),
            ("s", "32B"),
            ("X", "Qwen3.5-9B"),
            ("P", "Llama-3.1-8B"),
            ("^", "1.7B"),
        ]
    ]
    leg2 = axB.legend(
        handles=mk_handles,
        fontsize=7,
        framealpha=0.85,
        edgecolor="#ccc",
        loc="lower right",
        title="Model",
        bbox_to_anchor=(0.72, 0.0),
    )
    axB.add_artist(leg1)  # restore first legend after adding second
    axB.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    axB.set_xlabel("Closed-book accuracy", fontsize=13, labelpad=8)
    axB.set_ylabel("Normalized gate value", fontsize=13, labelpad=8)
    axB.set_xlim(0.1, 0.75)
    axB.set_title(
        "b  Value is positive but not a clean CB-accuracy law",
        fontsize=13.5,
        loc="left",
        pad=10,
    )
    style_axes(axB)

    fig.tight_layout()
    save(fig, "fig7_generalization")


if __name__ == "__main__":
    main()
