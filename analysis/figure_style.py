"""Shared matplotlib style: set_style(), style_axes(), save(), palettes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"


def set_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42  # editable text in Illustrator
    plt.rcParams["svg.fonttype"] = "none"
    sns.set_theme(style="ticks", rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})


def style_axes(ax, *, grid: bool = False) -> None:
    """Black spines, outward ticks — the reference look."""
    ax.tick_params(
        axis="x", bottom=True, direction="out", length=6, width=1.0, labelsize=12
    )
    ax.tick_params(
        axis="y", left=True, direction="out", length=6, width=1.0, labelsize=12
    )
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.6)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def save(fig, name: str) -> None:
    """Write <name>.pdf and <name>.png to figures/."""
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"  saved figures/{name}.pdf (+ .png)")


# our method vs baselines (consistent across all figures)
METHOD_COLORS = {
    "gate": "#d73027",  # our confidence gate (the hero, red)
    "confidence": "#f46d43",  # confidence-only
    "graded": "#a50026",  # graded skip/k_small/k_large (darker red)
    "adaptive-rag": "#4575b4",  # Adaptive-RAG baseline (blue)
    "self-rag": "#5e3c99",  # Self-RAG baseline (purple)
    "random": "#999999",  # random-skip
    "length": "#bdbdbd",  # length heuristic
    "oracle": "#1a9850",  # oracle ceiling (green)
    "always": "#222222",  # always-retrieve anchor
    "never": "#888888",  # never-retrieve anchor
}

# model scale ramp (Qwen3 family, light -> dark with size)
SCALE_COLORS = {
    "Qwen3-1.7B": "#fdae61",
    "Qwen3-8B": "#f46d43",
    "Qwen3-32B": "#d73027",
    "Qwen3.5-9B": "#74add1",
    "Llama-3.1-8B": "#4575b4",
}

# model-family accents (for cross-family figure)
FAMILY_COLORS = {"Qwen3": "#d73027", "Qwen3.5": "#4575b4", "Llama": "#5e3c99"}

# dataset accents (for cross-dataset figure)
DATASET_COLORS = {"TriviaQA-rc": "#d73027", "NQ-DPR": "#4575b4", "MS-MARCO": "#fdae61"}


def label_pct(x: float) -> str:
    return f"{x * 100:.0f}%"
