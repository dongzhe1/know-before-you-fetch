"""Figure 3: selective RAG risk-coverage curves."""

from __future__ import annotations
import os, sys, csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
try:
    from figure_style import apply_style, save

    apply_style()
except ImportError:

    def save(fig, name):
        fig.savefig(os.path.join(HERE, "figures", f"{name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(HERE, "figures", f"{name}.png"), bbox_inches="tight")


DATA_PATH = os.path.join(HERE, "data", "figdata_selective.csv")
OUT_DIR = os.path.join(HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["TriviaQA-8B", "NQ-8B", "MS-MARCO-8B"]
TITLES = ["TriviaQA (8B)", "NQ (8B)", "MS-MARCO (8B)"]

METHOD_STYLE = {
    "proper": dict(
        color="#2166ac", lw=2.0, ls="-", label="Proper (chosen-answer conf)", zorder=3
    ),
    "naive": dict(
        color="#d6604d", lw=1.6, ls="--", label="Naive (closed-book conf)", zorder=2
    ),
    "random": dict(
        color="#888888", lw=1.2, ls=":", label="Random abstention", zorder=1
    ),
}

AURC_VALUES = {
    "TriviaQA-8B": {"proper": 0.1118, "naive": 0.1558, "random": 0.2146},
    "NQ-8B": {"proper": 0.2219, "naive": 0.2998, "random": 0.3145},
    "MS-MARCO-8B": {"proper": 0.6464, "naive": 0.6110, "random": 0.6888},
}


def load_data():
    data = {}  # dataset -> method -> (cov[], risk[])
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds, method = row["dataset"], row["method"]
            cov, risk = float(row["coverage"]), float(row["risk"])
            data.setdefault(ds, {}).setdefault(method, ([], []))
            data[ds][method][0].append(cov)
            data[ds][method][1].append(risk)
    return data


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Missing {DATA_PATH} — run selective_rag.py first.")
        return

    data = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=False)

    for ax, ds, title in zip(axes, DATASETS, TITLES):
        if ds not in data:
            ax.set_title(title)
            continue
        ds_data = data[ds]

        # shade between proper and naive
        if "proper" in ds_data and "naive" in ds_data:
            cov_p, risk_p = ds_data["proper"]
            cov_n, risk_n = ds_data["naive"]
            # interpolate naive onto proper's coverage grid
            risk_n_interp = np.interp(cov_p, cov_n, risk_n)
            ax.fill_between(
                cov_p,
                risk_p,
                risk_n_interp,
                alpha=0.12,
                color="#2166ac",
                label="_nolegend_",
            )

        for method in ["random", "naive", "proper"]:
            if method not in ds_data:
                continue
            cov, risk = ds_data[method]
            ax.plot(cov, risk, **METHOD_STYLE[method])

        # annotate AURC
        aurc_vals = AURC_VALUES.get(ds, {})
        txt = "\n".join(f"AURC({m[:3]})={v:.4f}" for m, v in aurc_vals.items())
        ax.text(
            0.04,
            0.97,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"),
        )

        ax.set_xlabel("Coverage", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Risk (1 − accuracy)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

    # single legend below
    handles = [
        plt.Line2D([0], [0], **{k: v for k, v in s.items() if k != "zorder"})
        for s in [METHOD_STYLE["proper"], METHOD_STYLE["naive"], METHOD_STYLE["random"]]
    ]
    fig.legend(
        handles,
        [
            s["label"]
            for s in [
                METHOD_STYLE["proper"],
                METHOD_STYLE["naive"],
                METHOD_STYLE["random"],
            ]
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )
    fig.tight_layout()
    save(fig, "fig3_selective")
    print(f"wrote fig3_selective.pdf")


if __name__ == "__main__":
    main()
