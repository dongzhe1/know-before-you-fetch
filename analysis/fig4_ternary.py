"""Figure 4: ternary operating space (skip / retrieve / abstain)."""

import os, sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "figdata_ternary.csv")
FIGS_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

ANCHORS = {
    "TriviaQA-8B": {"always_skip": 0.572, "always_ret": 0.785},
    "TriviaQA-32B": {"always_skip": 0.695, "always_ret": 0.818},
}
DATASETS = ["TriviaQA-8B", "TriviaQA-32B"]
PANEL_LABELS = ["TriviaQA (Qwen3-8B)", "TriviaQA (Qwen3-32B)"]

SQRT3 = np.sqrt(3)
V_SKIP = np.array([0.5, SQRT3 / 2])
V_RET = np.array([0.0, 0.0])
V_ABS = np.array([1.0, 0.0])


def bary(s, r, a):
    return s * V_SKIP[0] + r * V_RET[0] + a * V_ABS[0], s * V_SKIP[1] + r * V_RET[
        1
    ] + a * V_ABS[1]


def draw_ternary_base(ax):
    ax.add_patch(
        plt.Polygon(
            [V_RET, V_ABS, V_SKIP], fill=False, edgecolor="#333", lw=1.6, zorder=5
        )
    )
    for i in range(1, 5):
        f = i / 5
        kw = dict(color="#ccc", lw=0.4, alpha=0.5, zorder=1)
        ax.plot(*zip(bary(f, 1 - f, 0), bary(f, 0, 1 - f)), **kw)
        ax.plot(*zip(bary(1 - f, f, 0), bary(0, f, 1 - f)), **kw)
        ax.plot(*zip(bary(1 - f, 0, f), bary(0, 1 - f, f)), **kw)
    for i in range(1, 5):
        f = i / 5
        x, y = bary(f, 1 - f, 0)
        ax.text(
            x - 0.03,
            y + 0.005,
            f"{int(f * 100)}",
            fontsize=6,
            ha="right",
            va="center",
            color="#999",
            rotation=60,
        )
        x, y = bary(0, f, 1 - f)
        ax.text(
            x + 0.005,
            y - 0.03,
            f"{int(f * 100)}",
            fontsize=6,
            ha="left",
            va="top",
            color="#999",
        )
        x, y = bary(1 - f, 0, f)
        ax.text(
            x + 0.03,
            y + 0.005,
            f"{int(f * 100)}",
            fontsize=6,
            ha="left",
            va="center",
            color="#999",
            rotation=-60,
        )
    ax.text(
        V_SKIP[0],
        V_SKIP[1] + 0.07,
        "100% Skip\n(closed-book)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#111",
    )
    ax.text(
        V_RET[0] - 0.07,
        V_RET[1] - 0.07,
        "100% Retrieve\n(open-book)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#111",
    )
    ax.text(
        V_ABS[0] + 0.07,
        V_ABS[1] - 0.07,
        "100% Abstain\n(decline to answer)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#111",
    )
    ax.text(0.10, -0.025, "retrieve %", fontsize=7, ha="center", va="top", color="#777")
    ax.text(1.10, 0.03, "abstain %", fontsize=7, ha="center", va="top", color="#777")
    ax.text(
        0.50,
        SQRT3 / 2 + 0.03,
        "skip %",
        fontsize=7,
        ha="center",
        va="bottom",
        color="#777",
    )
    ax.set_xlim(-0.26, 1.26)
    ax.set_ylim(-0.26, SQRT3 / 2 + 0.20)
    ax.set_aspect("equal")
    ax.axis("off")


def pareto_envelope(ret_pct, acc, bins=100):
    edges = np.linspace(0, 100, bins + 1)
    xs, ys = [], []
    for i in range(bins):
        m = (ret_pct >= edges[i]) & (ret_pct < edges[i + 1])
        if m.sum():
            xs.append((edges[i] + edges[i + 1]) / 2)
            ys.append(acc[m].max())
    xs, ys = np.array(xs), np.array(ys)
    return xs, np.maximum.accumulate(ys[::-1])[::-1]


def main():
    df = pd.read_csv(DATA_PATH)
    acc_min, acc_max = df.accuracy.min(), df.accuracy.max()
    norm_acc = Normalize(vmin=acc_min, vmax=acc_max)
    sm_acc = ScalarMappable(cmap="RdYlGn", norm=norm_acc)
    sm_acc.set_array([])

    fig = plt.figure(figsize=(13.0, 10.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.045],
        height_ratios=[1.15, 0.85],
        hspace=0.38,
        wspace=0.34,
        left=0.06,
        right=0.935,
        top=0.89,
        bottom=0.065,
    )
    ax_t0 = fig.add_subplot(gs[0, 0])
    ax_t1 = fig.add_subplot(gs[0, 1])
    ax_cb_tern = fig.add_subplot(gs[0, 2])
    ax_p0 = fig.add_subplot(gs[1, 0])
    ax_p1 = fig.add_subplot(gs[1, 1])
    ax_cb_pare = fig.add_subplot(gs[1, 2])

    for col, ds in enumerate(DATASETS):
        sub = df[df["dataset"] == ds].copy().reset_index(drop=True)
        anch = ANCHORS[ds]
        at = [ax_t0, ax_t1][col]
        ap = [ax_p0, ax_p1][col]
        draw_ternary_base(at)
        xs_t, ys_t = bary(
            sub.skip_frac.values, sub.ret_frac.values, sub.abs_frac.values
        )
        pts = np.column_stack([xs_t, ys_t])
        try:
            hull = ConvexHull(pts)
            at.add_patch(
                plt.Polygon(
                    pts[hull.vertices],
                    closed=True,
                    facecolor="#4a90d9",
                    alpha=0.10,
                    edgecolor="#4a90d9",
                    lw=0.8,
                    linestyle="--",
                    zorder=2,
                )
            )
        except Exception:
            pass
        at.plot(
            [V_RET[0], V_SKIP[0]],
            [V_RET[1], V_SKIP[1]],
            color="#c0392b",
            lw=4.0,
            zorder=6,
            solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=6.0, foreground="white")],
        )
        order = np.argsort(sub.abs_frac.values)
        at.scatter(
            xs_t[order],
            ys_t[order],
            c=sub.accuracy.values[order],
            cmap="RdYlGn",
            norm=norm_acc,
            s=14,
            alpha=0.78,
            linewidths=0.2,
            edgecolors="white",
            zorder=3,
        )
        xd, yd = bary(1, 0, 0)
        at.scatter(
            [xd],
            [yd],
            marker="D",
            s=70,
            c=[[anch["always_skip"]]],
            cmap="RdYlGn",
            norm=norm_acc,
            edgecolors="#111",
            linewidths=1.2,
            zorder=9,
        )
        at.text(
            xd + 0.08,
            yd - 0.03,
            f"always-skip\nacc={anch['always_skip']:.3f}",
            fontsize=7,
            ha="left",
            va="top",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#bbb", alpha=0.85),
        )
        xd, yd = bary(0, 1, 0)
        at.scatter(
            [xd],
            [yd],
            marker="D",
            s=70,
            c=[[anch["always_ret"]]],
            cmap="RdYlGn",
            norm=norm_acc,
            edgecolors="#111",
            linewidths=1.2,
            zorder=9,
        )
        at.text(
            xd + 0.08,
            yd + 0.06,
            f"always-retrieve\nacc={anch['always_ret']:.3f}",
            fontsize=7,
            ha="left",
            va="bottom",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#bbb", alpha=0.85),
        )
        at.annotate(
            "TARG\n(binary only)",
            xy=(0.15, 0.42),
            xytext=(-0.03, 0.62),
            fontsize=8,
            color="#c0392b",
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            zorder=10,
        )
        at.annotate(
            "Ours\n(full interior)",
            xy=(0.60, 0.24),
            xytext=(0.84, 0.38),
            fontsize=8,
            color="#1a56a0",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#1a56a0", lw=1.2),
            zorder=10,
        )
        at.set_title(PANEL_LABELS[col], fontsize=11, fontweight="bold", pad=8)

        ret = sub.ret_frac.values * 100
        acc = sub.accuracy.values
        ab = sub.abs_frac.values
        norm_ab = Normalize(vmin=0, vmax=max(ab.max(), 0.01))
        sc = ap.scatter(
            ret,
            acc,
            c=ab,
            cmap="YlOrBr",
            norm=norm_ab,
            s=9,
            alpha=0.38,
            linewidths=0,
            zorder=2,
        )
        px, py = pareto_envelope(ret, acc, bins=100)
        ap.plot(
            px,
            py,
            "-",
            color="#1a5fa8",
            lw=2.6,
            zorder=4,
            solid_capstyle="round",
            label="Pareto frontier",
        )
        dy_text = (acc.max() - acc.min()) * 0.015
        for yval, lbl, clr in [
            (anch["always_ret"], "always-retrieve", "#2166ac"),
            (anch["always_skip"], "always-skip (CB)", "#d6604d"),
        ]:
            ap.axhline(yval, color=clr, lw=1.0, ls="--", alpha=0.6, zorder=3)
            ap.text(
                102,
                yval + dy_text,
                lbl,
                ha="right",
                va="bottom",
                fontsize=8,
                color=clr,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75),
            )
        zero = sub[sub.ret_frac <= 0.05]
        if len(zero):
            b0 = zero.loc[zero.accuracy.idxmax()]
            ap.axhline(
                b0.accuracy, color="#762a83", lw=0.8, ls=":", alpha=0.6, zorder=3
            )
            ap.text(
                102,
                b0.accuracy + dy_text,
                f"best@0%ret, abs={b0.abs_frac * 100:.0f}%",
                ha="right",
                va="bottom",
                fontsize=8,
                color="#762a83",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75),
            )
        mod = (px >= 5) & (px <= 50)
        if mod.sum() and py[mod].max() > 0:
            bi = np.argmax(py[mod])
            bx, by = px[mod][bi], py[mod][bi]
            nearby = sub[np.abs(sub.ret_frac * 100 - bx) < 6]
            ab_best = nearby.abs_frac.max() * 100 if len(nearby) else 0
            ap.scatter(
                [bx], [by], s=60, color="#1a5fa8", edgecolors="white", lw=1.0, zorder=5
            )
            ap.annotate(
                f"best: ret={bx:.0f}%, acc={by:.3f}, abs={ab_best:.0f}%",
                xy=(bx, by),
                xytext=(bx + 14, by - 0.04),
                fontsize=8.5,
                color="#1a5fa8",
                arrowprops=dict(arrowstyle="->", color="#1a5fa8", lw=1.0),
                zorder=6,
            )
        ylo = min(acc.min(), anch["always_skip"], anch["always_ret"]) - 0.025
        yhi = acc.max() + 0.05
        ap.set_xlim(-3, 110)
        ap.set_ylim(ylo, yhi)
        ap.set_xlabel("Retrieval rate (%)", fontsize=9.5)
        if col == 0:
            ap.set_ylabel("Accuracy (on answered queries)", fontsize=9.5)
        ap.set_title(PANEL_LABELS[col], fontsize=11, fontweight="bold")
        ap.tick_params(labelsize=8)
        ap.grid(True, lw=0.3, alpha=0.3, color="#ccc")
        gap_y = (anch["always_skip"] + anch["always_ret"]) / 2
        ap.legend(
            fontsize=8.5,
            loc="center left",
            framealpha=0.85,
            edgecolor="#ccc",
            bbox_to_anchor=(0.02, gap_y),
            bbox_transform=ap.transData,
        )

    cbp = fig.colorbar(sc, cax=ax_cb_pare)
    cbp.set_label("Abstention\nrate", fontsize=8.5, rotation=90, labelpad=8)
    cbp.ax.tick_params(labelsize=7.5)
    cbp.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    cbt = fig.colorbar(sm_acc, cax=ax_cb_tern)
    cbt.set_label("Accuracy", fontsize=9, rotation=270, labelpad=14)
    cbt.ax.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], color="#c0392b", lw=3.5, label="TARG\n(binary, abs=0)"),
        mpatches.Patch(
            facecolor="#4a90d9",
            alpha=0.25,
            edgecolor="#4a90d9",
            linestyle="--",
            label="Ours\n(ternary interior)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#888",
            markeredgecolor="#111",
            markersize=7,
            label="Anchor\npoints",
        ),
    ]
    leg = fig.legend(
        handles=handles,
        fontsize=7.5,
        framealpha=0.92,
        edgecolor="#bbb",
        loc="center",
        bbox_to_anchor=(0.46, 0.60),
        ncol=1,
    )
    leg.set_zorder(100)
    fig.text(
        0.46,
        0.94,
        "(a)  Ternary operating space: skip / retrieve / abstain",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.46,
        0.495,
        "(b)  Pareto slice: accuracy vs retrieval rate (colored by abstention rate)",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGS_DIR, f"fig4_ternary.{ext}")
        fig.savefig(
            path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.15
        )
        print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    main()
