"""Binary and graded RAG frontier metrics with bootstrap CI."""

from __future__ import annotations
from typing import Literal
import numpy as np

Axis = Literal["full_context", "retrieval_call", "passage_budget"]


def auc_from_points(points: dict[float, float]) -> float:
    xs = np.array(sorted(points.keys()), dtype=float)
    ys = np.array([points[x] for x in xs], dtype=float)
    return float(np.trapezoid(ys, xs))


def binary_frontier_points(p_skip, cb, ob5):
    """Binary frontier. x = retrieval rate (fraction that retrieve k=5).
    Sorted by p_skip descending: top a queries are skipped, rest retrieve.
    x=0 → all skip (closed-book), x=1 → all retrieve k=5 (open-book).
    """
    p_skip = np.asarray(p_skip, dtype=float)
    cb = np.asarray(cb, dtype=float)
    ob5 = np.asarray(ob5, dtype=float)
    n = len(cb)
    idx = np.argsort(-p_skip)
    cb_s = cb[idx]
    ob_s = ob5[idx]
    pref_cb = np.r_[0.0, np.cumsum(cb_s)]
    pref_ob = np.r_[0.0, np.cumsum(ob_s)]
    total_ob = pref_ob[-1]
    points = {}
    for a in range(n + 1):
        acc = (pref_cb[a] + (total_ob - pref_ob[a])) / n
        x = (n - a) / n  # retrieval rate
        points[x] = max(points.get(x, -np.inf), float(acc))
    return points


def binary_frontier_auc(p_skip, cb, ob5):
    return auc_from_points(binary_frontier_points(p_skip, cb, ob5))


def graded_frontier_points(p_skip, cb, ob1, ob5, axis: Axis, grid_step: int = 60):
    """Three-tier frontier: skip / k=1 / k=5.
    Sorted by p_skip descending:
      [0:a)   → skip (closed-book)
      [a:b)   → k=1 (compact retrieval)
      [b:n)   → k=5 (full retrieval)
    grid_step controls the coarseness of (a,b) scan (default 60 → up to ~1800 points).
    """
    p_skip = np.asarray(p_skip, dtype=float)
    cb = np.asarray(cb, dtype=float)
    ob1 = np.asarray(ob1, dtype=float)
    ob5 = np.asarray(ob5, dtype=float)
    n = len(cb)
    idx = np.argsort(-p_skip)
    cb_s = cb[idx]
    ob1_s = ob1[idx]
    ob5_s = ob5[idx]
    pref_cb = np.r_[0.0, np.cumsum(cb_s)]
    pref_ob1 = np.r_[0.0, np.cumsum(ob1_s)]
    pref_ob5 = np.r_[0.0, np.cumsum(ob5_s)]
    total_ob5 = pref_ob5[-1]

    step = max(1, n // grid_step)
    points = {}
    for a in range(0, n + 1, step):
        for b in range(a, n + 1, step):
            acc = (
                pref_cb[a] + (pref_ob1[b] - pref_ob1[a]) + (total_ob5 - pref_ob5[b])
            ) / n
            n_skip = a
            n_k1 = b - a
            n_k5 = n - b
            if axis == "full_context":
                x = n_k5 / n
            elif axis == "retrieval_call":
                x = (n_k1 + n_k5) / n
            elif axis == "passage_budget":
                x = (n_k1 + 5 * n_k5) / (5 * n)
            else:
                raise ValueError(f"Unknown axis: {axis}")
            points[x] = max(points.get(x, -np.inf), float(acc))
    return points


def graded_frontier_auc(p_skip, cb, ob1, ob5, axis: Axis, grid_step: int = 60):
    return auc_from_points(
        graded_frontier_points(p_skip, cb, ob1, ob5, axis, grid_step)
    )


def bootstrap_ci(fn, arrays, B=1000, seed=42, **kw):
    """Bootstrap confidence interval for any scalar metric. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        vals.append(fn(*[a[s] for a in arrays], **kw))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_bootstrap_delta(fn, p1, p2, cb, ob_ref, B=1000, seed=42, **kw):
    """Paired bootstrap: Δ = fn(p1, ...) - fn(p2, ...) on same resampled queries.
    Returns (mean_delta, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(cb)
    deltas = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        v1 = fn(p1[s], cb[s], ob_ref[s], **kw)
        v2 = fn(p2[s], cb[s], ob_ref[s], **kw)
        deltas.append(v1 - v2)
    return (
        float(np.nanmean(deltas)),
        float(np.nanpercentile(deltas, 2.5)),
        float(np.nanpercentile(deltas, 97.5)),
    )


def paired_bootstrap_graded_vs_binary(
    p_cal, cb, ob1, ob5, axis: Axis = "full_context", B=500, seed=42, grid_step=60
):
    """Paired bootstrap: graded_frontier_auc - binary_frontier_auc on same resample."""
    rng = np.random.default_rng(seed)
    n = len(cb)
    deltas = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        g = graded_frontier_auc(p_cal[s], cb[s], ob1[s], ob5[s], axis, grid_step)
        b = binary_frontier_auc(p_cal[s], cb[s], ob5[s])
        deltas.append(g - b)
    return (
        float(np.nanmean(deltas)),
        float(np.nanpercentile(deltas, 2.5)),
        float(np.nanpercentile(deltas, 97.5)),
    )
