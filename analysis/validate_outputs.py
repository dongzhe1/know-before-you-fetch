"""Check that all required CSVs and figures exist and are consistent."""

from __future__ import annotations
import json, os, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"
TABLES = HERE / "tables"

REQUIRED_CSVS = [
    "results_manifest.csv",
    "figdata_frontier_main.csv",
    "figdata_frontier_byscale.csv",
    "figdata_frontier_anchors.csv",
    "figdata_scaling.csv",
    "figdata_targ_graded.csv",
    "figdata_selective.csv",
    "figdata_cross_selective.csv",
    "figdata_baselines.csv",
    "figdata_regime.csv",
    "figdata_calibration_evidence.csv",
    "figdata_reliability.csv",
    "figdata_deployment.csv",
    "figdata_retriever_robustness.csv",
    "figdata_multihop_complexity.csv",
    "figdata_token_cost.csv",
    "figdata_k10_degradation.csv",
    "figdata_wiki_vs_pool.csv",
]

REQUIRED_FIGURES = [
    f"fig{i}_{name}.pdf"
    for i, name in enumerate(
        [
            "frontier",
            "scaling",
            "selective",
            "ternary",
            "baselines",
            "regime",
            "generalization",
            "difficulty",
            "motivation",
        ],
        1,
    )
]

ERRORS = []
WARNINGS = []


def check(path, label):
    if not path.exists():
        ERRORS.append(f"MISSING {label}: {path}")
        return False
    if path.stat().st_size == 0:
        ERRORS.append(f"EMPTY  {label}: {path}")
        return False
    return True


def main():
    # Check CSVs
    for csv_name in REQUIRED_CSVS:
        p = DATA / csv_name if not csv_name.startswith("results") else HERE / csv_name
        check(p, csv_name)

    # Check figures
    for fig_name in REQUIRED_FIGURES:
        check(FIGS / fig_name, fig_name)

    # Check manifest source JSONL files exist
    manifest_path = HERE / "results_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            m_rows = list(csv.DictReader(f))
        for r in m_rows:
            tbl = r.get("jsonl_table", "")
            if tbl and not (TABLES / tbl).exists():
                ERRORS.append(f"MANIFEST references missing table: {tbl}")

    # Check deprecated tables not used in manifest
    try:
        from experiment_registry import DEPRECATED

        if manifest_path.exists():
            for r in m_rows:
                tbl = r.get("jsonl_table", "")
                if tbl in DEPRECATED:
                    ERRORS.append(
                        f"MANIFEST uses deprecated table: {tbl} — {DEPRECATED[tbl]}"
                    )
    except ImportError:
        pass

    # Check registry n_expected matches actual row count
    try:
        from experiment_registry import TABLE_REGISTRY

        for key, exp in TABLE_REGISTRY.items():
            tbl = exp.get("table", "")
            n_expected = exp.get("n_expected")
            if not tbl:
                continue
            tbl_path = TABLES / tbl
            if tbl_path.exists() and tbl_path.is_file() and n_expected:
                actual = sum(1 for _ in open(tbl_path) if _.strip())
                if actual != n_expected:
                    WARNINGS.append(
                        f"registry {key}: n_expected={n_expected} but actual={actual}"
                    )
    except ImportError:
        pass

    # Check scaling.csv: params_b monotonic, 1.7B not 17
    scaling_path = DATA / "figdata_scaling.csv"
    if scaling_path.exists():
        with open(scaling_path) as f:
            s_rows = list(csv.DictReader(f))
        params = [float(r["params_b"]) for r in s_rows]
        if any(p > 10 and p < 20 for p in params):
            ERRORS.append(
                f"figdata_scaling.csv: params_b contains {params} — 1.7B likely parsed as 17"
            )
        if params != sorted(params):
            WARNINGS.append(f"figdata_scaling.csv: params_b not monotonic: {params}")
        for r in s_rows:
            if float(r.get("graded_auc", 1)) == 0.0:
                ERRORS.append(
                    f"figdata_scaling.csv: graded_auc=0.0 placeholder for {r.get('model')}"
                )

    # Check manifest consistency with scaling data
    if manifest_path.exists() and scaling_path.exists():
        with open(scaling_path) as f:
            s_rows = list(csv.DictReader(f))
        s32 = [r for r in s_rows if "32B" in r.get("model", "")]
        if s32:
            s_cb = float(s32[0].get("cb_acc", -1))
            m32 = [
                r
                for r in m_rows
                if "32B" in r.get("model", "")
                and r.get("dataset") == "TriviaQA-rc"
                and r.get("retriever") == "bge-large"
            ]
            if m32:
                m_cb = float(m32[0].get("closed_book_acc", -1))
                if abs(s_cb - m_cb) > 0.01:
                    ERRORS.append(
                        f"INCONSISTENT: scaling 32B CB={s_cb:.3f} vs manifest CB={m_cb:.3f}"
                    )
        else:
            ERRORS.append("MISSING 32B row in figdata_scaling.csv")

    # Check no NaN in main frontier CSV
    frontier_path = DATA / "figdata_frontier_main.csv"
    if frontier_path.exists():
        with open(frontier_path) as f:
            for i, line in enumerate(f):
                if "NaN" in line or "nan" in line:
                    ERRORS.append(f"NaN in figdata_frontier_main.csv line {i}")
                    break

    # Check TARG graded CSV has no all-NaN variance signal
    targ_path = DATA / "figdata_targ_graded.csv"
    if targ_path.exists():
        with open(targ_path) as f:
            targ_rows = list(csv.DictReader(f))
        variance_rows = [
            r for r in targ_rows if "variance" in r.get("signal", "").lower()
        ]
        if variance_rows:
            ERRORS.append(
                f"figdata_targ_graded.csv still contains TARG variance rows (all-NaN signal)"
            )

    # Check selective_summary.csv has no NaN
    sel_summary = DATA / "selective_summary.csv"
    if sel_summary.exists():
        with open(sel_summary) as f:
            content = f.read()
        if "nan" in content.lower():
            ERRORS.append(f"selective_summary.csv contains NaN values")

    # Check cost_metrics.csv uses updated timing
    cost_path = DATA / "cost_metrics.csv"
    if cost_path.exists():
        with open(cost_path) as f:
            c_rows = list(csv.DictReader(f))
        for r in c_rows:
            ret_ms = float(r.get("retrieval_ms", 0))
            if 0 < ret_ms < 10.0:
                ERRORS.append(
                    f"cost_metrics.csv: retrieval_ms={ret_ms} is stale (measured=27.5)"
                )
            if (
                "8B" in r.get("model", "")
                and abs(float(r.get("closed_ms", 0)) - 76.5) < 1.0
            ):
                ERRORS.append(
                    f"cost_metrics.csv: 8B closed_ms={r['closed_ms']} is old timing (new=64.6)"
                )

    # Check figdata_regime.csv uses updated timing
    regime_path = DATA / "figdata_regime.csv"
    if regime_path.exists():
        with open(regime_path) as f:
            reg_rows = list(csv.DictReader(f))
        for r in reg_rows:
            if "8B" in r.get("model", "") and abs(float(r.get("c_cb", 0)) - 76.5) < 1.0:
                ERRORS.append(
                    f"figdata_regime.csv: 8B c_cb={r['c_cb']} is old timing (new=64.6)"
                )

    # Check baselines.csv no 0.0 placeholder
    baselines_path = DATA / "figdata_baselines.csv"
    if baselines_path.exists():
        with open(baselines_path) as f:
            b_rows = list(csv.DictReader(f))
        for r in b_rows:
            if "graded" in r.get("method", "").lower():
                if float(r.get("acc25", 1)) == 0.0:
                    WARNINGS.append(
                        f"figdata_baselines.csv: graded acc25=0.0 placeholder"
                    )

    # Check scaleup CSV has all three axes
    scaleup_path = DATA / "figdata_scaleup.csv"
    if scaleup_path.exists():
        with open(scaleup_path) as f:
            header = f.readline()
        for axis in [
            "graded_auc_full_context",
            "graded_auc_retrieval_call",
            "graded_auc_passage_budget",
        ]:
            if axis not in header:
                ERRORS.append(f"figdata_scaleup.csv missing column: {axis}")

    # Report
    if ERRORS:
        print(f"\n{len(ERRORS)} ERROR(s):")
        for e in ERRORS:
            print(f"  ✗ {e}")
    if WARNINGS:
        print(f"\n{len(WARNINGS)} WARNING(s):")
        for w in WARNINGS:
            print(f"  ⚠ {w}")
    if not ERRORS and not WARNINGS:
        print("✅ All validations passed.")
    n_csv = len(REQUIRED_CSVS)
    n_fig = len(REQUIRED_FIGURES)
    print(f"   {n_csv} CSVs, {n_fig} figures checked")
    if ERRORS:
        sys.exit(1)


if __name__ == "__main__":
    main()
