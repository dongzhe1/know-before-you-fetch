#!/usr/bin/env python3
"""One-shot reproduction: manifest -> figdata -> figures -> validation."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Core data pipeline + figures + validation
CORE_SCRIPTS = [
    ("make_manifest.py", []),
    ("targ_graded_baseline.py", []),
    ("selective_rag.py", []),
    ("cross_domain_selective.py", []),
    ("calibration_evidence.py", []),
    ("deployment_eval.py", []),
    ("retriever_robustness.py", []),
    ("multihop_analysis.py", []),
    ("make_cost_metrics.py", []),
    ("make_data.py", []),
]

FIGURE_SCRIPTS = [
    ("fig1_frontier.py", []),
    ("fig2_scaling.py", []),
    ("fig3_selective.py", []),
    ("fig4_ternary.py", []),
    ("fig5_baselines.py", []),
    ("fig6_regime.py", []),
    ("fig7_generalization.py", []),
    ("fig8_difficulty.py", []),
    ("fig9_motivation.py", []),
]

VALIDATION = [
    ("validate_outputs.py", []),
]


def run(script: str, extra_args: list[str] | None = None) -> None:
    path = HERE / script
    if not path.exists():
        print(f"[run_all] SKIP {script} (not found)")
        return
    print(f"\n{'=' * 60}")
    print(f"[run_all] {script}")
    print(f"{'=' * 60}")
    cmd = [PY, str(path)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(HERE), capture_output=False)
    if result.returncode != 0:
        print(f"[run_all] FAILED: {script} (exit {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    fast = "--fast" in sys.argv
    bootstrap = ["--bootstrap", "200"] if fast else []

    print(f"[run_all] Python: {PY}")
    print(f"[run_all] Working directory: {HERE}")
    print(f"[run_all] Mode: {'fast (B=200)' if fast else 'full (B=1000)'}")

    all_scripts = CORE_SCRIPTS + FIGURE_SCRIPTS + VALIDATION
    for script, extra in all_scripts:
        args = extra[:]
        if script == "make_manifest.py":
            args += bootstrap
        run(script, args)
    print(f"\n[run_all] ✅ All {len(all_scripts)} steps complete.")


if __name__ == "__main__":
    main()
