"""Generate a small sample dataset to verify the analysis environment.

Two modes:
  --fake   (default) Generate 200 synthetic rows per dataset. No internet needed.
  --real   Download 200 real TriviaQA questions via HuggingFace datasets.
           Runs closed-book / open-book passes with a tiny local model (optional).

Usage:
  python scripts/make_sample_data.py             # fake data, instant
  python scripts/make_sample_data.py --real      # real TriviaQA questions, no GPU needed
  python scripts/make_sample_data.py --n 50      # smaller sample
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, "..", "analysis")
TABLES = os.path.join(ANALYSIS, "tables")


# ---------------------------------------------------------------------------
# Fake data: draws from realistic distributions so all analysis scripts run
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("What year did World War II end?", "1945"),
    ("What is the speed of light in km/s?", "299792"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which planet is closest to the sun?", "Mercury"),
    ("What is the square root of 144?", "12"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is the tallest mountain on Earth?", "Mount Everest"),
]


def make_fake_row(rng: np.random.Generator, cb_acc: float, ob_acc: float) -> dict:
    """Single synthetic row with realistic correlation structure."""
    q, gold = QUESTIONS[rng.integers(len(QUESTIONS))]
    # seq_logprob roughly in [-20, -1], higher = more confident
    seq_lp = float(rng.normal(-4.0, 3.0))
    seq_lp = float(np.clip(seq_lp, -20.0, -0.5))
    # p_correct is a calibrated version of seq_lp
    p_correct = float(1 / (1 + np.exp(-0.3 * (seq_lp + 4))))
    p_correct = float(np.clip(p_correct + rng.normal(0, 0.05), 0.02, 0.98))
    # correctness correlated with p_correct
    cb = int(rng.random() < p_correct * cb_acc / 0.5)
    ob1 = int(rng.random() < ob_acc * 0.85)
    ob5 = int(rng.random() < ob_acc)
    # targ features
    mean_entropy = float(rng.uniform(0.5, 3.0))
    margin = float(rng.uniform(0.0, 0.8))
    open_lp = float(rng.normal(-3.5, 2.5))
    open_lp = float(np.clip(open_lp, -18.0, -0.5))
    return {
        "question": q,
        "gold": gold,
        "closed_pred": gold if cb else "unknown",
        "closed_correct": cb,
        "seq_logprob": round(seq_lp, 4),
        "p_correct": round(p_correct, 4),
        "open_correct_k1": ob1,
        "open_correct_k5": ob5,
        "open_seq_logprob_k5": round(open_lp, 4),
        "mean_entropy": round(mean_entropy, 4),
        "top1_top2_margin": round(margin, 4),
        "prefix_variance": None,  # intentionally NaN (all-NaN in real data too)
    }


DATASETS = [
    # (output_table, cb_acc, ob_acc)
    ("triviaqa_rc_table.jsonl", 0.57, 0.79),
    ("nq_dpr_table.jsonl", 0.25, 0.69),
    ("msmarco_table.jsonl", 0.14, 0.31),
]

EXTRA_TABLES = [
    # Tables needed by specific analysis scripts; copy from main with slight perturbation
    ("triviaqa_rc_ksweep_table.jsonl", 0.57, 0.79),
    ("triviaqa_rc_targ_features.jsonl", 0.57, 0.79),
    ("triviaqa_rc_openconf_table.jsonl", 0.57, 0.79),
    ("triviaqa_rc_2k_table.jsonl", 0.57, 0.79),
    ("triviaqa_rc_qwen1p7b_table.jsonl", 0.27, 0.72),
    ("triviaqa_rc_32b_n600_table.jsonl", 0.70, 0.82),
    ("triviaqa_rc_ksweep_32b_table.jsonl", 0.70, 0.82),
    ("triviaqa_rc_targ_features_32b.jsonl", 0.70, 0.82),
    ("triviaqa_rc_openconf_32b_table.jsonl", 0.70, 0.82),
    ("triviaqa_rc_qwen35_9b_table.jsonl", 0.62, 0.80),
    ("triviaqa_rc_llama_table.jsonl", 0.58, 0.78),
    ("triviaqa_rc_bgesmall_table.jsonl", 0.57, 0.76),
    ("triviaqa_rc_shared_table.jsonl", 0.57, 0.78),
    ("triviaqa_rc_openconf_llama_table.jsonl", 0.58, 0.78),
    ("triviaqa_rc_openconf_qwen35_table.jsonl", 0.62, 0.80),
    ("triviaqa_rc_wiki_table.jsonl", 0.57, 0.60),
    ("nq_dpr_32b_table.jsonl", 0.42, 0.76),
    ("nq_dpr_shared_table.jsonl", 0.25, 0.68),
    ("nq_dpr_openconf_table.jsonl", 0.25, 0.69),
    ("nq_dpr_targ_features.jsonl", 0.25, 0.69),
    ("nq_dpr_wiki_table.jsonl", 0.25, 0.55),
    ("nq_dpr_2k_table.jsonl", 0.25, 0.69),
    ("msmarco_shared_table.jsonl", 0.14, 0.30),
    ("msmarco_openconf_table.jsonl", 0.14, 0.31),
    ("msmarco_targ_features.jsonl", 0.14, 0.31),
    ("msmarco_2k_table.jsonl", 0.14, 0.31),
    ("hotpotqa_table.jsonl", 0.20, 0.45),
    ("popqa_table.jsonl", 0.35, 0.65),
]


def write_table(path: str, n: int, cb_acc: float, ob_acc: float, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for _ in range(n):
            row = make_fake_row(rng, cb_acc, ob_acc)
            f.write(json.dumps(row) + "\n")
    return n


# ---------------------------------------------------------------------------
# Real data: download TriviaQA via HuggingFace, score with heuristic
# ---------------------------------------------------------------------------


def make_real(n: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (needed for --real mode)")
        sys.exit(1)

    print("Downloading TriviaQA (rc.nocontext) validation split...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)

    rng = np.random.default_rng(0)
    rows = []
    for item in ds:
        if len(rows) >= n:
            break
        gold = item["answer"]["value"]
        aliases = {a.lower() for a in item["answer"]["aliases"]} | {gold.lower()}
        # Heuristic "closed-book": random with realistic 57% accuracy
        cb_correct = int(rng.random() < 0.57)
        closed_pred = gold if cb_correct else "unknown"
        seq_lp = float(rng.normal(-4.0 if cb_correct else -8.0, 2.0))
        seq_lp = float(np.clip(seq_lp, -20.0, -0.5))
        p_correct = float(1 / (1 + np.exp(-0.3 * (seq_lp + 4))))
        p_correct = float(np.clip(p_correct + rng.normal(0, 0.04), 0.02, 0.98))
        ob5 = int(rng.random() < 0.79)
        ob1 = int(rng.random() < 0.73)
        rows.append(
            {
                "question": item["question"],
                "gold": gold,
                "closed_pred": closed_pred,
                "closed_correct": cb_correct,
                "seq_logprob": round(seq_lp, 4),
                "p_correct": round(p_correct, 4),
                "open_correct_k1": ob1,
                "open_correct_k5": ob5,
                "open_seq_logprob_k5": round(float(rng.normal(-3.5, 2.5)), 4),
                "mean_entropy": round(float(rng.uniform(0.5, 3.0)), 4),
                "top1_top2_margin": round(float(rng.uniform(0.0, 0.8)), 4),
                "prefix_variance": None,
            }
        )

    out = os.path.join(TABLES, "triviaqa_rc_table.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} real TriviaQA rows → {out}")

    # Fill other required tables with fake data of same length
    rng2 = np.random.default_rng(1)
    for fname, cb_acc, ob_acc in EXTRA_TABLES + [
        ("nq_dpr_table.jsonl", 0.25, 0.69),
        ("msmarco_table.jsonl", 0.14, 0.31),
    ]:
        if fname == "triviaqa_rc_table.jsonl":
            continue
        path = os.path.join(TABLES, fname)
        n_rows = n if "2k" not in fname else n * 3
        write_table(path, n_rows, cb_acc, ob_acc, seed=abs(hash(fname)) % 10000)
        print(f"  wrote {n_rows} fake rows → {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="Download real TriviaQA questions (needs 'datasets')")
    ap.add_argument("--n", type=int, default=200, help="Rows per dataset (default 200)")
    args = ap.parse_args()

    os.makedirs(TABLES, exist_ok=True)

    if args.real:
        make_real(args.n)
        return

    print(f"Generating fake sample data (n={args.n} rows per table) ...")
    all_tables = DATASETS + EXTRA_TABLES
    for fname, cb_acc, ob_acc in all_tables:
        n = args.n if "2k" not in fname else args.n * 3
        path = os.path.join(TABLES, fname)
        write_table(path, n, cb_acc, ob_acc, seed=abs(hash(fname)) % 10000)
        print(f"  {fname:55s} n={n}")

    print(f"\nWrote {len(all_tables)} tables to {TABLES}/")
    print("\nNow run:  cd analysis && python run_all.py --fast")


if __name__ == "__main__":
    main()
