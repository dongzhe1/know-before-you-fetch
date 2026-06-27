# Know Before You Fetch

Code and data for *"Calibrated Confidence for Retrieval-Budget Allocation"* (WSDM 2027).

## What this is

Standard RAG retrieves a fixed number of passages for every query. We ask: can the model's own **calibrated closed-book confidence** decide *how much* retrieval each query needs?

We calibrate sequence log-probability into a probability of correctness via out-of-fold logistic regression, then allocate a **graded retrieval budget** per query:

| Action | Condition |
|--------|-----------|
| **Skip** | Model is confident — answer closed-book, no retrieval |
| **k=1** | Borderline — compact retrieval |
| **k=5** | Uncertain — full retrieval |
| **Abstain** | Neither source is reliable — selective prediction |

**Key findings:**
- The graded policy dominates every binary gate (including TARG prefix-entropy) under paired bootstrap: +0.040 frontier AUC on TriviaQA-8B, +0.016 on TriviaQA-32B
- Proper chosen-answer confidence (closed-book if skipped, open-book if retrieved) beats naive closed-book-only abstention on TriviaQA and NQ; MS-MARCO is a low-headroom negative case
- An honest two-stage cost model reveals a break-even condition: `skip_rate > c_CB / (c_ret + c_OB)`. At Qwen3-8B with cheap retrieval the gate is ~47% *slower*; at Qwen3-32B (54% skip rate) it saves ~4%
- The operational skip rate scales from ~0.5% → 33.5% → 54% across 1.7B → 8B → 32B

## Repository layout

```
know-before-you-fetch/
├── README.md
├── requirements.txt
├── analysis/                  # all CPU analysis — run everything from here
│   ├── _tables.py             # path resolver (all scripts import this)
│   ├── frontier_metrics.py    # core: binary/graded AUC, bootstrap CI
│   ├── experiment_registry.py # metadata for every table file
│   ├── figure_style.py        # shared matplotlib style
│   │
│   ├── make_manifest.py       # → results_manifest.csv (all paper numbers + CI)
│   ├── make_data.py           # → data/figdata_*.csv
│   ├── make_cost_metrics.py   # → data/cost_metrics.csv
│   ├── scaleup_summary.py     # → data/figdata_scaleup.csv
│   │
│   ├── targ_graded_baseline.py      # TARG signals + our calibration framework
│   ├── selective_rag.py             # proper vs naive abstention (AURC)
│   ├── cross_domain_selective.py    # cross-domain calibration transfer
│   ├── calibration_evidence.py      # ECE / Brier / NLL / reliability diagram
│   ├── calibration_methods.py       # logistic vs isotonic vs temperature scaling
│   ├── calibration_size_sensitivity.py
│   ├── deployment_eval.py           # held-out threshold protocol
│   ├── retriever_robustness.py      # bge-large vs bge-small vs DPR vs shared-corpus
│   ├── feature_ablation.py          # single-feature AUROC ablation
│   ├── multihop_analysis.py         # HotpotQA multi-hop routing
│   ├── decision_theoretic.py        # argmax utility policy
│   ├── retrieval_harm.py            # 2×2 harm/rescue matrix
│   ├── k10_degradation.py           # why k=10 ≤ k=5
│   ├── token_cost_comparison.py     # TARG token accounting
│   ├── wiki_vs_pool_comparison.py   # Wikipedia-scale vs per-question-pool
│   ├── unified_decision.py          # three-action utility sweep
│   │
│   ├── fig1_frontier.py      # cost–quality frontier (main figure)
│   ├── fig2_scaling.py       # scaling with model size
│   ├── fig3_selective.py     # selective RAG risk-coverage curves
│   ├── fig4_ternary.py       # ternary operating space
│   ├── fig5_baselines.py     # vs Self-RAG / Adaptive-RAG
│   ├── fig6_regime.py        # cost regime map
│   ├── fig7_generalization.py # cross-family / cross-dataset
│   ├── fig8_difficulty.py    # difficulty stratification
│   ├── fig9_motivation.py    # PopQA entity-popularity motivation
│   │
│   ├── run_all.py            # one-shot reproduction
│   ├── validate_outputs.py   # consistency checks
│   ├── results_manifest.csv  # pre-computed manifest
│   │
│   ├── tables/               # per-query JSONL data (31 files, ~9 MB)
│   ├── data/                 # derived figdata CSVs + timing benchmarks
│   └── figures/              # output PDFs (fig1–fig9)
│
└── scripts/                   # GPU data-generation scripts (and env tester)
    ├── make_sample_data.py    # generate/download sample data for env check (CPU)
    ├── prepare_qa.py          # download and format raw QA datasets
    ├── openbook_frontier_probe.py  # main closed/open-book probe (GPU)
    ├── targ_baseline.py       # extract prefix-logit TARG features (GPU)
    ├── timing_benchmark.py    # measure LLM latency (GPU)
    └── retriever_benchmark.py # measure retriever latency
```

## Quickstart

```bash
git clone https://github.com/YOUR-ORG/know-before-you-fetch
cd know-before-you-fetch
pip install -r requirements.txt
```

### Verify your environment (no GPU, no internet required)

Generate 200 synthetic rows per table and run the full analysis pipeline in fast mode (~1–2 min):

```bash
python scripts/make_sample_data.py        # writes synthetic tables to analysis/tables/
cd analysis
python run_all.py --fast                  # B=200 bootstrap, all scripts
python validate_outputs.py               # consistency checks
```

To use 50 real TriviaQA questions instead (requires `pip install datasets`):

```bash
python scripts/make_sample_data.py --real --n 50
```

Numbers will differ from the paper (fake data), but all scripts should complete without errors.

### Reproduce all figures and tables from pre-computed data

The `analysis/tables/` directory contains all pre-computed per-query results. Run everything on CPU:

```bash
cd analysis

# Full reproduction (bootstrap B=1000, ~5–10 min)
python run_all.py

# Fast smoke test (B=200, ~1–2 min)
python run_all.py --fast

# Validate consistency
python validate_outputs.py
```

Individual steps:

```bash
cd analysis

python make_manifest.py --bootstrap 1000  # results_manifest.csv
python make_data.py                        # all figdata CSVs
python targ_graded_baseline.py             # TARG signal comparison
python selective_rag.py                    # selective RAG curves
python fig1_frontier.py                    # regenerate figure 1
# etc.
```

## Regenerate data from scratch (requires GPU)

If you want to reproduce the JSONL tables themselves rather than using the pre-computed ones:

**Step 1 — Download raw QA datasets (CPU, internet)**

```bash
pip install datasets transformers

# TriviaQA with passage pools
python scripts/prepare_qa.py --task triviaqa_rc --n 600 --out data/triviaqa_rc.jsonl

# NQ with DPR pools
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz \
     -O data/biencoder-nq-dev.json.gz
python scripts/prepare_qa.py --task nq_dpr --n 600 --out data/nq_dpr.jsonl

# MS-MARCO, HotpotQA, PopQA
python scripts/prepare_qa.py --task msmarco   --n 600 --out data/msmarco.jsonl
python scripts/prepare_qa.py --task hotpotqa  --n 600 --out data/hotpotqa.jsonl
python scripts/prepare_qa.py --task popqa     --n 800 --out data/popqa.jsonl
```

**Step 2 — Run frontier probe (GPU, ~24 GB VRAM for 8B)**

```bash
# Main frontier tables
python scripts/openbook_frontier_probe.py \
    --data data/triviaqa_rc.jsonl \
    --model /path/to/Qwen3-8B \
    --encoder /path/to/bge-large-en-v1.5 \
    --n 600 --k_small 1 --k_large 5 \
    --dump analysis/tables/triviaqa_rc_table.jsonl

# TARG prefix-logit features
python scripts/openbook_frontier_probe.py \
    --data data/triviaqa_rc.jsonl \
    --model /path/to/Qwen3-8B \
    --encoder /path/to/bge-large-en-v1.5 \
    --n 600 --targ_features \
    --dump analysis/tables/triviaqa_rc_targ_features.jsonl

# Open-book confidence tables (for selective RAG)
python scripts/openbook_frontier_probe.py \
    --data data/triviaqa_rc.jsonl \
    --model /path/to/Qwen3-8B \
    --encoder /path/to/bge-large-en-v1.5 \
    --n 600 --dump_open_conf \
    --dump analysis/tables/triviaqa_rc_openconf_table.jsonl
```

**Step 3 — Timing benchmarks (GPU)**

```bash
python scripts/timing_benchmark.py \
    --model /path/to/Qwen3-8B \
    --out analysis/data/timing_raw.csv

python scripts/retriever_benchmark.py \
    --encoder /path/to/bge-large-en-v1.5 \
    --out analysis/data/retriever_benchmark.csv
```

## Data format

Every JSONL row in `analysis/tables/` contains at minimum:

| Field | Type | Description |
|-------|------|-------------|
| `question` | str | Query text |
| `gold` | str | Gold answer |
| `closed_pred` | str | Model's closed-book prediction |
| `closed_correct` | 0/1 | Whether closed-book answer is correct |
| `seq_logprob` | float | Raw sequence log-probability |
| `p_correct` | float | Calibrated probability of correctness (OOF logistic) |
| `open_correct_k1` | 0/1 | Open-book correctness at k=1 |
| `open_correct_k5` | 0/1 | Open-book correctness at k=5 |

TARG feature tables add `mean_entropy`, `top1_top2_margin` (`prefix_variance` is all-NaN, excluded). Open-confidence tables add `open_seq_logprob_k5`.

## Key numbers

| Metric | Qwen3-8B | Qwen3-32B |
|--------|----------|-----------|
| CB accuracy (TriviaQA) | 0.572 | 0.695 |
| OB@5 accuracy | 0.785 | 0.822 |
| Gate AUROC | 0.795 | 0.812 |
| Graded frontier AUC (full-context) | 0.773 | 0.810 |
| Skip rate at matched accuracy | 33.5% | 54% |
| Gate vs always-RAG latency | +47% (slower) | −4% (faster) |
| Normalized gate value | 0.43 | 0.42 |

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20954595.svg)](https://doi.org/10.5281/zenodo.20954595)

## License

Code: MIT.  
Data tables are derived from TriviaQA, Natural Questions, MS-MARCO, HotpotQA, and PopQA — each subject to its original license.
