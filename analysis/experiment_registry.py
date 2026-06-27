"""Registry mapping experiment keys to table files and metadata."""

from __future__ import annotations
from typing import Dict, Optional

# Canonical tables. Key format: <dataset>_<model>_<retriever>
TABLE_REGISTRY: Dict[str, dict] = {
    # === TriviaQA-rc main ===
    "triviaqa_qwen1.7b_bgelarge": {
        "dataset": "TriviaQA-rc",
        "model": "Qwen3-1.7B",
        "retriever": "bge-large",
        "table": "triviaqa_rc_qwen1p7b_table.jsonl",
        "n_expected": 600,
        "role": "scaling",
    },
    "triviaqa_qwen8b_bgelarge": {
        "dataset": "TriviaQA-rc",
        "model": "Qwen3-8B",
        "retriever": "bge-large",
        "table": "triviaqa_rc_table.jsonl",
        "ksweep_table": "triviaqa_rc_ksweep_table.jsonl",
        "openconf_table": "triviaqa_rc_openconf_table.jsonl",
        "targ_features": "triviaqa_rc_targ_features.jsonl",
        "n_expected": 600,
        "role": "main",
    },
    "triviaqa_qwen8b_bgesmall": {
        "dataset": "TriviaQA-rc",
        "model": "Qwen3-8B",
        "retriever": "bge-small",
        "table": "triviaqa_rc_bgesmall_table.jsonl",
        "n_expected": 600,
        "role": "retriever_ablation",
    },
    "triviaqa_qwen8b_shared": {
        "dataset": "TriviaQA-rc",
        "model": "Qwen3-8B",
        "retriever": "shared-corpus",
        "table": "triviaqa_rc_shared_table.jsonl",
        "n_expected": 600,
        "role": "robustness",
    },
    "triviaqa_qwen32b_bgelarge": {
        "dataset": "TriviaQA-rc",
        "model": "Qwen3-32B",
        "retriever": "bge-large",
        "table": "triviaqa_rc_32b_n600_table.jsonl",
        "ksweep_table": "triviaqa_rc_ksweep_32b_table.jsonl",
        "openconf_table": "triviaqa_rc_openconf_32b_table.jsonl",
        "targ_features": "triviaqa_rc_targ_features_32b.jsonl",
        "n_expected": 600,
        "role": "scaling",
    },
    "triviaqa_qwen35_9b_bgelarge": {
        "dataset": "TriviaQA-rc",
        "model": "Qwen3.5-9B",
        "retriever": "bge-large",
        "table": "triviaqa_rc_qwen35_9b_table.jsonl",
        "openconf_table": "triviaqa_rc_openconf_qwen35_table.jsonl",
        "n_expected": 600,
        "role": "cross_family",
    },
    "triviaqa_llama8b_bgelarge": {
        "dataset": "TriviaQA-rc",
        "model": "Llama-3.1-8B",
        "retriever": "bge-large",
        "table": "triviaqa_rc_llama_table.jsonl",
        "openconf_table": "triviaqa_rc_openconf_llama_table.jsonl",
        "n_expected": 600,
        "role": "cross_family",
    },
    # === NQ-DPR ===
    "nq_qwen8b_dpr": {
        "dataset": "NQ-DPR",
        "model": "Qwen3-8B",
        "retriever": "DPR",
        "table": "nq_dpr_table.jsonl",
        "openconf_table": "nq_dpr_openconf_table.jsonl",
        "targ_features": "nq_dpr_targ_features.jsonl",
        "n_expected": 600,
        "role": "main",
    },
    "nq_qwen8b_shared": {
        "dataset": "NQ-DPR",
        "model": "Qwen3-8B",
        "retriever": "shared-corpus",
        "table": "nq_dpr_shared_table.jsonl",
        "n_expected": 600,
        "role": "robustness",
    },
    "nq_qwen32b_dpr": {
        "dataset": "NQ-DPR",
        "model": "Qwen3-32B",
        "retriever": "DPR",
        "table": "nq_dpr_32b_table.jsonl",
        "targ_features": "nq_dpr_targ_features_32b.jsonl",
        "n_expected": 300,
        "role": "scaling",
    },
    # === MS-MARCO ===
    "msmarco_qwen8b_passagepool": {
        "dataset": "MS-MARCO",
        "model": "Qwen3-8B",
        "retriever": "passage-pool",
        "table": "msmarco_table.jsonl",
        "openconf_table": "msmarco_openconf_table.jsonl",
        "targ_features": "msmarco_targ_features.jsonl",
        "n_expected": 600,
        "role": "main",
    },
    "msmarco_qwen8b_shared": {
        "dataset": "MS-MARCO",
        "model": "Qwen3-8B",
        "retriever": "shared-corpus",
        "table": "msmarco_shared_table.jsonl",
        "n_expected": 600,
        "role": "robustness",
    },
    # === Analysis datasets ===
    "popqa_qwen8b": {
        "dataset": "PopQA",
        "model": "Qwen3-8B",
        "retriever": "n/a",
        "table": "popqa_table.jsonl",
        "n_expected": 1000,
        "role": "motivation",
    },
    "hotpotqa_qwen8b": {
        "dataset": "HotpotQA",
        "model": "Qwen3-8B",
        "retriever": "passage-pool",
        "table": "hotpotqa_table.jsonl",
        "n_expected": 600,
        "role": "analysis",
    },
    # === NQ-32B TARG features (32B, not 8B main) ===
    "nq_qwen32b_dpr_targ": {
        "dataset": "NQ-DPR",
        "model": "Qwen3-32B",
        "retriever": "DPR",
        "table": "nq_dpr_32b_table.jsonl",
        "targ_features": "nq_dpr_targ_features_32b.jsonl",
        "n_expected": 300,
        "role": "targ_baseline",
    },
    "msmarco_qwen32b_passagepool_targ": {
        "dataset": "MS-MARCO",
        "model": "Qwen3-32B",
        "retriever": "passage-pool",
        "targ_features": "msmarco_targ_features_32b.jsonl",
        "n_expected": 600,
        "role": "targ_baseline",
    },
}

# Deprecated files — kept for reference, should NOT be used in main pipeline
DEPRECATED = {
    "triviaqa_rc_qwen8b_table.jsonl": "Duplicate of triviaqa_rc_bgesmall_table.jsonl (not triviaqa_rc_table.jsonl). Do not use.",
    "triviaqa_rc_32b_table.jsonl": "Old n=300 run. Use triviaqa_rc_32b_n600_table.jsonl.",
}


def get_experiment(key: str) -> Optional[dict]:
    """Look up experiment by registry key."""
    return TABLE_REGISTRY.get(key)


def get_experiments_by_role(role: str) -> list:
    """Return all experiments with a given role."""
    return [(k, v) for k, v in TABLE_REGISTRY.items() if v.get("role") == role]


def get_experiments_by_dataset(dataset: str) -> list:
    """Return all experiments for a dataset."""
    return [(k, v) for k, v in TABLE_REGISTRY.items() if v["dataset"] == dataset]


def is_deprecated(table_name: str) -> bool:
    """Check if a table filename is deprecated."""
    return table_name in DEPRECATED
