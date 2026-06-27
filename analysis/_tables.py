"""Path resolver for tables/, data/, and figures/ directories."""

from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def table_path(name: str) -> Path:
    """Resolve a table JSONL file. Checks tables/ then ../logs/ as fallback."""
    p = TABLES / name
    if p.exists():
        return p
    fallback = ROOT.parent / "logs" / name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Table not found: {name} (looked in {TABLES} and {ROOT.parent / 'logs'})"
    )


def data_path(name: str) -> Path:
    """Resolve or create a file under data/."""
    DATA.mkdir(exist_ok=True)
    return DATA / name


def figure_path(name: str) -> Path:
    """Resolve or create a file under figures/."""
    FIGURES.mkdir(exist_ok=True)
    return FIGURES / name


def resolve(fname: str) -> str:
    """Backward-compatible string-path resolver. Returns str path."""
    return str(table_path(fname))
