"""Canonical paths for SafeNull (nested under AlphaEdit repo)."""

from pathlib import Path
from typing import Union

SAFENULL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = SAFENULL_ROOT / "results" / "safenull_run"


def resolve_results_dir(path: Union[str, Path]) -> Path:
    """Resolve output dir relative to SafeNull root when path is relative."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (SAFENULL_ROOT / p).resolve()
