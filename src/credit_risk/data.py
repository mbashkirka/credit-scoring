from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re

import pandas as pd


def find_target_file(data_dir: Path) -> Path:
    matches = list(data_dir.rglob("train_target.csv"))
    if not matches:
        raise FileNotFoundError(
            f"train_target.csv not found under {data_dir}. "
            "Download and extract the Kaggle competition data first."
        )
    return matches[0]


def find_train_history_files(data_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("*.pq", "*.parquet"):
        candidates.extend(data_dir.rglob(pattern))

    result: list[Path] = []
    for path in candidates:
        name = path.name.lower()
        if "test" in name:
            continue
        if "train" in name:
            result.append(path)

    def train_chunk_key(path: Path) -> tuple[int, str]:
        match = re.search(r"train_data_(\d+)", path.stem)
        if match:
            return int(match.group(1)), str(path)
        return 10**9, str(path)

    # Numeric chunk order matters for --max-files: 0,1,2,... rather than
    # lexicographic 0,1,10,11,2,... . The M1/M2 one-file runs were unaffected,
    # but M2.1 fixes this before we scale to multiple chunks.
    result = sorted(set(result), key=train_chunk_key)
    if not result:
        raise FileNotFoundError(
            f"No train parquet files found under {data_dir}. "
            "Expected extracted competition train-history files."
        )
    return result


def load_target(data_dir: Path) -> pd.DataFrame:
    target = pd.read_csv(find_target_file(data_dir))
    required = {"id", "flag"}
    missing = required - set(target.columns)
    if missing:
        raise ValueError(f"Target file misses required columns: {sorted(missing)}")
    if target["id"].duplicated().any():
        raise ValueError("Target must contain exactly one row per application id.")
    return target


def iter_train_history(
    data_dir: Path,
    max_files: int | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield one raw history parquet at a time to keep peak RAM bounded."""
    files = find_train_history_files(data_dir)
    if max_files is not None:
        files = files[:max_files]

    for path in files:
        print(f"Reading {path}")
        frame = pd.read_parquet(path)
        if "id" not in frame.columns:
            raise ValueError(f"History file {path} has no 'id' column.")
        yield path, frame


def load_train_history(data_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    """Compatibility loader for small smoke tests; M2 uses iter_train_history."""
    frames = [frame for _, frame in iter_train_history(data_dir, max_files=max_files)]
    return pd.concat(frames, ignore_index=True)
