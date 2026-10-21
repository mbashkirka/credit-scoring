from pathlib import Path
import pandas as pd

from credit_risk.data import (
    find_target_file,
    find_train_history_files,
    load_target,
)

DATA_DIR = Path("data/raw")


def main() -> None:
    target_path = find_target_file(DATA_DIR)
    history_files = find_train_history_files(DATA_DIR)
    target = load_target(DATA_DIR)

    print("\n=== TARGET ===")
    print("file:", target_path)
    print("shape:", target.shape)
    print("columns:", target.columns.tolist())
    print("\nflag distribution:")
    print(target["flag"].value_counts(dropna=False).sort_index())
    print("\ndefault rate:", round(float(target["flag"].mean()), 6))

    print("\n=== TRAIN HISTORY FILES ===")
    print("count:", len(history_files))
    for p in history_files[:10]:
        print("-", p)

    first = history_files[0]
    sample = pd.read_parquet(first)

    print("\n=== FIRST HISTORY CHUNK ===")
    print("file:", first)
    print("shape:", sample.shape)
    print("columns:")
    for col in sample.columns:
        print(f"  - {col}: {sample[col].dtype}")

    print("\nhead:")
    print(sample.head())


if __name__ == "__main__":
    main()
