from pathlib import Path

from credit_risk.data import find_train_history_files


def test_train_history_files_use_numeric_chunk_order(tmp_path: Path):
    for index in (10, 2, 1, 0, 11):
        (tmp_path / f"train_data_{index}.pq").touch()

    result = find_train_history_files(tmp_path)
    assert [path.name for path in result] == [
        "train_data_0.pq",
        "train_data_1.pq",
        "train_data_2.pq",
        "train_data_10.pq",
        "train_data_11.pq",
    ]
