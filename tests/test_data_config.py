import os
from pathlib import Path

import h5py
import numpy as np

from training.data import load_data, resolve_data_dir


def _write_h5(path: Path):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros((2, 8, 3), dtype=np.float32))
        f.create_dataset("label", data=np.array([[0], [1]], dtype=np.int64))


def test_explicit_data_dir_is_used(tmp_path):
    _write_h5(tmp_path / "ply_data_train.h5")
    data, labels = load_data("train", data_dir=tmp_path)
    assert data.shape == (2, 8, 3)
    assert labels.tolist() == [0, 1]


def test_environment_data_dir_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("CORN_DATA_DIR", str(tmp_path))
    assert resolve_data_dir() == tmp_path.resolve()


def test_missing_dataset_has_actionable_error(tmp_path):
    try:
        load_data("test", data_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "ply_data_test.h5" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
