import os
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "corn"


def resolve_data_dir(data_dir=None):
    """Resolve the corn H5 dataset directory.

    Priority: explicit ``data_dir`` argument -> ``CORN_DATA_DIR`` environment
    variable -> repository-local ``data/corn`` directory.
    """
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    env_dir = os.environ.get("CORN_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_DATA_DIR.resolve()


def load_data(partition, data_dir=None):
    """Load corn point-cloud data from H5 files.

    Expected files are ``ply_data_train.h5`` and ``ply_data_test.h5`` with
    datasets named ``data`` and ``label``.
    """
    if partition not in {"train", "test"}:
        raise ValueError("partition must be 'train' or 'test'")

    root = resolve_data_dir(data_dir)
    h5_path = root / f"ply_data_{partition}.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {h5_path}. "
            "Pass --data_dir, set CORN_DATA_DIR, or place the H5 files in data/corn/."
        )

    with h5py.File(h5_path, "r") as f:
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")

    if len(label.shape) == 2:
        label = label.squeeze(1)
    return data, label


def random_point_dropout(pc, max_dropout_ratio=0.2):
    """Randomly replace dropped points with the first point."""
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random(pc.shape[0]) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    return np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    n, c = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(n, c), -clip, clip)
    return pointcloud


class CornDataset(Dataset):
    """Corn point-cloud dataset used for the two-class growth-stage model."""

    def __init__(self, num_points, partition="train", data_dir=None):
        self.data, self.label = load_data(partition, data_dir=data_dir)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points].copy()
        label = self.label[item]
        if self.partition == "train":
            pointcloud = random_point_dropout(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    train = CornDataset(4096)
    for count, (data, label) in enumerate(train, start=1):
        print("point cloud shape:", data.shape)
        print("label shape:", np.asarray(label).shape)
        if count >= 15:
            break
