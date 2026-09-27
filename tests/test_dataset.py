from pathlib import Path

import numpy as np

from worldsense3d.data import WorldSenseDataset


def test_dataset_shapes(tmp_path: Path):
    split = tmp_path / "train"
    split.mkdir()
    np.savez_compressed(
        split / "episode_000000.npz",
        rgb=np.zeros((9, 32, 32, 3), dtype=np.uint8),
        depth=np.ones((9, 32, 32), dtype=np.float32),
        segmentation=np.zeros((9, 32, 32), dtype=np.uint8),
        states=np.zeros((9, 5, 13), dtype=np.float32),
        action=np.zeros(6, dtype=np.float32),
        object_mask=np.array([1, 1, 1, 0, 0], dtype=np.float32),
        target_index=np.asarray(1, dtype=np.int64),
        num_objects=np.asarray(3, dtype=np.int64),
    )
    ds = WorldSenseDataset(tmp_path, "train", context_frames=4, future_frames=5)
    x = ds[0]
    assert x["rgb"].shape == (4, 3, 32, 32)
    assert x["depth"].shape == (4, 1, 32, 32)
    assert x["state"].shape == (5, 13)
    assert x["target_positions"].shape == (5, 5, 3)
