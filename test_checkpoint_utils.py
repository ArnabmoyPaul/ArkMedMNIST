"""test_checkpoint_utils.py — Run: python test_checkpoint_utils.py"""
import os
import random
import shutil
import tempfile

import numpy as np
import torch

from checkpoint_utils import (
    save_checkpoint_atomic, load_checkpoint_with_fallback,
    capture_rng_state, restore_rng_state,
)


def test_round_trip():
    d = tempfile.mkdtemp()
    try:
        latest, prev = os.path.join(d, "latest.pth"), os.path.join(d, "prev.pth")
        state = {"epoch": 3, "dataset_index": 7, "weights": torch.randn(10)}
        save_checkpoint_atomic(state, latest, prev)
        loaded = load_checkpoint_with_fallback(latest, prev)
        assert loaded["epoch"] == 3 and loaded["dataset_index"] == 7
        assert torch.allclose(loaded["weights"], state["weights"])
    finally:
        shutil.rmtree(d)


def test_falls_back_to_prev_when_latest_is_corrupt():
    d = tempfile.mkdtemp()
    try:
        latest, prev = os.path.join(d, "latest.pth"), os.path.join(d, "prev.pth")
        save_checkpoint_atomic({"epoch": 1, "dataset_index": 0}, latest, prev)
        save_checkpoint_atomic({"epoch": 2, "dataset_index": 0}, latest, prev)  # prev now holds epoch=1
        with open(latest, "wb") as f:
            f.write(b"not a valid checkpoint")  # simulate a power cut mid-write
        loaded = load_checkpoint_with_fallback(latest, prev)
        assert loaded is not None, "should have fallen back to prev, not returned None"
        assert loaded["epoch"] == 1, f"expected fallback to prev (epoch=1), got {loaded['epoch']}"
    finally:
        shutil.rmtree(d)


def test_rng_round_trip_reproduces_next_draws():
    rng = capture_rng_state()
    expected_torch = torch.rand(5)
    expected_np = np.random.rand(5)
    expected_py = [random.random() for _ in range(5)]

    torch.rand(100); np.random.rand(100); [random.random() for _ in range(100)]  # perturb state

    restore_rng_state(rng)
    assert torch.allclose(torch.rand(5), expected_torch)
    assert np.allclose(np.random.rand(5), expected_np)
    assert [random.random() for _ in range(5)] == expected_py


if __name__ == "__main__":
    test_round_trip()
    test_falls_back_to_prev_when_latest_is_corrupt()
    test_rng_round_trip_reproduces_next_draws()
    print("test_checkpoint_utils.py: all checks passed")
