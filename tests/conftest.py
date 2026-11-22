"""Fixtures compartilhadas: dados sintéticos no formato do CIFAR-10 (sem download)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.dataset import build_splits, make_dataloaders
from src.utils import configure_fp32, load_config

ROOT = Path(__file__).resolve().parent.parent

configure_fp32()


def synthetic_arrays(n_train=600, n_test=200, num_classes=10, seed=0):
    """Imagens uint8 NHWC 32x32x3 com classes balanceadas."""
    rng = np.random.default_rng(seed)
    train_x = rng.integers(0, 256, size=(n_train, 32, 32, 3), dtype=np.uint8)
    test_x = rng.integers(0, 256, size=(n_test, 32, 32, 3), dtype=np.uint8)
    train_y = np.tile(np.arange(num_classes), n_train // num_classes)
    test_y = np.tile(np.arange(num_classes), n_test // num_classes)
    return train_x, rng.permutation(train_y), test_x, test_y


@pytest.fixture(scope="session")
def arrays():
    return synthetic_arrays()


@pytest.fixture(scope="session")
def splits(arrays):
    train_x, train_y, test_x, test_y = arrays
    return build_splits(train_x, train_y, test_x, test_y, train_size=500, val_size=100, split_seed=42)


@pytest.fixture()
def loaders(splits):
    return make_dataloaders(splits, train_batch_size=64, eval_batch_size=64, seed=0)


@pytest.fixture()
def tiny_config():
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    cfg["training"]["epochs"] = 2
    cfg["training"]["batch_size"] = 64
    cfg["evaluation"].update({"batch_size": 64, "warmup_iterations": 2, "benchmark_iterations": 5})
    return cfg
