import numpy as np
import pytest
import torch

from src.dataset import build_splits, compute_channel_stats, stratified_split, to_normalized_tensor
from tests.conftest import synthetic_arrays


def test_sample_counts(splits):
    assert len(splits.train) == 500
    assert len(splits.val) == 100
    assert len(splits.test) == 200


def test_split_is_disjoint_and_complete(splits):
    train, val = set(splits.train_indices.tolist()), set(splits.val_indices.tolist())
    assert not train & val
    assert train | val == set(range(600))


def test_split_is_stratified(arrays, splits):
    _, train_y, _, _ = arrays
    val_counts = np.bincount(train_y[splits.val_indices], minlength=10)
    train_counts = np.bincount(train_y[splits.train_indices], minlength=10)
    assert (val_counts == 10).all() and (train_counts == 50).all()


def test_stratified_split_cifar_sized_labels():
    labels = np.repeat(np.arange(10), 5000)
    train_idx, val_idx = stratified_split(labels, 5000, seed=42)
    assert len(train_idx) == 45000 and len(val_idx) == 5000
    assert (np.bincount(labels[val_idx]) == 500).all()


def test_stratified_split_unbalanced_preserves_proportion():
    labels = np.array([0] * 700 + [1] * 200 + [2] * 100)
    _, val_idx = stratified_split(labels, 100, seed=1)
    assert np.bincount(labels[val_idx]).tolist() == [70, 20, 10]


def test_split_reproducible(arrays):
    a = build_splits(*arrays, train_size=500, val_size=100, split_seed=7)
    b = build_splits(*arrays, train_size=500, val_size=100, split_seed=7)
    c = build_splits(*arrays, train_size=500, val_size=100, split_seed=8)
    assert np.array_equal(a.val_indices, b.val_indices)
    assert a.split_fingerprint() == b.split_fingerprint() != c.split_fingerprint()
    assert torch.equal(a.train.tensors[0], b.train.tensors[0])


def test_labels_match_num_classes(splits):
    for ds in (splits.train, splits.val, splits.test):
        labels = ds.tensors[1]
        assert labels.dtype == torch.int64
        assert labels.min() >= 0 and labels.max() < splits.num_classes == 10


def test_tensor_shapes_and_dtype(splits):
    x = splits.train.tensors[0]
    assert x.shape == (500, 3, 32, 32) and x.dtype == torch.float32


def test_normalization_uses_training_data_only(arrays, splits):
    train_x, _, _, _ = arrays
    mean, std = compute_channel_stats(train_x[splits.train_indices])
    assert np.allclose(splits.mean, mean) and np.allclose(splits.std, std)
    x = splits.train.tensors[0]
    assert torch.allclose(x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(x.std(dim=(0, 2, 3), unbiased=False), torch.ones(3), atol=1e-4)


def test_normalization_stats_ignore_validation_and_test():
    train_x, train_y, test_x, test_y = synthetic_arrays()
    base = build_splits(train_x, train_y, test_x, test_y, 500, 100, split_seed=3)
    altered_train = train_x.copy()
    altered_train[base.val_indices] = 255  # altera somente a validação
    altered = build_splits(altered_train, train_y, np.zeros_like(test_x), test_y, 500, 100, split_seed=3)
    assert base.mean == altered.mean and base.std == altered.std


def test_normalization_matches_torchvision_transforms():
    from torchvision import transforms

    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    mean, std = (0.49, 0.48, 0.45), (0.25, 0.24, 0.26)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    expected = torch.stack([tf(img) for img in imgs])
    assert torch.allclose(to_normalized_tensor(imgs, mean, std), expected, atol=1e-6)


def test_invalid_sizes_raise(arrays):
    with pytest.raises(ValueError):
        build_splits(*arrays, train_size=450, val_size=100, split_seed=0)


def test_test_set_is_untouched(arrays, splits):
    _, _, _, test_y = arrays
    assert torch.equal(splits.test.tensors[1], torch.from_numpy(test_y))
