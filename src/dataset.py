"""Carregamento, divisão estratificada e normalização do CIFAR-10.

Decisões:
- O conjunto oficial de treino (50.000) é dividido de forma ESTRATIFICADA em
  treino (45.000) e validação (5.000) com uma semente própria (`split_seed`).
- Média e desvio padrão por canal RGB são calculados EXCLUSIVAMENTE sobre as
  45.000 imagens de treino e reutilizados em validação e teste.
- O desvio padrão é o populacional (ddof=0) sobre todos os pixels do canal.
- O pré-processamento equivale a `ToTensor()` + `Normalize(mean, std)`:
  x = (pixel / 255 - mean) / std, em float32, layout NCHW. Como não há data
  augmentation, os tensores são pré-computados uma única vez em memória.
- O conjunto de teste (10.000) é mantido intacto e nunca é usado no
  treinamento, na seleção de hiperparâmetros ou do checkpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import make_generator, resolve_path, seed_worker

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_TRAIN_TOTAL = 50000
CIFAR10_TEST_TOTAL = 10000


@dataclass
class DataSplits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    mean: "tuple[float, float, float]"
    std: "tuple[float, float, float]"
    train_indices: np.ndarray
    val_indices: np.ndarray
    class_names: "tuple[str, ...]" = CIFAR10_CLASSES
    metadata: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def split_fingerprint(self) -> str:
        """Hash dos índices de treino/validação (verifica que a divisão não mudou)."""
        h = hashlib.sha256()
        h.update(np.sort(self.train_indices).astype(np.int64).tobytes())
        h.update(np.sort(self.val_indices).astype(np.int64).tobytes())
        return h.hexdigest()[:16]


def stratified_split(labels: np.ndarray, val_size: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Divide índices em treino/validação preservando a proporção de cada classe.

    A cota de validação por classe é proporcional à sua frequência; o resto da
    divisão inteira é distribuído às classes com maior parte fracionária
    (método de Hamilton). No CIFAR-10 (5.000 imagens por classe) isso resulta
    em exatamente 500 imagens de validação por classe.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if not 0 < val_size < n:
        raise ValueError(f"val_size deve estar em (0, {n}); recebido {val_size}.")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)

    exact = counts * val_size / n
    quotas = np.floor(exact).astype(int)
    remainder = val_size - quotas.sum()
    if remainder > 0:
        order = np.argsort(-(exact - quotas), kind="stable")
        quotas[order[:remainder]] += 1

    train_parts, val_parts = [], []
    for cls, quota in zip(classes, quotas):
        idx = np.flatnonzero(labels == cls)
        idx = rng.permutation(idx)
        val_parts.append(idx[:quota])
        train_parts.append(idx[quota:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    return train_idx, val_idx


def compute_channel_stats(images: np.ndarray) -> "tuple[tuple[float, ...], tuple[float, ...]]":
    """Média e desvio (ddof=0) por canal de imagens uint8 NHWC, na escala [0, 1]."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Esperado array NHWC com 3 canais; recebido {images.shape}.")
    means, stds = [], []
    for c in range(images.shape[-1]):
        channel = images[..., c].astype(np.float64) / 255.0
        means.append(float(channel.mean()))
        stds.append(float(channel.std()))
    return tuple(means), tuple(stds)


def to_normalized_tensor(images: np.ndarray, mean, std) -> torch.Tensor:
    """uint8 NHWC -> float32 NCHW normalizado ((x/255 - mean) / std)."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    return x.sub_(mean_t).div_(std_t).contiguous()


def build_splits(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    train_size: int,
    val_size: int,
    split_seed: int,
    normalize: bool = True,
    stratified: bool = True,
    class_names: "tuple[str, ...]" = CIFAR10_CLASSES,
) -> DataSplits:
    """Constrói os três conjuntos a partir de arrays brutos (uint8 NHWC).

    Função pura e independente de download: é a mesma usada nos testes com
    dados sintéticos.
    """
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if train_size + val_size != len(train_images):
        raise ValueError(
            f"train_size ({train_size}) + validation_size ({val_size}) deve ser igual ao "
            f"total do treino original ({len(train_images)})."
        )
    if stratified:
        train_idx, val_idx = stratified_split(train_labels, val_size, split_seed)
    else:
        perm = np.random.default_rng(split_seed).permutation(len(train_images))
        train_idx, val_idx = perm[val_size:], perm[:val_size]

    train_raw = train_images[train_idx]
    if normalize:
        mean, std = compute_channel_stats(train_raw)  # somente dados de treino
    else:
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def make(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
        return TensorDataset(to_normalized_tensor(images, mean, std), torch.from_numpy(labels))

    return DataSplits(
        train=make(train_raw, train_labels[train_idx]),
        val=make(train_images[val_idx], train_labels[val_idx]),
        test=make(test_images, test_labels),
        mean=mean,
        std=std,
        train_indices=train_idx,
        val_indices=val_idx,
        class_names=tuple(class_names),
        metadata={
            "split_seed": split_seed,
            "stratified": stratified,
            "normalize": normalize,
            "train_size": int(len(train_idx)),
            "validation_size": int(len(val_idx)),
            "test_size": int(len(test_labels)),
        },
    )


def load_cifar10_arrays(data_dir: "str | Path", download: bool = True):
    """Arrays brutos do CIFAR-10 via torchvision (uint8 NHWC + rótulos)."""
    from torchvision.datasets import CIFAR10

    root = str(resolve_path(data_dir))
    train = CIFAR10(root=root, train=True, download=download)
    test = CIFAR10(root=root, train=False, download=download)
    return (
        np.asarray(train.data, dtype=np.uint8),
        np.asarray(train.targets, dtype=np.int64),
        np.asarray(test.data, dtype=np.uint8),
        np.asarray(test.targets, dtype=np.int64),
        tuple(train.classes),
    )


_SPLIT_CACHE: "dict[tuple, DataSplits]" = {}


def load_cifar10_splits(config: dict) -> DataSplits:
    """Carrega o CIFAR-10 e aplica a divisão/normalização definida na configuração.

    O resultado é mantido em cache no processo, pois a divisão é idêntica
    entre as repetições (mesmo `split_seed`).
    """
    ds = config["dataset"]
    if ds.get("name", "cifar10") != "cifar10":
        raise ValueError(f"Dataset não suportado: {ds.get('name')!r}")
    key = (
        str(resolve_path(ds.get("data_dir", "data"))),
        int(ds["train_size"]),
        int(ds["validation_size"]),
        int(ds["split_seed"]),
        bool(ds.get("normalize", True)),
        bool(ds.get("stratified", True)),
    )
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]

    train_x, train_y, test_x, test_y, classes = load_cifar10_arrays(ds.get("data_dir", "data"), ds.get("download", True))
    if len(test_x) != int(ds.get("test_size", CIFAR10_TEST_TOTAL)):
        raise ValueError(f"Conjunto de teste com {len(test_x)} amostras; esperado {ds.get('test_size')}.")
    splits = build_splits(
        train_x,
        train_y,
        test_x,
        test_y,
        train_size=int(ds["train_size"]),
        val_size=int(ds["validation_size"]),
        split_seed=int(ds["split_seed"]),
        normalize=bool(ds.get("normalize", True)),
        stratified=bool(ds.get("stratified", True)),
        class_names=classes,
    )
    _SPLIT_CACHE[key] = splits
    return splits


def make_dataloaders(
    splits: DataSplits,
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> "dict[str, DataLoader]":
    """DataLoaders de treino (embaralhado com gerador semeado), validação e teste."""
    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        common["worker_init_fn"] = seed_worker
    return {
        "train": DataLoader(
            splits.train,
            batch_size=train_batch_size,
            shuffle=True,
            generator=make_generator(seed),
            drop_last=False,
            **common,
        ),
        "val": DataLoader(splits.val, batch_size=eval_batch_size, shuffle=False, **common),
        "test": DataLoader(splits.test, batch_size=eval_batch_size, shuffle=False, **common),
    }


def dataloaders_from_config(splits: DataSplits, config: dict, seed: int, device: torch.device) -> "dict[str, DataLoader]":
    return make_dataloaders(
        splits,
        train_batch_size=int(config["training"]["batch_size"]),
        eval_batch_size=int(config["evaluation"]["batch_size"]),
        seed=seed,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
