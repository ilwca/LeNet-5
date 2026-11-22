"""Métricas preditivas e de complexidade, independentes da arquitetura.

Todas as funções recebem um `nn.Module` genérico, de modo que modelos podados
ou quantizados no futuro possam ser avaliados pelo mesmo código.

Distinção entre métricas:
- MEDIDAS: acurácia, loss, precisão/recall/F1, matriz de confusão, tamanho do
  arquivo do checkpoint em disco.
- TEÓRICAS: memória ocupada pelos parâmetros (numel x bytes por elemento),
  que não inclui metadados de serialização nem buffers do runtime.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

BYTES_PER_MB = 1_000_000  # MB decimal (SI)
BYTES_PER_MIB = 1024 * 1024  # MiB binário


# --------------------------------------------------------------------------- #
# Inferência
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion: "nn.Module | None", device: torch.device):
    """Retorna (loss média por amostra, predições, rótulos) sobre um DataLoader."""
    model.eval()
    total_loss, total = 0.0, 0
    preds, targets = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
        total += labels.size(0)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    loss = total_loss / total if (criterion is not None and total) else float("nan")
    return loss, torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate_loss_accuracy(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    loss, preds, targets = predict(model, loader, criterion, device)
    return loss, accuracy(preds, targets)


# --------------------------------------------------------------------------- #
# Métricas preditivas
# --------------------------------------------------------------------------- #
def accuracy(preds, targets) -> float:
    preds, targets = np.asarray(preds), np.asarray(targets)
    if preds.shape != targets.shape:
        raise ValueError("preds e targets devem ter a mesma shape.")
    if preds.size == 0:
        raise ValueError("Não há amostras para calcular a acurácia.")
    return float((preds == targets).mean())


def confusion_matrix(preds, targets, num_classes: int) -> np.ndarray:
    """Matriz [verdadeiro, predito] com contagens absolutas."""
    preds, targets = np.asarray(preds, dtype=np.int64), np.asarray(targets, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def classification_metrics(preds, targets, num_classes: int, class_names=None) -> dict:
    """Acurácia + precisão/recall/F1 macro e por classe.

    Convenção para divisões por zero (classe nunca predita ou ausente): a
    métrica correspondente vale 0 (equivalente a zero_division=0 do scikit-learn).
    """
    cm = confusion_matrix(preds, targets, num_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    return {
        "accuracy": accuracy(preds, targets),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "num_samples": int(cm.sum()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(actual[i]),
            }
            for i, name in enumerate(names)
        },
        "confusion_matrix": cm.tolist(),
    }


def evaluate_split(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int, class_names=None) -> dict:
    loss, preds, targets = predict(model, loader, criterion, device)
    metrics = classification_metrics(preds, targets, num_classes, class_names)
    metrics["loss"] = float(loss)
    return metrics


# --------------------------------------------------------------------------- #
# Complexidade
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> "dict[str, int]":
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def parameter_dtypes(model: nn.Module) -> "list[str]":
    return sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})


def theoretical_parameter_bytes(model: nn.Module) -> int:
    """Memória TEÓRICA dos parâmetros: soma de numel x bytes por elemento."""
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def serialized_state_dict_bytes(model: nn.Module) -> int:
    """Tamanho MEDIDO do state_dict serializado com torch.save (em memória)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def file_size(path: "str | Path") -> "dict[str, float]":
    size = Path(path).stat().st_size
    return {"bytes": int(size), "mb": size / BYTES_PER_MB, "mib": size / BYTES_PER_MIB}


def model_complexity(model: nn.Module, checkpoint_path: "str | Path | None" = None) -> dict:
    params = count_parameters(model)
    theoretical = theoretical_parameter_bytes(model)
    serialized = serialized_state_dict_bytes(model)
    result = {
        **params,
        "precision": parameter_dtypes(model),
        "theoretical_parameter_bytes": theoretical,
        "theoretical_parameter_mb": theoretical / BYTES_PER_MB,
        "serialized_state_dict_bytes": serialized,
        "serialized_state_dict_mb": serialized / BYTES_PER_MB,
        "notes": (
            "theoretical_* = numel x bytes/elemento (teórico, sem overhead). "
            "serialized_state_dict_* = torch.save(state_dict) medido em memória. "
            "checkpoint_* = arquivo em disco medido, incluindo metadados do checkpoint. "
            "MB = 10^6 bytes; MiB = 2^20 bytes."
        ),
    }
    if checkpoint_path is not None:
        size = file_size(checkpoint_path)
        result.update(
            {
                "checkpoint_file": Path(checkpoint_path).name,
                "checkpoint_bytes": size["bytes"],
                "checkpoint_mb": size["mb"],
                "checkpoint_mib": size["mib"],
            }
        )
    return result
