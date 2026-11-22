"""Loop de treinamento FP32 com avaliação por época, checkpoints e retomada.

Este módulo NÃO contém lógica de quantização, poda, QAT ou mixed precision.

Checkpoints gravados no diretório da execução:
- `checkpoint_best.pth`: somente pesos (state_dict) + metadados mínimos, da
  época com MENOR perda de validação. É o artefato avaliado no teste e o
  usado para medir o tamanho do modelo.
- `checkpoint_last.pth`: estado completo (modelo, otimizador, scheduler,
  histórico, estados dos geradores aleatórios) da última época concluída,
  usado para retomar uma execução interrompida.

Observação sobre a acurácia de treino: é acumulada durante a própria época
(com os pesos mudando a cada passo), prática usual que a torna ligeiramente
diferente de uma reavaliação do conjunto de treino ao final da época.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluate import evaluate_loss_accuracy
from src.utils import LOGGER_NAME, capture_rng_state, restore_rng_state

BEST_CHECKPOINT = "checkpoint_best.pth"
LAST_CHECKPOINT = "checkpoint_last.pth"
HISTORY_FILE = "history.csv"
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "learning_rate",
    "epoch_time_s",
    "is_best",
)


# --------------------------------------------------------------------------- #
# Construção dos componentes
# --------------------------------------------------------------------------- #
def build_criterion(training_cfg: dict) -> nn.Module:
    if training_cfg.get("loss", "cross_entropy") != "cross_entropy":
        raise ValueError(f"Loss não suportada: {training_cfg.get('loss')!r}")
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    name = training_cfg.get("optimizer", "adam").lower()
    lr = float(training_cfg["learning_rate"])
    wd = float(training_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=tuple(training_cfg.get("adam_betas", (0.9, 0.999))),
            eps=float(training_cfg.get("adam_eps", 1e-8)),
        )
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=float(training_cfg.get("momentum", 0.0)))
    raise ValueError(f"Otimizador não suportado: {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: dict):
    name = training_cfg.get("scheduler")
    if name in (None, "none"):
        return None
    if name == "reduce_on_plateau":
        p = training_cfg["scheduler_params"]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=p.get("mode", "min"),
            factor=float(p["factor"]),
            patience=int(p["patience"]),
            threshold=float(p.get("threshold", 1e-4)),
            threshold_mode=p.get("threshold_mode", "rel"),
            cooldown=int(p.get("cooldown", 0)),
            min_lr=float(p.get("min_lr", 0.0)),
        )
    raise ValueError(f"Scheduler não suportado: {name!r}")


# --------------------------------------------------------------------------- #
# Época de treinamento
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> "tuple[float, float]":
    """Uma passada pelo conjunto de treino. Retorna (loss média, acurácia)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss não finita durante o treinamento: {loss.item()}")
        loss.backward()
        optimizer.step()
        batch = labels.size(0)
        total_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch
    return total_loss / total, correct / total


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_best_checkpoint(path: "str | Path", model: nn.Module, epoch: int, val_loss: float, val_accuracy: float, metadata: "dict | None" = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "precision": "float32",
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_best_checkpoint(path: "str | Path", model: nn.Module, device: "torch.device | str" = "cpu") -> dict:
    """Carrega os pesos do melhor checkpoint (somente tensores/primitivos: weights_only=True)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_last_checkpoint(path: "str | Path", state: dict) -> None:
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # escrita atômica: um checkpoint parcial nunca substitui o anterior


def load_last_checkpoint(path: "str | Path", device: "torch.device | str" = "cpu") -> dict:
    # weights_only=False: o arquivo contém estados de RNG (NumPy/Python) e é
    # produzido pelo próprio projeto; não carregue checkpoints de terceiros assim.
    return torch.load(path, map_location=device, weights_only=False)


def write_history(path: "str | Path", history: "list[dict]") -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in HISTORY_FIELDS})


def read_history(path: "str | Path") -> "list[dict]":
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        out.append(
            {
                "epoch": int(row["epoch"]),
                **{k: float(row[k]) for k in HISTORY_FIELDS if k not in ("epoch", "is_best")},
                "is_best": row["is_best"] == "True",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Treinamento completo
# --------------------------------------------------------------------------- #
def fit(
    model: nn.Module,
    loaders: "dict[str, DataLoader]",
    config: dict,
    run_dir: "str | Path",
    device: torch.device,
    resume: bool = True,
    checkpoint_metadata: "dict | None" = None,
    logger: "logging.Logger | None" = None,
) -> dict:
    """Treina o modelo por `training.epochs` épocas usando treino e validação.

    O conjunto de teste NÃO é recebido nem utilizado aqui. O scheduler e a
    seleção do melhor checkpoint usam exclusivamente a perda de validação.
    """
    logger = logger or logging.getLogger(LOGGER_NAME)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = config["training"]
    epochs = int(training_cfg["epochs"])
    if training_cfg.get("checkpoint_metric", "val_loss") != "val_loss":
        raise ValueError("O melhor checkpoint é selecionado pela perda de validação (checkpoint_metric: val_loss).")

    model = model.to(device=device, dtype=torch.float32)
    criterion = build_criterion(training_cfg)
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg)
    early_stopping = bool(training_cfg.get("early_stopping", False))
    es_patience = int(training_cfg.get("early_stopping_patience", 0))

    best_path, last_path, history_path = run_dir / BEST_CHECKPOINT, run_dir / LAST_CHECKPOINT, run_dir / HISTORY_FILE
    history: "list[dict]" = []
    start_epoch = 1
    best_val_loss, best_epoch, epochs_without_improvement = math.inf, 0, 0
    stopped_early = False

    if resume and last_path.exists():
        state = load_last_checkpoint(last_path, device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        history = state["history"]
        best_val_loss, best_epoch = state["best_val_loss"], state["best_epoch"]
        epochs_without_improvement = state.get("epochs_without_improvement", 0)
        stopped_early = state.get("stopped_early", False)
        restore_rng_state(state["rng_state"])
        loaders["train"].generator.set_state(state["train_loader_generator_state"])
        start_epoch = state["epoch"] + 1
        logger.info("Retomando a partir da época %d (melhor val_loss=%.4f na época %d).", state["epoch"], best_val_loss, best_epoch)

    for epoch in range(start_epoch, epochs + 1):
        if stopped_early:
            break
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate_loss_accuracy(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            save_best_checkpoint(best_path, model, epoch, val_loss, val_acc, checkpoint_metadata)
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": lr,
                "epoch_time_s": elapsed,
                "is_best": is_best,
            }
        )
        if early_stopping and epochs_without_improvement >= es_patience:
            stopped_early = True

        save_last_checkpoint(
            last_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "stopped_early": stopped_early,
                "rng_state": capture_rng_state(),
                "train_loader_generator_state": loaders["train"].generator.get_state(),
            },
        )
        write_history(history_path, history)
        logger.info(
            "Época %3d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.2e | %.1fs%s",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed, " *" if is_best else "",
        )
        if stopped_early:
            logger.info("Early stopping na época %d.", epoch)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "stopped_early": stopped_early,
        "training_time_s": float(sum(r["epoch_time_s"] for r in history)),
    }
