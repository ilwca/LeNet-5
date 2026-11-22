"""Gráficos dos experimentos (PNG estático, backend Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # paleta categórica (ordem fixa)
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9cc3ef", "#2a78d6", "#0d3c75"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_PRIMARY,
        "lines.linewidth": 2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _curve(history, key_train, key_val, ylabel, title, path, best_epoch=None):
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(epochs, [r[key_train] for r in history], color=SERIES[0], label="Treino")
    ax.plot(epochs, [r[key_val] for r in history], color=SERIES[1], label="Validação")
    if best_epoch:
        ax.axvline(best_epoch, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.annotate(f"melhor época ({best_epoch})", xy=(best_epoch, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", color=TEXT_SECONDARY, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(history: "list[dict]", out_dir: "str | Path", best_epoch: "int | None" = None) -> "list[Path]":
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_path, acc_path = out_dir / "loss.png", out_dir / "accuracy.png"
    _curve(history, "train_loss", "val_loss", "Cross-entropy", "Loss por época", loss_path, best_epoch)
    _curve(history, "train_accuracy", "val_accuracy", "Acurácia", "Acurácia por época", acc_path, best_epoch)
    return [loss_path, acc_path]


def plot_confusion_matrix(cm, class_names, path: "str | Path", title: str = "Matriz de confusão (teste)") -> Path:
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.grid(False)
    im = ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    n = len(class_names)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if norm[i, j] > 0.55 else TEXT_PRIMARY)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fração da classe verdadeira (recall)", color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_repetitions(rows: "list[dict]", path: "str | Path") -> Path:
    """Acurácia e F1 macro de teste por repetição (pontos), com a média como referência.

    Usa pontos em vez de barras: o eixo y não parte de zero para tornar visível
    a variação entre sementes, e barras com eixo truncado distorceriam a escala.
    """
    labels = [f"seed {r['seed']}" for r in rows]
    metrics = [("test_accuracy", "Acurácia de teste por repetição"), ("test_f1_macro", "F1 macro de teste por repetição")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = np.arange(len(rows))
    for ax, (key, title) in zip(axes, metrics):
        values = np.array([r[key] for r in rows], dtype=float)
        mean = values.mean()
        ax.axhline(mean, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
        ax.scatter(x, values, s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.4f}", xy=(xi, v), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_PRIMARY)
        ax.annotate(f"média {mean:.4f}", xy=(1, mean), xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="right", color=TEXT_SECONDARY, fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        spread = max(values.max() - values.min(), 0.01)
        ax.set_ylim(values.min() - spread, values.max() + spread)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
