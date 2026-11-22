"""Interface de linha de comando e orquestração do baseline LeNet-5 FP32 / CIFAR-10.

Exemplos:
    python main.py info
    python main.py train     --config configs/baseline.yaml [--seed 42]
    python main.py evaluate  --run-id RUN_ID
    python main.py benchmark --run-id RUN_ID
    python main.py run-all   --config configs/baseline.yaml
    python main.py summarize --config configs/baseline.yaml

Execuções já completas nunca são refeitas nem sobrescritas, exceto quando
solicitado explicitamente (`--new` cria NOVAS execuções; `--overwrite`
regrava métricas/benchmark de uma execução existente).
"""

from __future__ import annotations

import argparse
import copy
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from models import build_model
from src.benchmark import run_benchmarks
from src.dataset import dataloaders_from_config, load_cifar10_splits
from src.evaluate import evaluate_split, model_complexity
from src.plots import plot_confusion_matrix, plot_repetitions, plot_training_curves
from src.train import BEST_CHECKPOINT, HISTORY_FILE, build_criterion, fit, load_best_checkpoint, read_history
from src.utils import (
    PROJECT_ROOT,
    collect_environment_info,
    config_hash,
    configure_fp32,
    get_seeds,
    load_config,
    load_json,
    remove_file_handlers,
    resolve_device,
    resolve_path,
    save_config,
    save_json,
    set_seed,
    setup_logging,
    timestamp,
)

RUN_INFO = "run_info.json"
METRICS_FILE = "metrics.json"
BENCHMARK_FILE = "benchmark.json"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

logger = setup_logging()


# --------------------------------------------------------------------------- #
# Execuções (runs)
# --------------------------------------------------------------------------- #
def output_dir(config: dict) -> Path:
    return resolve_path(config["experiment"].get("output_dir", "experiments/baseline"))


def is_trained(run_dir: Path) -> bool:
    info_path = run_dir / RUN_INFO
    return info_path.exists() and load_json(info_path).get("status") == "trained"


def is_complete(run_dir: Path) -> bool:
    return is_trained(run_dir) and (run_dir / METRICS_FILE).exists() and (run_dir / BENCHMARK_FILE).exists()


def list_runs(out_dir: Path) -> "list[tuple[Path, dict]]":
    runs = []
    if out_dir.exists():
        for d in sorted(out_dir.glob("run_*")):
            if (d / RUN_INFO).exists():
                runs.append((d, load_json(d / RUN_INFO)))
    return runs


def find_run(out_dir: Path, chash: str, seed: int) -> "Path | None":
    """Execução mais recente com a mesma configuração e semente (prioriza completas)."""
    matches = [d for d, info in list_runs(out_dir) if info.get("config_hash") == chash and info.get("seed") == seed]
    complete = [d for d in matches if is_complete(d)]
    pool = complete or matches
    return pool[-1] if pool else None


def resolve_run_dir(run_id: str) -> Path:
    candidate = Path(run_id)
    if candidate.is_dir() and (candidate / RUN_INFO).exists():
        return candidate.resolve()
    hits = [p for p in (PROJECT_ROOT / "experiments").glob(f"*/{run_id}") if (p / RUN_INFO).exists()]
    if not hits:
        raise SystemExit(f"Execução não encontrada: {run_id!r} (procurado em experiments/*/).")
    if len(hits) > 1:
        raise SystemExit(f"RUN_ID ambíguo: {[str(h) for h in hits]}. Informe o caminho completo.")
    return hits[0]


def run_config(base_config: dict, seed: int, repetition: int, run_id: str) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["experiment"]["seed"] = seed
    cfg["run"] = {"run_id": run_id, "repetition": repetition, "seed": seed, "config_hash": config_hash(base_config)}
    return cfg


def create_run(base_config: dict, seed: int, repetition: int) -> Path:
    out_dir = output_dir(base_config)
    run_id = f"run_{timestamp()}_seed{seed}"
    run_dir = out_dir / run_id
    while run_dir.exists():  # nunca reutiliza um diretório existente
        run_id += "b"
        run_dir = out_dir / run_id
    (run_dir / "plots").mkdir(parents=True)
    cfg = run_config(base_config, seed, repetition, run_id)
    save_config(cfg, run_dir / "config.yaml")
    save_json(
        {
            "run_id": run_id,
            "experiment_name": base_config["experiment"]["name"],
            "seed": seed,
            "repetition": repetition,
            "config_hash": config_hash(base_config),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "created",
        },
        run_dir / RUN_INFO,
    )
    logger.info("Nova execução criada: %s", run_dir.relative_to(PROJECT_ROOT))
    return run_dir


def update_run_info(run_dir: Path, **fields) -> dict:
    info = load_json(run_dir / RUN_INFO)
    info.update(fields)
    save_json(info, run_dir / RUN_INFO)
    return info


def prepare_runtime(config: dict, device_override: "str | None" = None):
    configure_fp32()
    device = resolve_device(device_override or config["environment"].get("device", "auto"))
    return device


def attach_run_log(run_dir: Path) -> None:
    remove_file_handlers(logger)
    setup_logging(LOG_DIR / f"{run_dir.name}.log")


def check_split(run_dir: Path, splits) -> None:
    """Garante que avaliação usa exatamente a divisão/normalização do treinamento."""
    info = load_json(run_dir / RUN_INFO)
    expected = info.get("split_fingerprint")
    if expected and expected != splits.split_fingerprint():
        raise RuntimeError(f"A divisão dos dados difere da usada no treinamento de {run_dir.name}.")
    norm = info.get("normalization")
    if norm and not (np.allclose(norm["mean"], splits.mean) and np.allclose(norm["std"], splits.std)):
        raise RuntimeError(f"Os parâmetros de normalização diferem dos usados em {run_dir.name}.")


# --------------------------------------------------------------------------- #
# Etapas
# --------------------------------------------------------------------------- #
def train_run(run_dir: Path, device_override: "str | None" = None) -> dict:
    cfg = load_config(run_dir / "config.yaml")
    attach_run_log(run_dir)
    device = prepare_runtime(cfg, device_override)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic=bool(cfg["training"].get("deterministic", True)))

    splits = load_cifar10_splits(cfg)
    check_split(run_dir, splits)
    loaders = dataloaders_from_config(splits, cfg, seed, device)
    model = build_model(cfg)
    model.verify_shapes()

    env = collect_environment_info(device)
    env_path = run_dir / "environment.json"
    if env_path.exists():  # retomada: preserva o registro original e anexa o novo
        previous = load_json(env_path)
        sessions = previous if isinstance(previous, list) else [previous]
        sessions.append(env)
        save_json(sessions, env_path)
    else:
        save_json(env, env_path)
    update_run_info(
        run_dir,
        status="training",
        split_fingerprint=splits.split_fingerprint(),
        normalization={"mean": list(splits.mean), "std": list(splits.std)},
        device=str(device),
    )
    logger.info("Treinando %s | seed=%d | dispositivo=%s | %s", run_dir.name, seed, device, env.get("gpu_name") or env["cpu_model"])
    logger.info("Normalização (treino): mean=%s std=%s", np.round(splits.mean, 4).tolist(), np.round(splits.std, 4).tolist())

    result = fit(
        model,
        loaders,
        cfg,
        run_dir,
        device,
        resume=True,
        checkpoint_metadata={
            "run_id": run_dir.name,
            "seed": seed,
            "config_hash": cfg["run"]["config_hash"],
            "architecture": cfg["model"]["architecture"],
            "normalization_mean": list(splits.mean),
            "normalization_std": list(splits.std),
        },
        logger=logger,
    )
    update_run_info(
        run_dir,
        status="trained",
        trained_at=datetime.now().isoformat(timespec="seconds"),
        best_epoch=result["best_epoch"],
        best_val_loss=result["best_val_loss"],
        epochs_completed=result["epochs_completed"],
        stopped_early=result["stopped_early"],
        training_time_s=result["training_time_s"],
    )
    plot_training_curves(result["history"], run_dir / "plots", result["best_epoch"])
    logger.info("Treino concluído: melhor época %d (val_loss=%.4f).", result["best_epoch"], result["best_val_loss"])
    return result


def evaluate_run(run_dir: Path, overwrite: bool = False, device_override: "str | None" = None) -> dict:
    if not is_trained(run_dir):
        raise SystemExit(f"{run_dir.name} ainda não terminou o treinamento.")
    metrics_path = run_dir / METRICS_FILE
    if metrics_path.exists() and not overwrite:
        logger.info("%s já possui %s (use --overwrite para regravar).", run_dir.name, METRICS_FILE)
        return load_json(metrics_path)
    cfg = load_config(run_dir / "config.yaml")
    attach_run_log(run_dir)
    device = prepare_runtime(cfg, device_override)
    splits = load_cifar10_splits(cfg)
    check_split(run_dir, splits)
    loaders = dataloaders_from_config(splits, cfg, int(cfg["experiment"]["seed"]), device)

    model = build_model(cfg).to(device)
    checkpoint = load_best_checkpoint(run_dir / BEST_CHECKPOINT, model, device)
    criterion = build_criterion(cfg["training"])
    n_classes = int(cfg["model"]["num_classes"])
    val = evaluate_split(model, loaders["val"], criterion, device, n_classes, splits.class_names)
    test = evaluate_split(model, loaders["test"], criterion, device, n_classes, splits.class_names)
    metrics = {
        "run_id": run_dir.name,
        "seed": int(cfg["experiment"]["seed"]),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "checkpoint": {"file": BEST_CHECKPOINT, "epoch": checkpoint["epoch"], "selection": "menor val_loss"},
        "validation": val,
        "test": test,
        "complexity": model_complexity(model, run_dir / BEST_CHECKPOINT),
    }
    save_json(metrics, metrics_path)
    plot_confusion_matrix(test["confusion_matrix"], splits.class_names, run_dir / "plots" / "confusion_matrix_test.png")
    plot_confusion_matrix(val["confusion_matrix"], splits.class_names, run_dir / "plots" / "confusion_matrix_val.png",
                          title="Matriz de confusão (validação)")
    if (run_dir / HISTORY_FILE).exists():
        plot_training_curves(read_history(run_dir / HISTORY_FILE), run_dir / "plots", checkpoint["epoch"])
    logger.info(
        "%s | val acc %.4f loss %.4f | TESTE acc %.4f loss %.4f F1 %.4f",
        run_dir.name, val["accuracy"], val["loss"], test["accuracy"], test["loss"], test["f1_macro"],
    )
    return metrics


def benchmark_run(run_dir: Path, overwrite: bool = False, device_override: "str | None" = None) -> dict:
    if not is_trained(run_dir):
        raise SystemExit(f"{run_dir.name} ainda não terminou o treinamento.")
    bench_path = run_dir / BENCHMARK_FILE
    if bench_path.exists() and not overwrite:
        logger.info("%s já possui %s (use --overwrite para regravar).", run_dir.name, BENCHMARK_FILE)
        return load_json(bench_path)
    cfg = load_config(run_dir / "config.yaml")
    attach_run_log(run_dir)
    device = prepare_runtime(cfg, device_override)
    model = build_model(cfg)
    load_best_checkpoint(run_dir / BEST_CHECKPOINT, model, "cpu")  # carregamento fora da medição
    ev = cfg["evaluation"]
    result = run_benchmarks(model, device, ev["latency_batch_sizes"], ev["warmup_iterations"], ev["benchmark_iterations"])
    result.update({"run_id": run_dir.name, "measured_at": datetime.now().isoformat(timespec="seconds"),
                   "environment": collect_environment_info(device)})
    save_json(result, bench_path)
    for key, r in result["results"].items():
        logger.info(
            "Latência %s (%s): média %.4f ms | mediana %.4f | p95 %.4f | p99 %.4f | por imagem %.5f ms",
            key, device, r["batch_latency"]["mean_ms"], r["batch_latency"]["median_ms"],
            r["batch_latency"]["p95_ms"], r["batch_latency"]["p99_ms"], r["per_image_latency"]["mean_ms"],
        )
    return result


# --------------------------------------------------------------------------- #
# Consolidação
# --------------------------------------------------------------------------- #
def run_row(run_dir: Path) -> dict:
    info, metrics, bench = (load_json(run_dir / f) for f in (RUN_INFO, METRICS_FILE, BENCHMARK_FILE))
    row = {
        "run_id": info["run_id"],
        "seed": info["seed"],
        "best_epoch": info.get("best_epoch"),
        "training_time_s": info.get("training_time_s"),
        "val_accuracy": metrics["validation"]["accuracy"],
        "val_loss": metrics["validation"]["loss"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_loss": metrics["test"]["loss"],
        "test_precision_macro": metrics["test"]["precision_macro"],
        "test_recall_macro": metrics["test"]["recall_macro"],
        "test_f1_macro": metrics["test"]["f1_macro"],
        "total_parameters": metrics["complexity"]["total_parameters"],
        "checkpoint_bytes": metrics["complexity"]["checkpoint_bytes"],
        "benchmark_device": bench["device"],
    }
    for key, r in bench["results"].items():
        bs = r["batch_size"]
        row[f"latency_bs{bs}_mean_ms"] = r["batch_latency"]["mean_ms"]
        row[f"latency_bs{bs}_median_ms"] = r["batch_latency"]["median_ms"]
        row[f"latency_bs{bs}_p95_ms"] = r["batch_latency"]["p95_ms"]
        if bs > 1:
            row[f"latency_bs{bs}_per_image_ms"] = r["per_image_latency"]["mean_ms"]
    return row


def write_csv(path: Path, rows: "list[dict]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(config: dict) -> "Path | None":
    """Agrega média e desvio padrão SOMENTE quando todas as repetições estão completas."""
    out_dir = output_dir(config)
    chash = config_hash(config)
    seeds = get_seeds(config)
    selected, missing = [], []
    for seed in seeds:
        run_dir = find_run(out_dir, chash, seed)
        (selected if run_dir is not None and is_complete(run_dir) else missing).append(run_dir or seed)
    if missing:
        logger.info(
            "Resumo NÃO gerado: %d de %d repetições completas (sementes pendentes: %s).",
            len(selected), len(seeds), [m if isinstance(m, int) else m.name for m in missing],
        )
        return None

    rows = [run_row(d) for d in selected]
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "seed"]
    summary = []
    for key in numeric:
        values = np.array([r[key] for r in rows], dtype=float)
        summary.append(
            {
                "metric": key,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n": len(values),
            }
        )
    name = config["experiment"]["name"]
    write_csv(out_dir / "runs.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    save_json({"experiment": name, "config_hash": chash, "std": "amostral (ddof=1)", "runs": [r["run_id"] for r in rows],
               "summary": summary}, out_dir / "summary.json")
    metrics_dir, plots_dir = PROJECT_ROOT / "results" / "metrics", PROJECT_ROOT / "results" / "plots"
    shutil.copy2(out_dir / "summary.csv", metrics_dir / f"{name}_summary.csv")
    shutil.copy2(out_dir / "runs.csv", metrics_dir / f"{name}_runs.csv")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_repetitions(rows, plots_dir / f"{name}_repetitions.png")
    shutil.copy2(plots_dir / f"{name}_repetitions.png", out_dir / "repetitions.png")
    acc = next(s for s in summary if s["metric"] == "test_accuracy")
    logger.info("Resumo (%d repetições): acurácia de teste = %.4f ± %.4f", len(rows), acc["mean"], acc["std"])
    logger.info("Gravado em %s", (out_dir / "summary.csv").relative_to(PROJECT_ROOT))
    return out_dir / "summary.csv"


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #
def complete_pipeline(run_dir: Path, device: "str | None") -> None:
    if not is_trained(run_dir):
        train_run(run_dir, device)
    evaluate_run(run_dir, device_override=device)
    benchmark_run(run_dir, device_override=device)


def obtain_run(config: dict, seed: int, repetition: int, new: bool) -> Path:
    existing = None if new else find_run(output_dir(config), config_hash(config), seed)
    if existing is not None:
        state = "completa" if is_complete(existing) else ("treinada" if is_trained(existing) else "incompleta (será retomada)")
        logger.info("Reutilizando execução %s [%s].", existing.name, state)
        return existing
    return create_run(config, seed, repetition)


def cmd_train(args) -> None:
    if args.run_id:
        run_dir = resolve_run_dir(args.run_id)
        if is_trained(run_dir):
            logger.info("%s já está treinada.", run_dir.name)
            return
        train_run(run_dir, args.device)
        return
    config = load_config(args.config)
    seeds = get_seeds(config)
    seed = args.seed if args.seed is not None else seeds[0]
    repetition = seeds.index(seed) + 1 if seed in seeds else 0
    run_dir = obtain_run(config, seed, repetition, args.new)
    if is_trained(run_dir):
        logger.info("%s já está treinada (use --new para treinar uma nova execução).", run_dir.name)
        return
    train_run(run_dir, args.device)
    print(f"RUN_ID: {run_dir.name}")


def cmd_evaluate(args) -> None:
    evaluate_run(resolve_run_dir(args.run_id), args.overwrite, args.device)


def cmd_benchmark(args) -> None:
    benchmark_run(resolve_run_dir(args.run_id), args.overwrite, args.device)


def cmd_run_all(args) -> None:
    config = load_config(args.config)
    seeds = get_seeds(config)
    logger.info("Experimento %s | hash %s | %d repetições: %s", config["experiment"]["name"], config_hash(config), len(seeds), seeds)
    for repetition, seed in enumerate(seeds, start=1):
        run_dir = obtain_run(config, seed, repetition, args.new)
        if is_complete(run_dir):
            continue
        logger.info("=== Repetição %d/%d (seed %d) ===", repetition, len(seeds), seed)
        complete_pipeline(run_dir, args.device)
    remove_file_handlers(logger)
    summarize(config)


def cmd_summarize(args) -> None:
    summarize(load_config(args.config))


def cmd_info(args) -> None:
    configure_fp32()
    config = load_config(args.config)
    device = resolve_device(args.device or config["environment"].get("device", "auto"))
    env = collect_environment_info(device)
    width = max(len(k) for k in env)
    for k, v in env.items():
        print(f"{k:<{width}}  {v}")
    model = build_model(config)
    shapes = model.verify_shapes()
    print("\nArquitetura LeNet-5 (dimensões verificadas):")
    for name, shape in shapes.items():
        print(f"  {name:<8} {shape}")
    from src.evaluate import count_parameters

    print(f"\nParâmetros: {count_parameters(model)}")
    print(f"Hash da configuração: {config_hash(config)} | sementes: {get_seeds(config)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline LeNet-5 FP32 / CIFAR-10")
    sub = parser.add_subparsers(dest="command", required=True)
    default_cfg = "configs/baseline.yaml"

    def add_device(p):
        p.add_argument("--device", default=None, help="Sobrescreve environment.device (auto|cpu|cuda|cuda:N).")

    p = sub.add_parser("info", help="Ambiente computacional, dispositivo e verificação da arquitetura.")
    p.add_argument("--config", default=default_cfg)
    add_device(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("train", help="Treina uma execução (ou retoma uma existente).")
    p.add_argument("--config", default=default_cfg)
    p.add_argument("--seed", type=int, default=None, help="Semente da execução (padrão: primeira de experiment.seeds).")
    p.add_argument("--run-id", default=None, help="Retoma o treinamento de uma execução existente.")
    p.add_argument("--new", action="store_true", help="Cria uma nova execução mesmo se já houver uma equivalente.")
    add_device(p)
    p.set_defaults(func=cmd_train)

    for name, func, help_ in (
        ("evaluate", cmd_evaluate, "Avalia o melhor checkpoint em validação e teste."),
        ("benchmark", cmd_benchmark, "Mede a latência de inferência do melhor checkpoint."),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--run-id", required=True)
        p.add_argument("--overwrite", action="store_true", help="Regrava resultados já existentes.")
        add_device(p)
        p.set_defaults(func=func)

    p = sub.add_parser("run-all", help="Todas as repetições: treino, avaliação, benchmark e resumo.")
    p.add_argument("--config", default=default_cfg)
    p.add_argument("--new", action="store_true", help="Ignora execuções existentes e cria novas (não apaga as antigas).")
    add_device(p)
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("summarize", help="Consolida média e desvio padrão das repetições completas.")
    p.add_argument("--config", default=default_cfg)
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
