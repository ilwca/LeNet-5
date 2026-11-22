"""Utilitários centrais: configuração, dispositivo, sementes, ambiente e E/S.

Toda decisão de dispositivo e de reprodutibilidade passa por este módulo, para
que o restante do código não dependa de detalhes do hardware.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SECTIONS = ("experiment", "model", "dataset", "training", "evaluation", "environment")

# Campos que identificam a repetição, e não o experimento. São ignorados no
# hash da configuração para que repetições do mesmo experimento compartilhem
# o mesmo identificador.
_RUN_SPECIFIC_KEYS = (("experiment", "seed"), ("run",))


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
def load_config(path: "str | Path") -> dict:
    """Lê e valida um arquivo YAML de configuração."""
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validações mínimas que protegem a integridade do baseline."""
    if not isinstance(config, dict):
        raise ValueError("Configuração inválida: esperado um mapeamento YAML.")
    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Seções ausentes na configuração: {missing}")

    for section in ("model", "environment"):
        precision = config[section].get("precision", "float32")
        if precision != "float32":
            raise ValueError(f"{section}.precision={precision!r}: o baseline exige float32.")

    if config["dataset"].get("augmentation", False):
        raise ValueError("Data augmentation não faz parte do baseline (dataset.augmentation deve ser false).")

    training = config["training"]
    if training.get("scheduler") == "reduce_on_plateau" and "scheduler_params" not in training:
        raise ValueError("training.scheduler_params deve ser definido explicitamente para reduce_on_plateau.")
    if training.get("early_stopping") and "early_stopping_patience" not in training:
        raise ValueError("training.early_stopping_patience é obrigatório quando early_stopping=true.")

    exp = config["experiment"]
    seeds = exp.get("seeds")
    if seeds is not None:
        if len(seeds) < int(exp["repetitions"]):
            raise ValueError("experiment.seeds possui menos sementes que experiment.repetitions.")
        if len(set(seeds[: int(exp["repetitions"])])) != int(exp["repetitions"]):
            raise ValueError("experiment.seeds não pode conter sementes repetidas.")


def save_config(config: dict, path: "str | Path") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)


def config_hash(config: dict) -> str:
    """Hash estável da configuração do experimento (independe da semente da repetição)."""
    cfg = copy.deepcopy(config)
    for keys in _RUN_SPECIFIC_KEYS:
        node = cfg
        for key in keys[:-1]:
            node = node.get(key, {})
        node.pop(keys[-1], None)
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def get_seeds(config: dict) -> "list[int]":
    """Sementes das repetições: `experiment.seeds` se definido, senão seed, seed+1, ..."""
    exp = config["experiment"]
    n = int(exp["repetitions"])
    if exp.get("seeds"):
        return [int(s) for s in exp["seeds"][:n]]
    return [int(exp["seed"]) + i for i in range(n)]


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Define as sementes de Python, NumPy, PyTorch (CPU) e CUDA.

    Com `deterministic=True`:
    - cuDNN opera em modo determinístico e sem autotuning (`benchmark=False`),
      o que pode reduzir a velocidade em GPU;
    - `torch.use_deterministic_algorithms(True, warn_only=True)` emite aviso
      quando uma operação não possui implementação determinística;
    - `CUBLAS_WORKSPACE_CONFIG` é definido para tornar o cuBLAS determinístico.

    Mesmo assim, resultados bit a bit idênticos só são esperados no MESMO
    hardware e nas MESMAS versões de bibliotecas. CPU e GPU (ou GPUs
    diferentes) produzem diferenças numéricas pequenas.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """Gerador dedicado para o embaralhamento dos DataLoaders."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # pragma: no cover - usado apenas com num_workers > 0
    """Semente de cada worker do DataLoader, derivada da semente do gerador."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def configure_fp32() -> None:
    """Garante aritmética FP32 'pura'.

    Em GPUs Ampere ou mais recentes (incluindo a RTX série 40), o PyTorch pode
    usar TF32 em convoluções (cuDNN) e multiplicações de matrizes, o que reduz
    a mantissa efetiva para 10 bits. Isso alteraria silenciosamente a precisão
    do baseline, por isso o TF32 é desabilitado explicitamente.
    """
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# --------------------------------------------------------------------------- #
# Dispositivo
# --------------------------------------------------------------------------- #
def cuda_is_usable() -> bool:
    """CUDA disponível E funcional.

    A presença física de uma GPU NVIDIA não basta: é preciso o driver
    proprietário e um build do PyTorch com CUDA. Por isso, além de
    `torch.cuda.is_available()`, executa-se uma operação mínima na GPU.
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(1, device="cuda")
        _ = (x + 1).item()
        torch.cuda.synchronize()
        return True
    except Exception:  # noqa: BLE001 - qualquer falha de inicialização implica CUDA inutilizável
        return False


def resolve_device(spec: str = "auto") -> torch.device:
    """Seleção centralizada do dispositivo.

    - "auto": CUDA se utilizável, senão CPU.
    - "cpu": força CPU.
    - "cuda"/"cuda:N": exige CUDA; falha explicitamente caso indisponível, em
      vez de cair silenciosamente para CPU (o que comprometeria a
      comparabilidade das medições de latência).
    """
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda") if cuda_is_usable() else torch.device("cpu")
    if spec.startswith("cuda"):
        if not cuda_is_usable():
            raise RuntimeError(
                f"Dispositivo {spec!r} solicitado, mas CUDA não está disponível/funcional. "
                "Verifique o driver NVIDIA e o build do PyTorch, ou use device: auto."
            )
        return torch.device(spec)
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Dispositivo desconhecido: {spec!r}")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconhecido"


def collect_environment_info(device: "torch.device | None" = None) -> dict:
    """Informações de hardware e software para o registro do experimento."""
    import matplotlib
    import torchvision

    device = device or resolve_device("auto")
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "device_type": device.type,
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "yaml_version": yaml.__version__,
        "torch_cuda_build_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_usable": cuda_is_usable(),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "default_dtype": str(torch.get_default_dtype()),
        "gpu_name": None,
        "gpu_count": 0,
    }
    if device.type == "cuda" and cuda_available:
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_memory_bytes": props.total_memory,
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


# --------------------------------------------------------------------------- #
# E/S e registro
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Tipo não serializável: {type(obj)}")


def save_json(data: Any, path: "str | Path") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: "str | Path") -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


LOGGER_NAME = "lenet"


def setup_logging(log_file: "str | Path | None" = None, level: int = logging.INFO) -> logging.Logger:
    """Logger do projeto, com saída no terminal e (opcionalmente) em arquivo."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def remove_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def resolve_path(path: "str | Path") -> Path:
    """Caminhos relativos da configuração são resolvidos a partir da raiz do projeto."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
