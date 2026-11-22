"""Benchmark de latência de inferência.

Procedimento (por batch size):
1. Modelo em `eval()`, gradientes desabilitados (`torch.no_grad()`).
2. Tensor de entrada FP32 alocado NO DISPOSITIVO antes da medição: a
   transferência inicial de dados e o pré-processamento ficam fora da região
   cronometrada. A entrada é sintética (N(0,1), semente fixa); para uma rede
   densa em FP32 o custo computacional não depende dos valores dos pixels.
3. Aquecimento (`warmup_iterations`) não cronometrado.
4. Cada iteração de benchmark é cronometrada individualmente com
   `time.perf_counter()`. Em CUDA, o dispositivo é sincronizado antes e depois
   da região cronometrada, pois os kernels são assíncronos.
5. Estatísticas: média, mediana, desvio padrão (amostral), p95, p99, mín., máx.

Para batch > 1, registra-se a latência TOTAL do lote e a latência POR IMAGEM
(= latência do lote / batch size), que não equivale à latência com batch 1.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_statistics(times_ms) -> "dict[str, float]":
    t = np.asarray(times_ms, dtype=np.float64)
    if t.size == 0:
        raise ValueError("Nenhuma medição de tempo.")
    return {
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "std_ms": float(t.std(ddof=1)) if t.size > 1 else 0.0,
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
    }


def benchmark_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    input_shape: "tuple[int, ...]" = (3, 32, 32),
    warmup_iterations: int = 50,
    benchmark_iterations: int = 200,
    seed: int = 0,
) -> dict:
    """Mede a latência de inferência de um único batch size."""
    if benchmark_iterations < 1:
        raise ValueError("benchmark_iterations deve ser >= 1.")
    model = model.to(device)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn((batch_size, *input_shape), generator=generator, dtype=torch.float32).to(device)
    _synchronize(device)

    times_ms = []
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(inputs)
        _synchronize(device)
        for _ in range(benchmark_iterations):
            _synchronize(device)
            start = time.perf_counter()
            model(inputs)
            _synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    batch_stats = latency_statistics(times_ms)
    per_image = {k: v / batch_size for k, v in batch_stats.items()}
    mean_s = batch_stats["mean_ms"] / 1000.0
    return {
        "batch_size": batch_size,
        "device": str(device),
        "warmup_iterations": warmup_iterations,
        "benchmark_iterations": benchmark_iterations,
        "batch_latency": batch_stats,
        "per_image_latency": per_image,
        "throughput_images_per_s": batch_size / mean_s if mean_s > 0 else float("inf"),
        "raw_times_ms": [float(t) for t in times_ms],
    }


def run_benchmarks(
    model: nn.Module,
    device: torch.device,
    batch_sizes,
    warmup_iterations: int,
    benchmark_iterations: int,
    input_shape: "tuple[int, ...]" = (3, 32, 32),
) -> dict:
    results = {
        "device": str(device),
        "dtype": "float32",
        "torch_num_threads": torch.get_num_threads(),
        "timer": "time.perf_counter",
        "input": "sintético N(0,1), alocado no dispositivo antes da medição",
        "cuda_synchronized": device.type == "cuda",
        "results": {},
    }
    for bs in batch_sizes:
        results["results"][f"batch_{bs}"] = benchmark_latency(
            model, device, int(bs), input_shape, int(warmup_iterations), int(benchmark_iterations)
        )
    return results
