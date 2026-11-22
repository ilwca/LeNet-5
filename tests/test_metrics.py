import numpy as np
import pytest
import torch

from models import LeNet5
from src.benchmark import benchmark_latency, latency_statistics, run_benchmarks
from src.evaluate import (
    accuracy,
    classification_metrics,
    confusion_matrix,
    count_parameters,
    evaluate_split,
    model_complexity,
    theoretical_parameter_bytes,
)


def test_accuracy():
    assert accuracy([0, 1, 2, 3], [0, 1, 0, 3]) == 0.75
    assert accuracy(np.ones(10), np.ones(10)) == 1.0
    with pytest.raises(ValueError):
        accuracy([], [])


def test_confusion_matrix():
    cm = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], num_classes=3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_macro_metrics_hand_computed():
    # classe 0: P=1, R=1; classe 1: P=1/2, R=1; classe 2: P=1, R=1/2
    m = classification_metrics([0, 1, 1, 2], [0, 1, 2, 2], num_classes=3)
    assert m["precision_macro"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert m["recall_macro"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert m["f1_macro"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
    assert m["num_samples"] == 4


def test_macro_metrics_zero_division():
    m = classification_metrics([0, 0], [0, 1], num_classes=3)
    assert m["per_class"]["2"]["precision"] == 0.0 and m["per_class"]["1"]["recall"] == 0.0


def test_parameter_counts():
    model = LeNet5()
    assert count_parameters(model) == {"total_parameters": 62006, "trainable_parameters": 62006}
    model.features.c1.weight.requires_grad_(False)
    assert count_parameters(model)["trainable_parameters"] == 62006 - 450
    assert theoretical_parameter_bytes(model) == 62006 * 4


def test_checkpoint_size_is_recorded(tmp_path):
    model = LeNet5()
    path = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)
    c = model_complexity(model, path)
    assert c["checkpoint_bytes"] == path.stat().st_size > 0
    assert c["checkpoint_mb"] == pytest.approx(c["checkpoint_bytes"] / 1e6)
    assert c["serialized_state_dict_bytes"] > c["theoretical_parameter_bytes"]  # overhead de serialização
    assert c["precision"] == ["float32"]


def test_evaluate_split(loaders):
    m = evaluate_split(LeNet5(), loaders["test"], torch.nn.CrossEntropyLoss(), torch.device("cpu"), 10)
    assert m["num_samples"] == 200
    assert np.array(m["confusion_matrix"]).sum() == 200
    assert m["loss"] > 0


def test_latency_statistics():
    s = latency_statistics([1.0, 2.0, 3.0, 4.0])
    assert s["mean_ms"] == 2.5 and s["median_ms"] == 2.5
    assert s["std_ms"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert s["min_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["max_ms"]


def test_benchmark_runs_on_cpu():
    r = benchmark_latency(LeNet5(), torch.device("cpu"), batch_size=8, warmup_iterations=2, benchmark_iterations=5)
    assert len(r["raw_times_ms"]) == 5
    assert r["batch_latency"]["mean_ms"] > 0
    assert r["per_image_latency"]["mean_ms"] == pytest.approx(r["batch_latency"]["mean_ms"] / 8)


def test_run_benchmarks_multiple_batch_sizes():
    out = run_benchmarks(LeNet5(), torch.device("cpu"), [1, 4], warmup_iterations=1, benchmark_iterations=3)
    assert set(out["results"]) == {"batch_1", "batch_4"}
    assert out["dtype"] == "float32" and not out["cuda_synchronized"]
