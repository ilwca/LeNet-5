import pytest
import torch
from torch import nn

from models import EXPECTED_PARAMETERS, LeNet5, build_model
from models.lenet import EXPECTED_SHAPES


@pytest.mark.parametrize("batch", [1, 7, 128])
def test_input_output_shapes(batch):
    model = LeNet5()
    out = model(torch.randn(batch, 3, 32, 32))
    assert out.shape == (batch, 10)
    assert out.dtype == torch.float32


def test_intermediate_shapes_match_specification():
    model = LeNet5()
    assert model.verify_shapes() == EXPECTED_SHAPES


def test_parameter_count():
    model = LeNet5()
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS == 62006
    per_layer = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_layers() if any(True for _ in m.parameters())}
    assert per_layer == {"c1": 456, "c2": 2416, "c3": 48120, "fc1": 10164, "fc2": 850}


def test_architecture_rules():
    model = LeNet5()
    modules = list(model.modules())
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout, nn.Softmax, nn.LogSoftmax)) for m in modules)
    convs = [m for m in modules if isinstance(m, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.padding) for c in convs] == [
        (3, 6, (5, 5), (0, 0)),
        (6, 16, (5, 5), (0, 0)),
        (16, 120, (5, 5), (0, 0)),
    ]
    assert sum(isinstance(m, nn.ReLU) for m in modules) == 4
    pools = [m for m in modules if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 2 and all(p.kernel_size == 2 and p.stride == 2 for p in pools)


def test_outputs_are_logits_not_probabilities():
    torch.manual_seed(0)
    out = LeNet5()(torch.randn(16, 3, 32, 32) * 5)
    assert not torch.allclose(out.sum(dim=1), torch.ones(16))


def test_rejects_non_rgb_input_channels():
    with pytest.raises(ValueError):
        LeNet5(input_channels=1)


def test_build_model_from_config(tiny_config):
    model = build_model(tiny_config)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    bad = {**tiny_config, "model": {**tiny_config["model"], "precision": "float16"}}
    with pytest.raises(ValueError):
        build_model(bad)


def test_cpu_forward():
    model = LeNet5().to("cpu")
    assert model(torch.randn(2, 3, 32, 32)).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indisponível")
def test_cuda_forward():
    model = LeNet5().to("cuda")
    out = model(torch.randn(4, 3, 32, 32, device="cuda"))
    assert out.device.type == "cuda" and out.shape == (4, 10)
    assert model.verify_shapes() == EXPECTED_SHAPES
