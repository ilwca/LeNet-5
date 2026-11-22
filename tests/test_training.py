import copy

import pytest
import torch

from models import LeNet5, build_model
from src.dataset import make_dataloaders
from src.train import (
    BEST_CHECKPOINT,
    HISTORY_FILE,
    LAST_CHECKPOINT,
    build_criterion,
    build_optimizer,
    build_scheduler,
    fit,
    load_best_checkpoint,
    read_history,
    train_one_epoch,
)
from src.utils import config_hash, get_seeds, resolve_device, set_seed, validate_config


def test_one_epoch_updates_parameters_and_computes_loss(loaders, tiny_config):
    set_seed(0)
    model = LeNet5()
    before = copy.deepcopy(model.state_dict())
    criterion = build_criterion(tiny_config["training"])
    optimizer = build_optimizer(model, tiny_config["training"])
    loss, acc = train_one_epoch(model, loaders["train"], criterion, optimizer, torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss)) and loss > 0
    assert 0.0 <= acc <= 1.0
    changed = [k for k in before if not torch.equal(before[k], model.state_dict()[k])]
    assert set(changed) == set(before)  # todos os pesos e bias foram atualizados


def test_initial_loss_close_to_log_num_classes(loaders):
    set_seed(0)
    model = LeNet5().eval()
    x, y = next(iter(loaders["val"]))
    with torch.no_grad():
        loss = torch.nn.CrossEntropyLoss()(model(x), y).item()
    assert abs(loss - torch.log(torch.tensor(10.0)).item()) < 0.3


def test_fit_writes_history_and_checkpoints(tmp_path, loaders, tiny_config):
    set_seed(1)
    result = fit(build_model(tiny_config), loaders, tiny_config, tmp_path, torch.device("cpu"))
    assert result["epochs_completed"] == 2
    for name in (BEST_CHECKPOINT, LAST_CHECKPOINT, HISTORY_FILE):
        assert (tmp_path / name).exists()
    history = read_history(tmp_path / HISTORY_FILE)
    assert [r["epoch"] for r in history] == [1, 2]
    assert result["best_epoch"] == min(history, key=lambda r: r["val_loss"])["epoch"]


def test_checkpoint_save_and_load(tmp_path, loaders, tiny_config):
    set_seed(2)
    model = build_model(tiny_config)
    fit(model, loaders, tiny_config, tmp_path, torch.device("cpu"))
    restored = LeNet5()
    ckpt = load_best_checkpoint(tmp_path / BEST_CHECKPOINT, restored)
    assert ckpt["precision"] == "float32" and ckpt["epoch"] in (1, 2)
    x = torch.randn(3, 3, 32, 32)
    if ckpt["epoch"] == 2:
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(x), restored.eval()(x))


def test_resume_matches_uninterrupted_training(tmp_path, splits, tiny_config):
    """Treinar 2 épocas direto == treinar 1 época, interromper e retomar."""
    cpu = torch.device("cpu")
    one_epoch = copy.deepcopy(tiny_config)
    one_epoch["training"]["epochs"] = 1

    set_seed(5)
    straight_model = build_model(tiny_config)
    fit(straight_model, make_dataloaders(splits, 64, 64, seed=5), tiny_config, tmp_path / "a", cpu)

    set_seed(5)
    fit(build_model(one_epoch), make_dataloaders(splits, 64, 64, seed=5), one_epoch, tmp_path / "b", cpu)
    # "Novo processo": RNG embaralhado e pesos diferentes; tudo deve vir do checkpoint.
    set_seed(999)
    resumed_model = build_model(tiny_config)
    result = fit(resumed_model, make_dataloaders(splits, 64, 64, seed=5), tiny_config, tmp_path / "b", cpu)
    assert [r["epoch"] for r in result["history"]] == [1, 2]

    straight, resumed = straight_model.state_dict(), resumed_model.state_dict()
    for key in straight:
        assert torch.allclose(straight[key], resumed[key], atol=1e-6), key


def test_scheduler_built_from_explicit_params(tiny_config):
    model = LeNet5()
    opt = build_optimizer(model, tiny_config["training"])
    sched = build_scheduler(opt, tiny_config["training"])
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert sched.factor == 0.5 and sched.patience == 5
    cfg = copy.deepcopy(tiny_config)
    del cfg["training"]["scheduler_params"]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_config_rejects_non_fp32_and_augmentation(tiny_config):
    for section, key, value in (("environment", "precision", "float16"), ("dataset", "augmentation", True)):
        cfg = copy.deepcopy(tiny_config)
        cfg[section][key] = value
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_seeds_and_config_hash(tiny_config):
    assert get_seeds(tiny_config) == [42, 123, 456, 789, 1024]
    other = copy.deepcopy(tiny_config)
    other["experiment"]["seed"] = 999
    assert config_hash(other) == config_hash(tiny_config)  # semente da repetição não altera o hash
    other["training"]["learning_rate"] = 0.01
    assert config_hash(other) != config_hash(tiny_config)


def test_device_selection():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
