"""MultiStepLR: learning rate drops at the listed epochs."""
from types import SimpleNamespace

import pytest
import torch

from googlehydrology.training.basetrainer import BaseTrainer


def _rates(strategy, drop, factor, epochs):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    cfg = SimpleNamespace(
        learning_rate_strategy=strategy,
        learning_rate_epochs_drop=drop,
        learning_rate_drop_factor=factor,
    )
    fake = SimpleNamespace(cfg=cfg, optimizer=opt)
    sched, step = BaseTrainer._create_lr_scheduler(fake)
    rates = []
    for _ in range(epochs):  # same order as training: read the rate, train an epoch, then step
        rates.append(sched.get_last_lr()[-1])
        opt.step()
        step(0.0)
    return rates


def test_multisteplr_drops_at_listed_epochs():
    r = _rates('MultiStepLR', [20, 30], 0.5, 50)
    assert r[0] == pytest.approx(0.0005)     # epoch 1
    assert r[19] == pytest.approx(0.0005)    # epoch 20
    assert r[20] == pytest.approx(0.00025)   # epoch 21
    assert r[29] == pytest.approx(0.00025)   # epoch 30
    assert r[30] == pytest.approx(0.000125)  # epoch 31
    assert r[49] == pytest.approx(0.000125)  # epoch 50


def test_multisteplr_accepts_single_int():
    r = _rates('MultiStepLR', 10, 0.5, 21)
    assert r[9] == pytest.approx(0.0005)
    assert r[10] == pytest.approx(0.00025)
    assert r[20] == pytest.approx(0.00025)   # only one drop


def test_steplr_unchanged():
    r = _rates('StepLR', 10, 0.5, 21)
    assert r[10] == pytest.approx(0.00025)
    assert r[20] == pytest.approx(0.000125)
