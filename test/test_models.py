# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for googlehydrology.modelzoo architectures and layers."""

from unittest.mock import MagicMock

import pytest
import torch

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.modelzoo.positional_encoding import PositionalEncoding


@pytest.mark.unit
def test_positional_encoding_concatenate():
    seq_len = 10
    batch_size = 4
    embedding_dim = 16

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        position_type='concatenate',
        dropout=0.0,
        max_len=100,
    )
    x = torch.randn(seq_len, batch_size, embedding_dim)
    out = pe(x)
    # Concatenate mode doubles embedding dimension: [seq, batch, 2 * dim]
    assert out.shape == (seq_len, batch_size, embedding_dim * 2)


@pytest.mark.unit
def test_positional_encoding_sum():
    seq_len = 10
    batch_size = 4
    embedding_dim = 16

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        position_type='sum',
        dropout=0.1,
        max_len=100,
    )
    x = torch.randn(seq_len, batch_size, embedding_dim)
    out = pe(x)
    # Sum mode keeps embedding dimension: [seq, batch, dim]
    assert out.shape == (seq_len, batch_size, embedding_dim)


@pytest.mark.unit
def test_positional_encoding_invalid_type():
    with pytest.raises(
        RuntimeError, match='Unrecognized positional encoding type'
    ):
        PositionalEncoding(
            embedding_dim=16,
            position_type='invalid_type',
            dropout=0.0,
        )


@pytest.mark.unit
def test_fc_empty_hidden_sizes():
    with pytest.raises(
        ValueError, match='hidden_sizes must at least have one entry'
    ):
        FC(input_size=10, hidden_sizes=[])


@pytest.mark.unit
def test_fc_activations_and_shapes():
    # Single layer linear (hidden_sizes=[output_size])
    fc_linear = FC(input_size=10, hidden_sizes=[5])
    x = torch.randn(4, 10)
    out = fc_linear(x)
    assert out.shape == (4, 5)

    # Multi-layer with activations and dropout
    activations = ['relu', 'tanh', 'sigmoid', 'linear']
    for act in activations:
        fc = FC(
            input_size=10,
            hidden_sizes=[20, 15, 3],
            activation=act,
            dropout=0.1,
            xavier_init=True,
        )
        out = fc(x)
        assert out.shape == (4, 3)

    # Multi-layer with list of activations
    fc_multi_act = FC(
        input_size=10,
        hidden_sizes=[20, 15, 3],
        activation=['relu', 'tanh'],
    )
    out = fc_multi_act(x)
    assert out.shape == (4, 3)

    # Unsupported activation
    with pytest.raises(
        NotImplementedError, match='currently not supported as activation'
    ):
        FC(input_size=10, hidden_sizes=[20, 3], activation='invalid_act')


@pytest.mark.unit
def test_head_regression():
    cfg = MagicMock()
    cfg.head = 'regression'
    cfg.output_activation = 'linear'
    head = get_head(cfg=cfg, n_in=32, n_out=1)

    x = torch.randn(4, 10, 32)  # [batch, seq, in_features]
    out = head(x)
    assert 'y_hat' in out
    assert out['y_hat'].shape == (4, 10, 1)


@pytest.mark.unit
def test_head_cmal():
    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.cmal_min_scale = 1e-5
    # n_out for CMAL should match n_targets * 4 * n_distributions
    head = get_head(cfg=cfg, n_in=32, n_out=12, n_hidden=50)

    x = torch.randn(4, 10, 32)
    out = head(x)
    assert 'mu' in out
    assert 'b' in out
    assert 'tau' in out
    assert 'pi' in out

    assert torch.all(out['b'] > 0)
    assert torch.all((out['tau'] > 0) & (out['tau'] < 1))


@pytest.mark.unit
def test_head_cmal_min_scale():
    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.cmal_min_scale = 1e-3
    head = get_head(cfg=cfg, n_in=8, n_out=12, n_hidden=16)
    with torch.no_grad():
        head.fc2.weight[3:6] = 0.0
        head.fc2.bias[3:6] = -50.0  # push the width outputs far below zero
    out = head(torch.randn(4, 8))
    assert torch.all(out['b'] >= 1e-3)


@pytest.mark.unit
@pytest.mark.parametrize('bias', [50.0, -50.0])
def test_head_cmal_tau_stays_inside_0_1(bias):
    """tau must never reach exactly 0 or 1: the loss takes log(tau) and log(1 - tau)."""
    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.cmal_min_scale = 1e-3
    head = get_head(cfg=cfg, n_in=8, n_out=12, n_hidden=16)
    with torch.no_grad():
        head.fc2.weight[6:9] = 0.0
        head.fc2.bias[6:9] = bias  # push the lean far to one side
    out = head(torch.randn(4, 8))
    t = out['tau']
    assert t.dtype == torch.float32
    assert torch.all(t < 1) and torch.all(t > 0)
    assert torch.isfinite(torch.log(t) + torch.log(1 - t)).all()


@pytest.mark.unit
def test_head_cmal_outputs_float32_under_half_precision_input():
    """Distribution parameters come out in float32 even if the layers ran in float16."""
    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.cmal_min_scale = 1e-3
    head = get_head(cfg=cfg, n_in=8, n_out=12, n_hidden=16).half()
    with torch.no_grad():
        head.fc2.weight[6:9] = 0.0
        head.fc2.bias[6:9] = 20.0  # sigmoid(20) is exactly 1 in float16
    out = head(torch.randn(4, 8).half())
    assert all(v.dtype == torch.float32 for v in out.values())
    assert torch.all(out['tau'] < 1)


@pytest.mark.unit
def test_head_invalid_type():
    cfg_empty = MagicMock(head='', model='lstm')
    with pytest.raises(ValueError, match="No 'head' specified"):
        get_head(cfg=cfg_empty, n_in=32, n_out=1)

    cfg_unsupported = MagicMock(head='unknown_head')
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_head(cfg=cfg_unsupported, n_in=32, n_out=1)


@pytest.mark.unit
def test_base_model_methods(monkeypatch):
    mock_sample_fn = MagicMock(return_value={'y_hat': torch.zeros(1)})
    monkeypatch.setattr(
        'googlehydrology.modelzoo.basemodel.sample_pointpredictions',
        mock_sample_fn,
    )
    monkeypatch.setattr(
        'googlehydrology.modelzoo.basemodel.Scaler', MagicMock()
    )

    cfg = MagicMock()
    cfg.target_variables = ['streamflow']
    cfg.head = 'regression'
    cfg.base_run_dir = None
    cfg.run_dir = None
    cfg.is_finetuning = False

    class SimpleModel(BaseModel):
        def forward(self, data):
            return data

    model = SimpleModel(cfg)
    data = {'x': torch.zeros(1)}

    # Test pre_model_hook
    hook_out = model.pre_model_hook(data, is_train=True)
    assert hook_out == data

    # Test sample method
    samples = model.sample(data, n_samples=5)
    assert 'y_hat' in samples
    mock_sample_fn.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize('save_compiled', [True, False])
@pytest.mark.parametrize('eval_compiled', [True, False])
def test_load_weights_torch_compile_compatibility(
    tmp_path, save_compiled, eval_compiled
):
    from googlehydrology.evaluation.tester import BaseTester
    from googlehydrology.training.basetrainer import BaseTrainer

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    if save_compiled:
        model = torch.compile(model)

    weight_path = tmp_path / 'model_epoch001.pt'
    torch.save(model.state_dict(), weight_path)

    tester = BaseTester.__new__(BaseTester)
    tester.run_dir = tmp_path
    tester.device = 'cpu'
    eval_model = DummyModel()
    if eval_compiled:
        eval_model = torch.compile(eval_model)
    tester.model = eval_model
    tester._load_weights(epoch=1)

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.device = 'cpu'
    train_model = DummyModel()
    if eval_compiled:
        train_model = torch.compile(train_model)
    trainer.model = train_model
    trainer._load_model_weights(weight_path)



@pytest.mark.unit
def test_cmal_loss_finite_when_lean_is_extreme():
    """The CMAL loss and its gradients stay finite when the head pushes tau to the edge."""

    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.cmal_min_scale = 1e-3
    head = get_head(cfg=cfg, n_in=8, n_out=12, n_hidden=16)
    with torch.no_grad():
        head.fc2.weight[6:9] = 0.0
        head.fc2.bias[6:9] = 40.0  # tau would be exactly 1 with the old formula
    x = torch.randn(5, 7, 8, requires_grad=True)
    out = head(x)
    # log density of one asymmetric Laplace, as in the loss: log t + log(1-t) - log b - ...
    t, b = out['tau'], out['b']
    logp = torch.log(t) + torch.log(1 - t) - torch.log(b)
    logp.sum().backward()
    assert torch.isfinite(logp).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
