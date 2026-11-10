from chemprop.data.collate import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.nn.predictors import (
    BinaryClassificationFFN,
    EvidentialFFN,
    MveFFN,
    QuantileFFN,
    RegressionFFN,
)
from chemprop.nn.transforms import GraphTransform, ScaleTransform, UnscaleTransform
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from chemprop_contrib.reaction_parity.featurizers import ParityCGRFeaturizer, reverse_bmg
from chemprop_contrib.reaction_parity.models import ParityEquivariantMPNN
from chemprop_contrib.reaction_parity.parity import EVEN, FREE, ODD

from .conftest import make_rxn, reverse_rxn_smi, rxn_smis

D_H = 16
PARITY = [ODD, EVEN, FREE]
# the fixture reactions have no atom with two bonds, so D-MPNN message updates (W_h) are inert
# on them; this one has a three-atom chain and exercises every encoder weight
CHAIN_SMI = "[CH3:1][CH2:2][OH:3]>>[CH2:1]=[CH2:2].[OH2:3]"


def _featurize(smis, feat=None):
    feat = feat or ParityCGRFeaturizer()
    return feat, BatchMolGraph([feat(make_rxn(s)) for s in smis])


def _reverse(feat, bmg):
    return reverse_bmg(bmg, feat.odd_atom_start, feat.odd_bond_start)


def _model(
    feat=None,
    predictor=None,
    task_parity=PARITY,
    batch_norm=False,
    d_vd=None,
    d_xd=0,
    graph_transform=None,
    **kwargs,
):
    feat = feat or ParityCGRFeaturizer()
    torch.manual_seed(0)
    mp = BondMessagePassing(
        d_v=feat.atom_fdim,
        d_e=feat.bond_fdim,
        d_h=D_H,
        d_vd=d_vd,
        graph_transform=graph_transform,
    )
    predictor = predictor or RegressionFFN(
        n_tasks=len(task_parity), input_dim=mp.output_dim + d_xd, hidden_dim=D_H
    )
    return ParityEquivariantMPNN.from_featurizer(
        feat, mp, MeanAggregation(), predictor, task_parity, batch_norm=batch_norm, **kwargs
    )


def _batch(bmg, n_tasks=3, V_d=None, X_d=None, targets=None):
    B = len(bmg)
    targets = torch.randn(B, n_tasks) if targets is None else targets
    return (
        bmg,
        V_d,
        X_d,
        targets,
        torch.ones(B, 1),
        torch.zeros(B, n_tasks, dtype=torch.bool),
        torch.zeros(B, n_tasks, dtype=torch.bool),
    )


class _MetricSpy(torch.nn.Module):
    """Module wrapper so it can replace an entry of ``model.metrics`` (a ``ModuleList``)."""

    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.preds = []

    def forward(self, preds, *args):
        self.preds.append(preds)
        return self.metric(preds, *args)


def _record_logs(model, monkeypatch):
    # LightningModule.log warns when no Trainer is attached; record the calls instead so the
    # step methods can be unit-tested and their logging checked.
    calls = []
    monkeypatch.setattr(model, "log", lambda name, value, **kw: calls.append((name, value, kw)))
    return calls


# ---------------------------------------------------------------------------------------------
# R3 / R4 in eval()


@torch.no_grad()
def test_odd_task_is_exactly_antisymmetric(rxn_smi):
    feat, bmg = _featurize([rxn_smi])
    model = _model(feat).eval()
    f = model(bmg)[:, 0]
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 0], -f, rtol=0, atol=1e-5)

    _, bmg_rev = _featurize([reverse_rxn_smi(rxn_smi)], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 0], -f, rtol=0, atol=1e-5)


@torch.no_grad()
def test_even_task_is_exactly_symmetric(rxn_smi):
    feat, bmg = _featurize([rxn_smi])
    model = _model(feat).eval()
    f = model(bmg)[:, 1]
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 1], f, rtol=0, atol=1e-5)

    _, bmg_rev = _featurize([reverse_rxn_smi(rxn_smi)], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 1], f, rtol=0, atol=1e-5)


@torch.no_grad()
def test_free_task_equals_forward_pass(rxn_smi):
    feat, bmg = _featurize([rxn_smi])
    model = _model(feat).eval()
    expected = model.predictor(model.fingerprint(bmg))[:, 2]
    torch.testing.assert_close(model(bmg)[:, 2], expected, rtol=0, atol=1e-5)
    # and it is not constrained: the reversed reaction gives a different value
    assert not torch.allclose(model(_reverse(feat, bmg))[:, 2], expected, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# R8: training and validation paths


def test_training_step_applies_the_parity_combination(monkeypatch):
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    model = _model(feat).train()
    logs = _record_logs(model, monkeypatch)
    spy = _MetricSpy(model.predictor.criterion)
    monkeypatch.setattr(model.predictor, "criterion", spy)

    targets = torch.randn(len(bmg), 3)
    targets[0, 1] = float("nan")
    loss = model.training_step(_batch(bmg, targets=targets), 0)
    model.training_step(_batch(_reverse(feat, bmg), targets=targets), 0)

    f, f_rev = spy.preds
    assert f.shape == (len(bmg), 3)
    # no batch norm and no dropout, so train mode is exact as well
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    # the objective sees the combined output, not the forward pass alone
    raw = model.predictor.train_step(model.fingerprint(bmg))
    assert not torch.allclose(f[:, 0], raw[:, 0], atol=1e-5)
    torch.testing.assert_close(f[:, 2], raw[:, 2], rtol=0, atol=1e-5)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert [name for name, *_ in logs] == ["train_loss", "train_loss"]
    assert logs[0][2]["batch_size"] == len(bmg)

    # gradients reach every encoder parameter through the combined objective
    model.zero_grad()
    model.training_step(_batch(bmg, targets=targets), 0).backward()
    for name, p in model.message_passing.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_validation_step_applies_the_parity_combination(monkeypatch):
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat)
    model.on_validation_model_eval()
    logs = _record_logs(model, monkeypatch)
    spy = _MetricSpy(model.metrics[-1])
    model.metrics[-1] = spy

    with torch.no_grad():
        model.validation_step(_batch(bmg), 0)
        model.validation_step(_batch(_reverse(feat, bmg)), 0)

    f, f_rev = spy.preds
    assert f.shape == (len(bmg), 3)
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)
    names = [name for name, *_ in logs]
    assert "val_loss" in names and any(n.startswith("val/") for n in names)


# ---------------------------------------------------------------------------------------------
# public contract of fingerprint / encoding


@torch.no_grad()
def test_fingerprint_keeps_batch_dim_B():
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, batch_norm=True).eval()
    fp = model.fingerprint(bmg)
    assert fp.shape == (len(bmg), D_H)
    torch.testing.assert_close(fp, MPNN.fingerprint(model, bmg), rtol=0, atol=0)
    paired = model._paired_fingerprint(bmg)
    assert paired.shape == (2 * len(bmg), D_H)
    torch.testing.assert_close(paired[: len(bmg)], fp, rtol=0, atol=1e-6)


@torch.no_grad()
def test_encoding_is_unchanged():
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat).eval()
    for i in (0, 1, -1):
        torch.testing.assert_close(
            model.encoding(bmg, i=i), MPNN.encoding(model, bmg, i=i), rtol=0, atol=0
        )
        assert model.encoding(bmg, i=i).shape[0] == len(bmg)


# ---------------------------------------------------------------------------------------------
# R7 validation and the target-scaler mean


def test_rejects_nonzero_odd_task_target_mean():
    transform = UnscaleTransform(mean=[0.0, 1.0, 2.0, 3.0], scale=[1.0] * 4)
    predictor = RegressionFFN(n_tasks=4, input_dim=D_H, output_transform=transform)
    with pytest.raises(ValueError, match=r"scaler\.mean_\[odd_cols\] = 0\.0") as e:
        _model(predictor=predictor, task_parity=[ODD, ODD, EVEN, ODD])
    assert "[1, 3]" in str(e.value)


def _targets_with_nonzero_means():
    rng = np.random.default_rng(0)
    return rng.normal(loc=[3.0, -2.0, 5.0], scale=[1.0, 2.0, 0.5], size=(20, 3))


def test_rejects_with_mean_false_scaler():
    # pitfall: with_mean=False still sets mean_, and from_standard_scaler copies it
    scaler = StandardScaler(with_mean=False).fit(_targets_with_nonzero_means())
    assert np.all(scaler.mean_ != 0)
    transform = UnscaleTransform.from_standard_scaler(scaler)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H, output_transform=transform)
    with pytest.raises(ValueError, match=r"\[0\]"):
        _model(predictor=predictor)


@torch.no_grad()
def test_accepts_scaler_with_zeroed_odd_mean():
    Y = _targets_with_nonzero_means()
    scaler = StandardScaler().fit(Y)
    odd_cols = [i for i, p in enumerate(PARITY) if p == ODD]
    scaler.mean_[odd_cols] = 0.0
    # transform and unscale stay consistent: odd column is only divided by its scale
    np.testing.assert_allclose(scaler.transform(Y)[:, 0], Y[:, 0] / scaler.scale_[0])

    transform = UnscaleTransform.from_standard_scaler(scaler)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H, output_transform=transform)
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, predictor=predictor).eval()
    f = model(bmg)
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 0], -f[:, 0], rtol=0, atol=1e-5)


def test_accepts_nonzero_mean_on_even_and_free_tasks():
    transform = UnscaleTransform(mean=[0.0, 5.0, 7.0], scale=[1.0] * 3)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H, output_transform=transform)
    _model(predictor=predictor)


@torch.no_grad()
def test_zero_mean_odd_task_is_unbiased():
    mean, scale = [0.0, 5.0, 7.0], [2.0, 3.0, 4.0]
    transform = UnscaleTransform(mean=mean, scale=scale)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H, output_transform=transform)
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, predictor=predictor).eval()

    g_f = model.predictor.ffn(model.fingerprint(bmg))
    g_r = model.predictor.ffn(model.fingerprint(_reverse(feat, bmg)))
    out = model(bmg)
    torch.testing.assert_close(out[:, 0], 0.5 * (g_f - g_r)[:, 0] * 2.0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out[:, 1], 0.5 * (g_f + g_r)[:, 1] * 3.0 + 5.0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out[:, 2], g_f[:, 2] * 4.0 + 7.0, rtol=0, atol=1e-5)
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 0], -out[:, 0], rtol=0, atol=1e-5)


def test_rejects_non_identity_graph_transform():
    feat = ParityCGRFeaturizer()
    V_t = ScaleTransform(mean=[0.0] * feat.atom_fdim, scale=[1.0] * feat.atom_fdim)
    E_t = ScaleTransform(mean=[0.0] * feat.bond_fdim, scale=[1.0] * feat.bond_fdim)
    with pytest.raises(ValueError, match="graph_transform"):
        _model(feat, graph_transform=GraphTransform(V_t, E_t))


def test_rejects_disallowed_predictor():
    predictor = BinaryClassificationFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="BinaryClassificationFFN"):
        _model(predictor=predictor)


def test_rejects_wrong_task_parity_length():
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="n_tasks"):
        _model(predictor=predictor, task_parity=[ODD, EVEN])


def test_task_parity_and_mask_are_stored():
    model = _model(task_parity=(ODD, EVEN, FREE))
    assert model.task_parity == [ODD, EVEN, FREE]
    assert all(type(p) is int for p in model.task_parity)
    assert "parity_mask" not in model.state_dict()
    assert torch.equal(model.parity_mask, torch.tensor([-1.0, 1.0, 0.0]))
    feat = ParityCGRFeaturizer()
    assert model.odd_atom_start == feat.odd_atom_start
    assert model.odd_bond_start == feat.odd_bond_start
    assert model.hparams["odd_atom_start"] == feat.odd_atom_start
    assert model.hparams["odd_bond_start"] == feat.odd_bond_start
    assert model.hparams["task_parity"] == [ODD, EVEN, FREE]


# ---------------------------------------------------------------------------------------------
# extra descriptors, batch norm, shapes


def test_extra_descriptors_reach_both_passes():
    d_xd = 4
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, d_xd=d_xd).eval()
    X_d = torch.randn(len(bmg), d_xd)
    with torch.no_grad():
        paired = model._paired_fingerprint(bmg, X_d=X_d)
        assert paired.shape == (2 * len(bmg), D_H + d_xd)
        torch.testing.assert_close(paired[: len(bmg), D_H:], X_d, rtol=0, atol=0)
        torch.testing.assert_close(paired[len(bmg) :, D_H:], X_d, rtol=0, atol=0)
        f = model(bmg, X_d=X_d)
        f_rev = model(_reverse(feat, bmg), X_d=X_d)
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)


def test_v_d_is_forwarded_to_both_passes(monkeypatch):
    d_vd = 3
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, d_vd=d_vd).eval()
    V_d = torch.randn(bmg.V.shape[0], d_vd)

    seen = []
    original = model.message_passing.forward

    def spy(bmg_, V_d_=None):
        seen.append((bmg_, V_d_))
        return original(bmg_, V_d_)

    monkeypatch.setattr(model.message_passing, "forward", spy)
    with torch.no_grad():
        f = model(bmg, V_d=V_d)
    assert len(seen) == 2
    assert seen[0][1] is V_d and seen[1][1] is V_d
    assert seen[0][0] is bmg
    torch.testing.assert_close(seen[1][0].V, _reverse(feat, bmg).V, rtol=0, atol=0)

    with torch.no_grad():
        f_rev = model(_reverse(feat, bmg), V_d=V_d)
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)


def test_antisymmetry_holds_with_batch_norm(rxn_smi):
    feat, bmg = _featurize([rxn_smi])
    model = _model(feat, batch_norm=True)

    # Train mode: both passes are normalised together, so the forward and reversed inputs see
    # identical batch statistics and the combination stays exact. No output/graph transforms are
    # attached, so their train-mode short-circuit cannot make this pass for unrelated reasons.
    model.train()
    with torch.no_grad():
        f = model(bmg)
        f_rev = model(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)

    # Eval mode with non-trivial running statistics
    with torch.no_grad():
        model.bn.running_mean.uniform_(-1, 1)
        model.bn.running_var.uniform_(0.5, 2)
    model.eval()
    with torch.no_grad():
        f = model(bmg)
        f_rev = model(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 0], -f[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1], f[:, 1], rtol=0, atol=1e-5)


@pytest.mark.parametrize("cls", [RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN])
@torch.no_grad()
def test_output_shape_matches_plain_mpnn(cls):
    feat, bmg = _featurize(rxn_smis)
    model = _model(feat, predictor=cls(n_tasks=3, input_dim=D_H, hidden_dim=D_H)).eval()
    plain = MPNN(model.message_passing, model.agg, model.predictor).eval()
    assert model(bmg).shape == plain(bmg).shape
    assert model.predictor.train_step(model._paired_fingerprint(bmg)).shape[0] == 2 * len(bmg)
    assert model._combine(model.predictor.train_step(model._paired_fingerprint(bmg))).shape == (
        plain.predictor.train_step(plain.fingerprint(bmg)).shape
    )
