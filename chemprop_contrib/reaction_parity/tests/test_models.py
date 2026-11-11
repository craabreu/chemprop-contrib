import csv
from pathlib import Path

from chemprop.data import ReactionDataset, ReactionDatapoint, build_dataloader
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
import lightning.pytorch as pl
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from chemprop_contrib.reaction_parity.featurizers import ParityCGRFeaturizer, reverse_bmg
from chemprop_contrib.reaction_parity.models import ParityEquivariantMPNN
from chemprop_contrib.reaction_parity.parity import EVEN, FREE, ODD

from . import make_test_data
from .conftest import alchemical_smis, make_rxn, reverse_rxn_smi, rxn_smis

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
def test_odd_task_is_exactly_antisymmetric(parity_smi):
    feat, bmg = _featurize([parity_smi])
    model = _model(feat).eval()
    f = model(bmg)[:, 0]
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 0], -f, rtol=0, atol=1e-5)

    _, bmg_rev = _featurize([reverse_rxn_smi(parity_smi)], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 0], -f, rtol=0, atol=1e-5)


@torch.no_grad()
def test_even_task_is_exactly_symmetric(parity_smi):
    feat, bmg = _featurize([parity_smi])
    model = _model(feat).eval()
    f = model(bmg)[:, 1]
    torch.testing.assert_close(model(_reverse(feat, bmg))[:, 1], f, rtol=0, atol=1e-5)

    _, bmg_rev = _featurize([reverse_rxn_smi(parity_smi)], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 1], f, rtol=0, atol=1e-5)


@torch.no_grad()
def test_free_task_equals_forward_pass(parity_smi):
    feat, bmg = _featurize([parity_smi])
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
@pytest.mark.parametrize("batch_norm", [False, True])
def test_paired_fingerprint_equals_two_separate_encoder_passes(batch_norm):
    # the single stacked pass must compute exactly what encoding x and x̄ separately computes
    d_vd, d_xd = 3, 2
    feat, bmg = _featurize(rxn_smis + alchemical_smis + [CHAIN_SMI])
    model = _model(feat, batch_norm=batch_norm, d_vd=d_vd, d_xd=d_xd).double().eval()
    bmg.V, bmg.E = bmg.V.double(), bmg.E.double()
    gen = torch.Generator().manual_seed(0)
    V_d = torch.randn(bmg.V.shape[0], d_vd, generator=gen, dtype=torch.float64)
    X_d = torch.randn(len(bmg), d_xd, generator=gen, dtype=torch.float64)

    H_f = model.agg(model.message_passing(bmg, V_d), bmg.batch)
    H_r = model.agg(model.message_passing(_reverse(feat, bmg), V_d), bmg.batch)
    H = model.bn(torch.cat([H_f, H_r]))
    expected = torch.cat((H, model.X_d_transform(X_d).repeat(2, 1)), dim=1)

    actual = model._paired_fingerprint(bmg, V_d, X_d)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


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


def _scale_transform(d):
    g = torch.Generator().manual_seed(d)
    return ScaleTransform(
        mean=torch.rand(d, generator=g).tolist(), scale=(0.5 + torch.rand(d, generator=g)).tolist()
    )


@pytest.mark.parametrize("rxn_smi", rxn_smis)
def test_parity_is_exact_with_atom_and_bond_feature_scaling(rxn_smi):
    # the graph transform runs inside the encoder, after reverse_bmg, so it is part of g: parity
    # holds even with nonzero means on the odd (difference) columns
    feat, bmg = _featurize([rxn_smi])
    gt = GraphTransform(_scale_transform(feat.atom_fdim), _scale_transform(feat.bond_fdim))
    assert torch.all(gt.V_transform.mean[0, feat.odd_atom_start :] != 0)
    model = _model(feat, graph_transform=gt).eval()
    _, bmg_rev = _featurize([reverse_rxn_smi(rxn_smi)], feat)
    out, out_rev = model(bmg), model(bmg_rev)
    torch.testing.assert_close(out_rev[:, 0], -out[:, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(out_rev[:, 1], out[:, 1], rtol=0, atol=1e-5)
    # the transform is active, not inert
    plain = _model(feat).eval()
    assert not torch.allclose(plain(bmg), out)


def test_from_featurizer_rejects_stock_cgr_featurizer():
    from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

    stock = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    mp = BondMessagePassing(d_v=stock.atom_fdim, d_e=stock.bond_fdim, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(TypeError, match="ParityCGRFeaturizer"):
        ParityEquivariantMPNN.from_featurizer(stock, mp, MeanAggregation(), predictor, PARITY)


@pytest.mark.parametrize("which", ["d_v", "d_e"])
def test_from_featurizer_rejects_encoder_width_mismatch(which):
    feat = ParityCGRFeaturizer()
    d_v = feat.atom_fdim - (1 if which == "d_v" else 0)
    d_e = feat.bond_fdim - (1 if which == "d_e" else 0)
    mp = BondMessagePassing(d_v=d_v, d_e=d_e, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match=f"{which} = .* but the featurizer"):
        ParityEquivariantMPNN.from_featurizer(feat, mp, MeanAggregation(), predictor, PARITY)


def test_start_index_upper_bound_skipped_without_message_passing_dims(monkeypatch):
    feat = ParityCGRFeaturizer()
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H)
    monkeypatch.setattr(type(mp), "hparams", property(lambda self: {}))
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    model = ParityEquivariantMPNN.from_featurizer(feat, mp, MeanAggregation(), predictor, PARITY)
    assert model.odd_atom_start == feat.odd_atom_start
    with pytest.raises(ValueError, match="odd_atom_start must satisfy 0 < odd_atom_start < d_v,"):
        ParityEquivariantMPNN(mp, MeanAggregation(), predictor, 0, feat.odd_bond_start, PARITY)


def test_rejects_disallowed_predictor():
    predictor = BinaryClassificationFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="BinaryClassificationFFN"):
        _model(predictor=predictor)


@pytest.mark.parametrize("start", [0, -1, "atom_fdim"])
def test_rejects_out_of_range_atom_start(start):
    feat = ParityCGRFeaturizer()
    start = feat.atom_fdim if start == "atom_fdim" else start
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="odd_atom_start must satisfy"):
        ParityEquivariantMPNN(mp, MeanAggregation(), predictor, start, feat.odd_bond_start, PARITY)


@pytest.mark.parametrize("start", [0, -1, "bond_fdim"])
def test_rejects_out_of_range_bond_start(start):
    feat = ParityCGRFeaturizer()
    start = feat.bond_fdim if start == "bond_fdim" else start
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="odd_bond_start must satisfy"):
        ParityEquivariantMPNN(mp, MeanAggregation(), predictor, feat.odd_atom_start, start, PARITY)


@pytest.mark.parametrize("bad", [10.5, True])
def test_rejects_non_integral_start(bad):
    feat = ParityCGRFeaturizer()
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=D_H)
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="odd_atom_start must be an integer"):
        ParityEquivariantMPNN(mp, MeanAggregation(), predictor, bad, feat.odd_bond_start, PARITY)
    with pytest.raises(ValueError, match="odd_bond_start must be an integer"):
        ParityEquivariantMPNN(mp, MeanAggregation(), predictor, feat.odd_atom_start, bad, PARITY)


def test_rejects_wrong_task_parity_length():
    predictor = RegressionFFN(n_tasks=3, input_dim=D_H)
    with pytest.raises(ValueError, match="n_tasks"):
        _model(predictor=predictor, task_parity=[ODD, EVEN])


@pytest.mark.parametrize("bad", [[-0.5, 0.7, 1.9], [ODD, True, FREE], [ODD, EVEN, 1.0]])
def test_rejects_non_integer_task_parity(bad):
    with pytest.raises(ValueError, match="Invalid parity values"):
        _model(task_parity=bad)


def test_accepts_numpy_integer_task_parity():
    model = _model(task_parity=np.array(PARITY))
    assert model.task_parity == PARITY
    assert all(type(p) is int for p in model.task_parity)


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


def test_v_d_is_forwarded_to_forward_and_reversed_graphs(monkeypatch):
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
    # one encoder pass over the forward graphs followed by the reversed ones
    assert len(seen) == 1
    both, V_d_seen = seen[0]
    n = bmg.V.shape[0]
    torch.testing.assert_close(both.V[:n], bmg.V, rtol=0, atol=0)
    torch.testing.assert_close(both.V[n:], _reverse(feat, bmg).V, rtol=0, atol=0)
    torch.testing.assert_close(V_d_seen, torch.cat((V_d, V_d)), rtol=0, atol=0)

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


# ---------------------------------------------------------------------------------------------
# R5 / R6: uncertainty predictors


def _odd_even_outputs(cls):
    """Eval-mode outputs ``(f, f_rev)``, each ``(B, 2, n_targets)``, for tasks ``[ODD, EVEN]``."""
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    predictor = cls(n_tasks=2, input_dim=D_H, hidden_dim=D_H)
    model = _model(feat, predictor=predictor, task_parity=[ODD, EVEN]).eval()
    with torch.no_grad():
        return model(bmg), model(_reverse(feat, bmg))


def _assert_component_parity(f, f_rev):
    # component 0 carries the task's parity (odd for task 0, even for task 1); every later
    # component is even, so a positive uncertainty stays positive under reversal
    torch.testing.assert_close(f_rev[:, 0, 0], -f[:, 0, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 1, 0], f[:, 1, 0], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[..., 1:], f[..., 1:], rtol=0, atol=1e-5)


def test_mve_variance_stays_positive_for_odd_task():
    f, f_rev = _odd_even_outputs(MveFFN)
    assert f.shape == (len(rxn_smis) + 1, 2, 2)
    for out in (f, f_rev):
        assert (out[..., 1] > 0).all()
    _assert_component_parity(f, f_rev)


def test_evidential_params_stay_positive_for_odd_task():
    f, f_rev = _odd_even_outputs(EvidentialFFN)
    assert f.shape == (len(rxn_smis) + 1, 2, 4)
    for out in (f, f_rev):
        v, alpha, beta = out[..., 1], out[..., 2], out[..., 3]
        assert (v > 0).all() and (alpha > 1).all() and (beta > 0).all()
    _assert_component_parity(f, f_rev)


def test_quantile_mean_is_odd_and_interval_is_even():
    # the interval's sign is not guaranteed for an untrained model, so only its parity is checked
    f, f_rev = _odd_even_outputs(QuantileFFN)
    assert f.shape == (len(rxn_smis) + 1, 2, 2)
    _assert_component_parity(f, f_rev)


def test_mve_loss_is_finite_for_odd_task(monkeypatch):
    # a negative variance would make the NLL NaN; every task here is odd, so the variance
    # column must come out of the combination positive
    feat, bmg = _featurize([*rxn_smis, CHAIN_SMI])
    predictor = MveFFN(n_tasks=2, input_dim=D_H, hidden_dim=D_H)
    model = _model(feat, predictor=predictor, task_parity=[ODD, ODD]).train()
    _record_logs(model, monkeypatch)

    assert (model._train_preds(bmg)[..., 1] > 0).all()
    loss = model.training_step(_batch(bmg, n_tasks=2), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------------------------
# R8 / R9: fixture data, training, checkpoint round-trip

FIXTURE_CSV = Path(__file__).with_name("test_data.csv")
FIXTURE_PARITY = [FREE, ODD, EVEN]  # CSV column order: ea, y_odd, y_even
FIXTURE_COLS = ["ea", "y_odd", "y_even"]
N_TRAIN, N_VAL = 24, 8


def _fixture_rows():
    with open(FIXTURE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def test_fixture_targets_have_the_parity_they_claim():
    rows = _fixture_rows()
    assert len(rows) == 100
    assert any(abs(float(r["y_odd"])) > 1e-3 for r in rows), "y_odd must not be vacuous"
    for r in rows[:10]:
        odd, even = make_test_data.derived_targets(reverse_rxn_smi(r["smiles"]))
        assert odd == pytest.approx(-float(r["y_odd"]), abs=1e-9)
        assert even == pytest.approx(float(r["y_even"]), abs=1e-9)


def _datasets(feat, n_train=N_TRAIN, n_val=N_VAL):
    rows = _fixture_rows()[: n_train + n_val]
    dps = [
        ReactionDatapoint.from_smi(
            r["smiles"], y=[float(r[c]) for c in FIXTURE_COLS], keep_h=True, add_h=False
        )
        for r in rows
    ]
    train = ReactionDataset(dps[:n_train], featurizer=feat)
    val = ReactionDataset(dps[n_train:], featurizer=feat)
    # supported scaler recipe (as in the README): zero the odd columns' mean, and scale them by
    # their root mean square about zero
    scaler = StandardScaler().fit(train.Y)
    odd_cols = [i for i, p in enumerate(FIXTURE_PARITY) if p == ODD]
    scaler.mean_[odd_cols] = 0.0
    scaler.scale_[odd_cols] = np.sqrt(np.nanmean(train.Y[:, odd_cols] ** 2, axis=0))
    train.normalize_targets(scaler)
    val.normalize_targets(scaler)
    return train, val, UnscaleTransform.from_standard_scaler(scaler)


def _fixture_model(feat, output_transform):
    torch.manual_seed(0)
    mp = BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=32)
    predictor = RegressionFFN(
        n_tasks=3, input_dim=mp.output_dim, hidden_dim=32, output_transform=output_transform
    )
    return ParityEquivariantMPNN.from_featurizer(
        feat, mp, MeanAggregation(), predictor, FIXTURE_PARITY, batch_norm=True
    )


# Two third-party warnings `trainer.fit` cannot avoid here: Lightning's CombinedLoader calls a
# torch pytree API that torch has deprecated, and the num_workers=0 dataloaders (required for
# a deterministic, fast test) trigger Lightning's "not many workers" advice
lightning_warnings = pytest.mark.filterwarnings(
    "ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated:FutureWarning",
    "ignore:The '.*dataloader' does not have many workers",
)


def _trainer(max_epochs, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=kwargs.pop("enable_checkpointing", False),
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        **kwargs,
    )


@torch.no_grad()
def _eval_loss(model, loader):
    """Criterion on eval-mode combined predictions, in the scaled target space."""
    was_training = model.training
    model.eval()
    # in eval() the output transform unscales, so map back to the space of the scaled targets
    tf = model.predictor.output_transform
    total, n = 0.0, 0
    for bmg, _, _, targets, weights, lt, gt in loader:
        preds = (model(bmg) - tf.mean) / tf.scale
        total += float(model.criterion(preds, targets, targets.isfinite(), weights, lt, gt)) * len(
            bmg
        )
        n += len(bmg)
    model.train(was_training)
    return total / n


def _fit(max_epochs=100, **trainer_kwargs):
    pl.seed_everything(0)
    feat = ParityCGRFeaturizer()
    train, val, transform = _datasets(feat)
    model = _fixture_model(feat, transform)
    train_loader = build_dataloader(train, batch_size=8, num_workers=0, seed=0)
    val_loader = build_dataloader(val, batch_size=8, num_workers=0, shuffle=False)
    eval_loader = build_dataloader(train, batch_size=N_TRAIN, num_workers=0, shuffle=False)
    before = _eval_loss(model, eval_loader)
    trainer = _trainer(max_epochs, **trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)
    return feat, model, trainer, before, _eval_loss(model, eval_loader), val


def test_gradients_reach_all_encoder_parameters(monkeypatch):
    feat = ParityCGRFeaturizer()
    train, _, transform = _datasets(feat)
    model = _fixture_model(feat, transform)
    _record_logs(model, monkeypatch)
    batch = next(iter(build_dataloader(train, batch_size=N_TRAIN, num_workers=0, shuffle=False)))
    model.train().zero_grad()
    model.training_step(batch, 0).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


@lightning_warnings
def test_training_reduces_loss_on_an_exactly_odd_target():
    # meaningful because y_odd genuinely has odd parity: a constrained model can fit a
    # function of the parity it exists for
    _, _, _, before, after, _ = _fit()
    assert after < 0.1 * before, (before, after)


@lightning_warnings
@torch.no_grad()
def test_antisymmetry_survives_training():
    feat, model, _, _, _, val = _fit(max_epochs=10)
    model.eval()
    bmg = BatchMolGraph([val[i].mg for i in range(len(val))])
    f = model(bmg)
    f_rev = model(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 1], -f[:, 1], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 2], f[:, 2], rtol=0, atol=1e-5)
    assert not torch.allclose(f_rev[:, 0], f[:, 0], atol=1e-5)  # free task stays free
    assert f[:, 1].abs().max() > 1e-3  # not vacuous

    # and through the real reversed SMILES, featurized from scratch
    rows = _fixture_rows()[N_TRAIN : N_TRAIN + 3]
    _, bmg_rev = _featurize([reverse_rxn_smi(r["smiles"]) for r in rows], feat)
    _, bmg_fwd = _featurize([r["smiles"] for r in rows], feat)
    torch.testing.assert_close(model(bmg_rev)[:, 1], -model(bmg_fwd)[:, 1], rtol=0, atol=1e-5)


@torch.no_grad()
@lightning_warnings
def test_checkpoint_round_trip_preserves_parity(tmp_path):
    feat, model, trainer, _, _, val = _fit(max_epochs=3)
    path = tmp_path / "parity.ckpt"
    trainer.save_checkpoint(path)

    loaded = ParityEquivariantMPNN.load_from_checkpoint(path)
    for hp in (loaded.hparams, model.hparams):
        assert hp["odd_atom_start"] == feat.odd_atom_start
        assert hp["odd_bond_start"] == feat.odd_bond_start
        assert hp["task_parity"] == FIXTURE_PARITY
    assert loaded.odd_atom_start == feat.odd_atom_start
    assert loaded.odd_bond_start == feat.odd_bond_start
    assert loaded.task_parity == FIXTURE_PARITY
    assert torch.equal(loaded.parity_mask, model.parity_mask)

    model.eval()
    loaded.eval()
    bmg = BatchMolGraph([val[i].mg for i in range(len(val))])
    f = loaded(bmg)
    torch.testing.assert_close(f, model(bmg), rtol=0, atol=1e-5)
    f_rev = loaded(_reverse(feat, bmg))
    torch.testing.assert_close(f_rev[:, 1], -f[:, 1], rtol=0, atol=1e-5)
    torch.testing.assert_close(f_rev[:, 2], f[:, 2], rtol=0, atol=1e-5)

    # chemprop's own loader rebuilds the model from the same checkpoint
    from_file = ParityEquivariantMPNN.load_from_file(path).eval()
    assert from_file.task_parity == FIXTURE_PARITY
    assert torch.equal(from_file.parity_mask, model.parity_mask)
    torch.testing.assert_close(from_file(bmg), model(bmg), rtol=0, atol=1e-5)
