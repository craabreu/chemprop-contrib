"""Tests of ParitySwapMPNN: parity under the swap of an ordered pair of molecules (A, B)."""

from chemprop.data import (
    BatchMolGraph,
    MoleculeDatapoint,
    MoleculeDataset,
    MulticomponentDataset,
    build_dataloader,
)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.nn.predictors import MveFFN, RegressionFFN
from chemprop.utils import make_mol
import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from chemprop_contrib.reversal_parity.models import ParitySwapMPNN
from chemprop_contrib.reversal_parity.targets import ParityTargetTransform

from .test_models import _trainer, lightning_warnings

D_H = 16
# no atom mapping, and no correspondence between the two molecules of a pair
PAIRS = [
    ("Oc1ccccc1", "Cc1ccccc1"),
    ("CCO", "CCN"),
    ("CC(=O)O", "CCC"),
    ("c1ccncc1", "c1ccccc1"),
    ("CC(C)O", "O"),
    ("CCCl", "CC.Cl"),  # one side may hold several molecules, as a disconnected graph
    ("C1CCOC1", "CCOCC"),
    ("NC=O", "CC#N"),
]
# columns: odd, even, the two directions of a pair, free
SPEC = dict(odd=[0], even=[1], paired_reverse={2: 3})
N_TASKS = 5


def _graphs(smis):
    feat = SimpleMoleculeMolGraphFeaturizer()
    return BatchMolGraph([feat(make_mol(smi, keep_h=False, add_h=False)) for smi in smis])


def _pair_graphs():
    return [_graphs(pair[k] for pair in PAIRS) for k in (0, 1)]


def _model(cls=RegressionFFN, target=None, d_xd=0, width=2):
    target = ParityTargetTransform(N_TASKS, **SPEC) if target is None else target
    torch.manual_seed(0)
    mp = BondMessagePassing(d_h=D_H)
    predictor = cls(n_tasks=N_TASKS, input_dim=width * mp.output_dim + d_xd, hidden_dim=D_H)
    return ParitySwapMPNN(mp, MeanAggregation(), predictor, target, batch_norm=True)


def _assert_swapped(y, y_swap, sign):
    """Columns of ``y`` for (A, B) against ``y_swap`` for (B, A); ``sign`` is the parity of the
    odd column's values (-1) or of every column's uncertainty components (+1)."""
    close = dict(rtol=0, atol=1e-5)
    torch.testing.assert_close(y_swap[:, 0], sign * y[:, 0], **close)
    torch.testing.assert_close(y_swap[:, 1], y[:, 1], **close)
    torch.testing.assert_close(y_swap[:, [2, 3]], y[:, [3, 2]], **close)
    assert not torch.allclose(y_swap[:, 4], y[:, 4], atol=1e-3)  # free: order-dependent


@torch.no_grad()
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("cls", [RegressionFFN, MveFFN])
def test_swapping_the_pair_acts_on_each_column_by_its_parity(cls, training):
    A, B = _pair_graphs()
    X_d = torch.randn(len(PAIRS), 2)
    # batch norm in train mode uses the statistics of the batch, which the swap must not change
    model = _model(cls, d_xd=2).train(training)

    f, f_swap = model([A, B], X_d=X_d), model([B, A], X_d=X_d)

    if cls is RegressionFFN:
        _assert_swapped(f, f_swap, sign=-1)
    else:
        _assert_swapped(f[..., 0], f_swap[..., 0], sign=-1)
        _assert_swapped(f[..., 1], f_swap[..., 1], sign=1)
        assert (f[..., 1] > 0).all()


@torch.no_grad()
def test_fingerprint_holds_the_two_molecules_in_order():
    A, B = _pair_graphs()
    model = _model().eval()

    h, h_swap = model.fingerprint([A, B]), model.fingerprint([B, A])

    assert h.shape == (len(PAIRS), 2 * D_H)
    torch.testing.assert_close(h_swap, h.roll(D_H, dims=1), rtol=0, atol=1e-6)


def test_rejects_a_predictor_too_narrow_for_two_fingerprints():
    with pytest.raises(ValueError, match="input_dim"):
        _model(width=1)


def test_rejects_inputs_that_are_not_pairs():
    A, B = _pair_graphs()
    with pytest.raises(ValueError, match="two components"):
        _model()([A, B, A])


@torch.no_grad()
@lightning_warnings
def test_swap_model_trains_and_reloads(tmp_path):
    pl.seed_everything(0)
    ys = np.random.default_rng(0).normal(size=(len(PAIRS), N_TASKS))
    data = MulticomponentDataset(
        [
            MoleculeDataset([MoleculeDatapoint.from_smi(p[k], y=y) for p, y in zip(PAIRS, ys)])
            for k in (0, 1)
        ]
    )
    target = ParityTargetTransform.fit(ys, **SPEC)
    data.normalize_targets(target.to_standard_scaler())
    loader = build_dataloader(data, batch_size=4, num_workers=0, seed=0)
    model = _model(target=target)
    trainer = _trainer(max_epochs=2)
    with torch.enable_grad():
        trainer.fit(model, loader, loader)
    path = tmp_path / "swap.ckpt"
    trainer.save_checkpoint(path)

    loaded = ParitySwapMPNN.load_from_checkpoint(path).eval()

    A, B = _pair_graphs()
    f = loaded([A, B])
    torch.testing.assert_close(f, model.eval()([A, B]), rtol=0, atol=1e-5)
    _assert_swapped(f, loaded([B, A]), sign=-1)
    predicted = torch.cat(
        trainer.predict(loaded, build_dataloader(data, num_workers=0, shuffle=False))
    )
    torch.testing.assert_close(predicted, f, rtol=0, atol=1e-5)
