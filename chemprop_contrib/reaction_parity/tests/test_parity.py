import pytest
import torch

from chemprop.nn.predictors import (
    BinaryClassificationFFN,
    EvidentialFFN,
    MulticlassClassificationFFN,
    MveFFN,
    QuantileFFN,
    RegressionFFN,
    SpectralFFN,
)

from chemprop_contrib.reaction_parity.parity import EVEN, FREE, ODD, build_parity_mask


def test_constants():
    assert (ODD, EVEN, FREE) == (-1, 1, 0)


def test_regression_mask_is_1d():
    mask = build_parity_mask([ODD, EVEN, FREE], RegressionFFN(n_tasks=3))
    assert mask.shape == (3,)
    assert mask.is_floating_point()
    assert torch.equal(mask, torch.tensor([-1.0, 1.0, 0.0]))


def test_mve_mask_is_2d_with_even_uncertainty():
    mask = build_parity_mask([ODD, ODD], MveFFN(n_tasks=2))
    assert mask.shape == (2, 2)
    assert torch.equal(mask, torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]))


def test_evidential_mask_shape_and_values():
    mask = build_parity_mask([ODD], EvidentialFFN(n_tasks=1))
    assert mask.shape == (1, 4)
    assert torch.equal(mask, torch.tensor([[-1.0, 1.0, 1.0, 1.0]]))


def test_quantile_uses_the_generic_rule():
    mask = build_parity_mask([ODD], QuantileFFN(n_tasks=1))
    assert mask.shape == (1, 2)
    assert torch.equal(mask, torch.tensor([[-1.0, 1.0]]))


def test_free_task_component_zero_stays_free():
    mask = build_parity_mask((FREE, EVEN), MveFFN(n_tasks=2))
    assert torch.equal(mask, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))


@pytest.mark.parametrize("cls", [RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN])
def test_mask_broadcasts_against_predictor_output(cls):
    predictor = cls(n_tasks=3, input_dim=8, hidden_dim=8).eval()
    mask = build_parity_mask([ODD, EVEN, FREE], predictor)
    Z = torch.randn(5, 8)
    out = predictor(Z)
    if isinstance(out, tuple):
        out = out[0]
    assert (mask * out).shape == out.shape


@pytest.mark.parametrize(
    "predictor",
    [
        BinaryClassificationFFN(n_tasks=1),
        MulticlassClassificationFFN(n_classes=3, n_tasks=1),
        SpectralFFN(n_tasks=1),
    ],
    ids=lambda p: type(p).__name__,
)
def test_rejects_predictors_outside_allowlist(predictor):
    with pytest.raises(ValueError, match=type(predictor).__name__):
        build_parity_mask([ODD], predictor)


def test_rejects_regression_subclass_not_in_allowlist():
    class Custom(RegressionFFN):
        pass

    with pytest.raises(ValueError, match="Custom"):
        build_parity_mask([ODD], Custom(n_tasks=1))


def test_rejects_wrong_length_task_parity():
    with pytest.raises(ValueError, match="length"):
        build_parity_mask([ODD, EVEN], RegressionFFN(n_tasks=3))


def test_rejects_invalid_parity_value():
    with pytest.raises(ValueError, match="parity"):
        build_parity_mask([ODD, 2], RegressionFFN(n_tasks=2))


@pytest.mark.parametrize("bad", [0.7, -1.0, True, False, "odd", None])
def test_rejects_non_integer_parity_value(bad):
    with pytest.raises(ValueError, match="Invalid parity values"):
        build_parity_mask([ODD, bad], RegressionFFN(n_tasks=2))
