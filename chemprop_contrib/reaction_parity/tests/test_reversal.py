from chemprop.data.collate import BatchMolGraph
import pytest
import torch

from chemprop_contrib.reaction_parity.featurizers import ParityCGRFeaturizer, reverse_bmg

from .conftest import encode, make_rxn, reverse_rxn_smi

ZERO_EDGE_SMI = "[CH4:1]>>[CH3:1].[H:2]"


def _forward(smi, balanced=False):
    feat = ParityCGRFeaturizer(balanced=balanced)
    return feat, BatchMolGraph([feat(make_rxn(smi))])


def _reverse(feat, bmg):
    return reverse_bmg(bmg, feat.odd_atom_start, feat.odd_bond_start)


def test_shares_topology_tensors(rxn_smi):
    feat, bmg = _forward(rxn_smi)
    out = _reverse(feat, bmg)
    assert out.edge_index is bmg.edge_index
    assert out.rev_edge_index is bmg.rev_edge_index
    assert out.batch is bmg.batch


def test_does_not_mutate_input(rxn_smi):
    feat, bmg = _forward(rxn_smi)
    V, E = bmg.V.clone(), bmg.E.clone()
    _reverse(feat, bmg)
    torch.testing.assert_close(bmg.V, V, rtol=0, atol=0)
    torch.testing.assert_close(bmg.E, E, rtol=0, atol=0)


def test_is_an_involution(rxn_smi):
    feat, bmg = _forward(rxn_smi)
    out = _reverse(feat, _reverse(feat, bmg))
    torch.testing.assert_close(out.V, bmg.V, rtol=0, atol=0)
    torch.testing.assert_close(out.E, bmg.E, rtol=0, atol=0)


def test_handles_zero_edge_reaction():
    feat, bmg = _forward(ZERO_EDGE_SMI)
    assert bmg.E.shape == (0, feat.bond_fdim)
    out = _reverse(feat, bmg)
    assert out.E.shape == (0, feat.bond_fdim)
    assert out.V.shape == bmg.V.shape


def test_preserves_len(rxn_smi):
    feat, bmg = _forward(rxn_smi)
    assert len(_reverse(feat, bmg)) == len(bmg)


@pytest.mark.parametrize("balanced", [False, True])
def test_matches_refeaturized_reverse(rxn_smi, balanced):
    feat, bmg = _forward(rxn_smi, balanced)
    expected = encode(BatchMolGraph([feat(make_rxn(reverse_rxn_smi(rxn_smi)))]))
    actual = encode(_reverse(feat, bmg))
    assert torch.allclose(actual, expected, atol=1e-5)
