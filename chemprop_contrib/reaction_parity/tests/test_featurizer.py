from chemprop.featurizers.molgraph.reaction import CondensedGraphOfReactionFeaturizer
import numpy as np
import pytest

from chemprop_contrib.reaction_parity.featurizers import ParityCGRFeaturizer

from .conftest import make_rxn


def test_dimensions_match_core():
    ours = ParityCGRFeaturizer()
    core = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    assert ours.atom_fdim == core.atom_fdim
    assert ours.bond_fdim == core.bond_fdim
    F = len(ours.atom_featurizer)
    k = len(ours.atom_featurizer.atomic_nums) + 1
    assert ours.atom_fdim == 2 * F - k
    assert ours.bond_fdim == 2 * len(ours.bond_featurizer)
    assert ours.odd_atom_start == F
    assert ours.odd_bond_start == len(ours.bond_featurizer)


@pytest.mark.parametrize("balanced", [False, True])
def test_is_invertible_map_of_reac_diff(rxn_smi, balanced):
    ours = ParityCGRFeaturizer(balanced=balanced)
    core = CondensedGraphOfReactionFeaturizer(
        mode_="REAC_DIFF_BALANCE" if balanced else "REAC_DIFF"
    )
    mg_ours = ours(make_rxn(rxn_smi))
    mg_core = core(make_rxn(rxn_smi))

    F, Bb = ours.odd_atom_start, ours.odd_bond_start
    k = len(ours.atom_featurizer.atomic_nums) + 1

    V = mg_ours.V.copy()
    V[:, k:F] -= 0.5 * V[:, F:]
    np.testing.assert_array_equal(V, mg_core.V)

    E = mg_ours.E.copy()
    E[:, :Bb] -= 0.5 * E[:, Bb:]
    np.testing.assert_array_equal(E, mg_core.E)

    np.testing.assert_array_equal(mg_ours.edge_index, mg_core.edge_index)
    np.testing.assert_array_equal(mg_ours.rev_edge_index, mg_core.rev_edge_index)


def test_rejects_explicit_mode():
    with pytest.raises(ValueError):
        ParityCGRFeaturizer(mode_="REAC_PROD")


def test_asserts_shared_atomic_number_block():
    # element-inconsistent atom map: mapped C on the reactant side, N on the product side
    rxn = make_rxn("[CH4:1]>>[NH3:1]")
    with pytest.raises(ValueError):
        ParityCGRFeaturizer()(rxn)
