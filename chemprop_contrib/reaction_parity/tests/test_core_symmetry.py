"""Regression test pinning the assumption the package rests on: core chemprop's CGR of a
reversed reaction is a fixed linear map of the forward one (spec section 3.1).

Uses no package code; ``apply_reversal`` exists only to interrogate core.
"""

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer, RxnMode
import pytest
import torch

from .conftest import encode, make_rxn, reverse_rxn_smi


def apply_reversal(
    mg: MolGraph, f: CondensedGraphOfReactionFeaturizer, mode: str | RxnMode
) -> MolGraph:
    mode = RxnMode.get(mode)
    n_atomic = len(f.atom_featurizer.atomic_nums) + 1  # shared head [0:k]
    F = len(f.atom_featurizer)
    Bb = f.bond_fdim // 2

    V, E = mg.V.copy(), mg.E.copy()
    mid, tail = slice(n_atomic, F), slice(F, 2 * F - n_atomic)
    if mode in (RxnMode.REAC_DIFF, RxnMode.REAC_DIFF_BALANCE):
        V[:, mid] += V[:, tail]
        V[:, tail] *= -1
        E[:, :Bb] += E[:, Bb:]
        E[:, Bb:] *= -1
    elif mode in (RxnMode.PROD_DIFF, RxnMode.PROD_DIFF_BALANCE):
        V[:, mid] -= V[:, tail]
        V[:, tail] *= -1
        E[:, :Bb] -= E[:, Bb:]
        E[:, Bb:] *= -1
    else:  # REAC_PROD(_BALANCE): swap the two trailing blocks, head is shared
        V[:, mid], V[:, tail] = V[:, tail].copy(), V[:, mid].copy()
        E[:, :Bb], E[:, Bb:] = E[:, Bb:].copy(), E[:, :Bb].copy()
    return mg._replace(V=V, E=E)


@pytest.mark.parametrize("mode", list(RxnMode.keys()))
def test_core_cgr_reversal_is_mirror(rxn_smi, mode):
    f = CondensedGraphOfReactionFeaturizer(mode_=mode)
    mg_fwd = f(make_rxn(rxn_smi))
    mg_rev = f(make_rxn(reverse_rxn_smi(rxn_smi)))
    assert torch.allclose(encode(apply_reversal(mg_fwd, f, mode)), encode(mg_rev), atol=1e-5)
