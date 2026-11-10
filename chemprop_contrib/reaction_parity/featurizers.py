"""CGR featurizer that emits features in the (sum, diff) basis.

With ``x_r`` / ``x_p`` the reactant- and product-side feature vectors and ``d = x_p - x_r``,
the stock ``REAC_DIFF`` mode emits ``(x_r, d)``. This featurizer emits ``(s, d)`` with
``s = (x_r + x_p) / 2 = x_r + d / 2``, so that reversing a reaction is a sign flip on the
trailing ``d`` column block and leaves ``s`` untouched.
"""

from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass
from typing import Iterable

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph.reaction import (
    CondensedGraphOfReactionFeaturizer,
    RxnMode,
)
import numpy as np
from rdkit.Chem.rdchem import Bond, Mol


@dataclass
class ParityCGRFeaturizer(CondensedGraphOfReactionFeaturizer):
    """Featurize reactions as a CGR in the ``(s, d)`` basis.

    Parameters
    ----------
    balanced : bool, default=False
        whether to use the ``*_BALANCE`` bond behaviour of the parent featurizer
        (``REAC_DIFF_BALANCE`` instead of ``REAC_DIFF``)
    mode_ : None
        not configurable; this featurizer defines its own layout. Passing a value raises
        :class:`ValueError`.

    Notes
    -----
    Node layout (``atom_fdim = 2F - k``): ``[0:k]`` atomic-number block (shared),
    ``[k:F]`` even block ``s``, ``[F:2F-k]`` odd block ``d``. Edge layout
    (``bond_fdim = 2B``): ``[0:B]`` even block ``s``, ``[B:2B]`` odd block ``d``.
    """

    balanced: bool = False
    mode_: InitVar[str | RxnMode | None] = None

    def __post_init__(self, mode_: str | RxnMode | None):
        if mode_ is not None:
            raise ValueError(
                "ParityCGRFeaturizer defines its own feature layout and does not accept "
                f"'mode_' (got {mode_!r}); use 'balanced' to select the balance behaviour."
            )
        super().__post_init__(RxnMode.REAC_DIFF_BALANCE if self.balanced else RxnMode.REAC_DIFF)

    @property
    def odd_atom_start(self) -> int:
        """First column of the odd (``d``) block of the node features."""
        return len(self.atom_featurizer)

    @property
    def odd_bond_start(self) -> int:
        """First column of the odd (``d``) block of the edge features."""
        return len(self.bond_featurizer)

    def _calc_node_feature_matrix(
        self,
        rct: Mol,
        pdt: Mol,
        r2p_idx_map: dict[int, int],
        pdt_idxs: Iterable[int],
        reac_idxs: Iterable[int],
    ) -> np.ndarray:
        # The atomic-number block is emitted once, which is only valid if the atom map is
        # element-consistent. The parent's output never exposes x_p[:k], so check here.
        reac_idxs = set(reac_idxs)
        for a in rct.GetAtoms():
            i = a.GetIdx()
            if i in reac_idxs:
                continue
            b = pdt.GetAtomWithIdx(r2p_idx_map[i])
            if a.GetAtomicNum() != b.GetAtomicNum():
                raise ValueError(
                    f"atom map {a.GetAtomMapNum()} is element-inconsistent: reactant atom is "
                    f"{a.GetSymbol()} but product atom is {b.GetSymbol()}"
                )

        X = super()._calc_node_feature_matrix(rct, pdt, r2p_idx_map, pdt_idxs, reac_idxs)
        k = len(self.atom_featurizer.atomic_nums) + 1
        F = self.odd_atom_start
        X[:, k:F] += 0.5 * X[:, F:]
        return X

    def _calc_edge_feature(self, b_reac: Bond | None, b_pdt: Bond | None) -> np.ndarray:
        # the parent's bond features are integer-valued; the half-sum needs a float dtype
        x_e = super()._calc_edge_feature(b_reac, b_pdt).astype(float)
        Bb = self.odd_bond_start
        x_e[:Bb] += 0.5 * x_e[Bb:]
        return x_e


def reverse_bmg(bmg: BatchMolGraph, odd_atom_start: int, odd_bond_start: int) -> BatchMolGraph:
    """Reverse every reaction in a batch by negating the odd feature blocks.

    In the ``(s, d)`` basis reversal is ``d -> -d`` with ``s`` unchanged, and the graph
    topology is identical, so ``edge_index``, ``rev_edge_index`` and ``batch`` are shared
    with the input. The input is not modified.

    Parameters
    ----------
    bmg : BatchMolGraph
        batch featurized with :class:`ParityCGRFeaturizer`
    odd_atom_start : int
        first node-feature column of the odd block
    odd_bond_start : int
        first edge-feature column of the odd block
    """
    V = bmg.V.clone()
    E = bmg.E.clone()
    V[:, odd_atom_start:] *= -1
    E[:, odd_bond_start:] *= -1

    out = copy.copy(bmg)
    out.V = V
    out.E = E
    return out
