"""CGR featurizer that emits features in the (sum, diff) basis.

With ``x_r`` / ``x_p`` the reactant- and product-side feature vectors, this featurizer emits
``(s, d)`` with ``s = (x_r + x_p) / 2`` and ``d = x_p - x_r``, so that reversing a reaction is a
sign flip on the trailing ``d`` column block and leaves ``s`` untouched.

Every atom feature, including the element, gets both parts, so mapped atoms may change element.
This admits alchemical transformations ``A>>B`` (e.g. phenol to toluene), with the atom map
alone defining the topology: a mapped element change is single topology, unmapped atoms on
each side are dual topology, and a mix is hybrid. Chemical reactions are the special case in
which no mapped atom changes element.
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
    Node layout (``atom_fdim = 2F``): ``[0:F]`` even block ``s``, ``[F:2F]`` odd block ``d``.
    Edge layout (``bond_fdim = 2B``): ``[0:B]`` even block ``s``, ``[B:2B]`` odd block ``d``.

    An atom absent from one side is a zero dummy atom there (``atom_featurizer(None)``), so a
    vanishing atom has ``d = -x_r``. With ``balanced=True`` it is instead copied across
    unchanged (``d = 0``), Chemprop's ``*_BALANCE`` meaning of an unrecorded spectator. Unlike
    Chemprop's ``REAC_DIFF``, the element is not written once: Chemprop assumes a mapped atom
    keeps its element, which this featurizer does not.
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
        self.atom_fdim = 2 * len(self.atom_featurizer)

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
        # Built here rather than from the parent's output, which writes the element once
        # (from whichever side is present) and so cannot represent an element change.
        f = self.atom_featurizer
        reac_only = set(reac_idxs)
        X_r, X_p = [], []
        for a in rct.GetAtoms():
            x_r = f(a)
            if a.GetIdx() in reac_only:
                x_p = x_r if self.balanced else f(None)
            else:
                x_p = f(pdt.GetAtomWithIdx(r2p_idx_map[a.GetIdx()]))
            X_r.append(x_r)
            X_p.append(x_p)
        for j in pdt_idxs:
            x_p = f(pdt.GetAtomWithIdx(j))
            X_r.append(x_p if self.balanced else f(None))
            X_p.append(x_p)

        X_r = np.array(X_r, dtype=float).reshape(-1, len(f))
        X_p = np.array(X_p, dtype=float).reshape(-1, len(f))
        return np.hstack((0.5 * (X_r + X_p), X_p - X_r))

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
