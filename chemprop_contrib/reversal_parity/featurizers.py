r"""CGR featurizer that emits features in the (mean, difference) basis.

With :math:`x_r` and :math:`x_p` the reactant- and product-side feature vectors, this
featurizer emits :math:`(m, d)` with :math:`m = \tfrac12(x_r + x_p)` and
:math:`d = x_p - x_r`, so that reversing a reaction is a sign flip on the trailing :math:`d`
column block and leaves :math:`m` untouched.

Every atom feature, including the element, gets both parts, so mapped atoms may change element.
This admits alchemical transformations ``A>>B`` (e.g. phenol to toluene), with the atom map
alone defining the topology: a mapped element change is single topology, unmapped atoms on
each side are dual topology, and a mix is hybrid. Chemical reactions are the special case in
which no mapped atom changes element.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import Iterable

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph.reaction import (
    CondensedGraphOfReactionFeaturizer,
    RxnMode,
)
import numpy as np
from rdkit.Chem.rdchem import Bond, Mol
import torch


@dataclass
class ParityCGRFeaturizer(CondensedGraphOfReactionFeaturizer):
    r"""Featurize reactions as a CGR in the :math:`(m, d)` basis.

    Parameters
    ----------
    balanced : bool, default=False
        whether to use the ``*_BALANCE`` bond behavior of the parent featurizer
        (``REAC_DIFF_BALANCE`` instead of ``REAC_DIFF``)
    mode_ : None
        not configurable; this featurizer defines its own layout. Passing a value raises
        :class:`ValueError`.

    Notes
    -----
    Node layout (``atom_fdim = 2F``): ``[0:F]`` is the even block :math:`m`, ``[F:2F]`` the
    odd block :math:`d`. Edge layout (``bond_fdim = 2B``): ``[0:B]`` is the even block
    :math:`m`, ``[B:2B]`` the odd block :math:`d`.

    An atom absent from one side is a zero dummy atom there (``atom_featurizer(None)``), so a
    vanishing atom has :math:`d = -x_r`. With ``balanced=True`` it is instead copied across
    unchanged (:math:`d = 0`), Chemprop's ``*_BALANCE`` meaning of an unrecorded spectator. Unlike
    Chemprop's ``REAC_DIFF``, the element is not written once: Chemprop assumes a mapped atom
    keeps its element, which this featurizer does not.
    """

    # keyword-only, so that a third positional argument cannot land in ``mode_``
    _: KW_ONLY
    balanced: bool = False
    mode_: InitVar[str | RxnMode | None] = None

    def __post_init__(self, mode_: str | RxnMode | None):
        if mode_ is not None:
            raise ValueError(
                "ParityCGRFeaturizer defines its own feature layout and does not accept "
                f"'mode_' (got {mode_!r}); use 'balanced' to select the balance behavior."
            )
        super().__post_init__(RxnMode.REAC_DIFF_BALANCE if self.balanced else RxnMode.REAC_DIFF)
        self.atom_fdim = 2 * len(self.atom_featurizer)

    @property
    def odd_atom_start(self) -> int:
        """First column of the odd (:math:`d`) block of the node features."""
        return len(self.atom_featurizer)

    @property
    def odd_bond_start(self) -> int:
        """First column of the odd (:math:`d`) block of the edge features."""
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
    r"""Reverse every reaction in a batch by negating the odd feature blocks.

    In the :math:`(m, d)` basis reversal is :math:`d \to -d` with :math:`m` unchanged, and the
    graph topology is identical, so ``edge_index``, ``rev_edge_index`` and ``batch`` are
    shared with the input. The input is not modified.

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


# the BatchMolGraph fields stack_reversed_bmg knows how to stack; the size is name-mangled
_STACKED_FIELDS = {"V", "E", "edge_index", "rev_edge_index", "batch", "_BatchMolGraph__size"}


def stack_reversed_bmg(
    bmg: BatchMolGraph, odd_atom_start: int, odd_bond_start: int
) -> BatchMolGraph:
    """Stack a batch and its reversal into one batch of ``2B`` graphs.

    Graphs ``0..B-1`` are the input and graphs ``B..2B-1`` their reversals, in the same order,
    so one encoder pass computes what :func:`reverse_bmg` and two passes would. The result
    equals chemprop's collate of the forward graphs followed by the reversed ones. The input
    is not modified.

    Parameters
    ----------
    bmg : BatchMolGraph
        batch featurized with :class:`ParityCGRFeaturizer`
    odd_atom_start : int
        first node-feature column of the odd block
    odd_bond_start : int
        first edge-feature column of the odd block

    Raises
    ------
    TypeError
        if the batch carries fields or attributes other than chemprop's ``BatchMolGraph``
        ones, which this function would otherwise silently leave out of the result
    """
    names = {f.name for f in dataclasses.fields(bmg)} | set(getattr(bmg, "__dict__", {}))
    if unknown := sorted(names - _STACKED_FIELDS):
        raise TypeError(
            f"cannot stack a {type(bmg).__name__} with fields {unknown}; only the fields of "
            "chemprop's BatchMolGraph (V, E, edge_index, rev_edge_index, batch) are supported."
        )
    rev = reverse_bmg(bmg, odd_atom_start, odd_bond_start)
    n_atoms, n_edges = bmg.V.shape[0], bmg.E.shape[0]

    out = copy.copy(bmg)
    out.V = torch.cat((bmg.V, rev.V))
    out.E = torch.cat((bmg.E, rev.E))
    out.edge_index = torch.cat((bmg.edge_index, bmg.edge_index + n_atoms), dim=1)
    out.rev_edge_index = torch.cat((bmg.rev_edge_index, bmg.rev_edge_index + n_edges))
    out.batch = torch.cat((bmg.batch, bmg.batch + len(bmg)))
    out._BatchMolGraph__size = 2 * len(bmg)
    return out
