"""An MPNN whose predictions are exactly parity-equivariant under reaction reversal."""

from __future__ import annotations

from collections.abc import Sequence

from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.models.model import BatchType
from chemprop.nn import Aggregation, MessagePassing, Predictor
from chemprop.nn.transforms import UnscaleTransform
import torch
from torch import Tensor

from chemprop_contrib.reaction_parity.featurizers import ParityCGRFeaturizer, stack_reversed_bmg
from chemprop_contrib.reaction_parity.parity import (
    FREE,
    ODD,
    build_parity_mask,
    normalize_task_parity,
)


class ParityEquivariantMPNN(MPNN):
    r"""An :class:`~chemprop.models.MPNN` that is exactly parity-equivariant under reversal.

    With ``g`` the full stack (message passing, aggregation, batch norm, predictor) and
    :math:`\bar{x}` the reversed reaction, each task is predicted as

    - ``ODD``:  :math:`\tfrac12[g(x) - g(\bar{x})]`, so :math:`f(\bar{x}) = -f(x)`
    - ``EVEN``: :math:`\tfrac12[g(x) + g(\bar{x})]`, so :math:`f(\bar{x}) = f(x)`
    - ``FREE``: :math:`g(x)`, unconstrained

    Both passes share every weight, and are stacked on the batch dimension before batch norm
    so that in train mode they see identical batch statistics. The guarantee is exact in
    ``eval()``; in train mode, independent dropout masks make it approximate.

    Inputs must be featurized with :class:`ParityCGRFeaturizer`; prefer
    :meth:`from_featurizer`, which reads ``odd_atom_start`` and ``odd_bond_start`` from it.

    Parameters
    ----------
    message_passing, agg, predictor, batch_norm, **kwargs
        as for :class:`~chemprop.models.MPNN`
    odd_atom_start : int
        first node-feature column of the odd (``d``) block
    odd_bond_start : int
        first edge-feature column of the odd (``d``) block
    task_parity : Sequence[int]
        one of ``ODD``, ``EVEN``, ``FREE`` per task

    Raises
    ------
    ValueError
        if ``task_parity`` holds a value other than ``ODD``, ``EVEN`` or ``FREE``; if the
        predictor is not one of the regression predictors supported by
        :func:`~chemprop_contrib.reaction_parity.parity.build_parity_mask`, or ``task_parity``
        is invalid for it; or if the predictor's output transform has a nonzero mean on an
        ``ODD`` task.

    Notes
    -----
    Inputs must come from :class:`ParityCGRFeaturizer`. Chemprop's own CGR featurizer is not in
    the ``(s, d)`` basis; :meth:`from_featurizer` rejects it, but the plain constructor only
    sees dimensions and offsets and cannot tell.

    Extra descriptors ``X_d`` and extra atom descriptors ``V_d`` are passed identically to the
    forward and reversed graphs, so they must be direction-invariant.

    Atom/bond feature scaling (``message_passing.graph_transform``) is allowed: it is applied
    inside the encoder, after the reversal, so it is part of ``g`` and the combination
    stays exact.

    :meth:`fingerprint` returns the forward-direction fingerprint only, so its shape is
    unchanged but it is direction-dependent; the parity structure lives in the combination of
    predictor outputs.
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        odd_atom_start: int,
        odd_bond_start: int,
        task_parity: Sequence[int],
        batch_norm: bool = False,
        **kwargs,
    ):
        # normalised before MPNN.__init__ so that the hyperparameters it collects from this
        # frame hold plain ints
        odd_atom_start = self._as_start_index("odd_atom_start", odd_atom_start)
        odd_bond_start = self._as_start_index("odd_bond_start", odd_bond_start)
        # the upper bounds are checked only when the message passing records its input dims
        # (chemprop's built-in classes do; a custom MessagePassing may not)
        hparams = getattr(message_passing, "hparams", None) or {}
        for name, start, dim_key in (
            ("odd_atom_start", odd_atom_start, "d_v"),
            ("odd_bond_start", odd_bond_start, "d_e"),
        ):
            dim = hparams.get(dim_key)
            if start <= 0 or (dim is not None and start >= dim):
                bound = f"{dim_key} = {dim}" if dim is not None else dim_key
                raise ValueError(f"{name} must satisfy 0 < {name} < {bound}, got {start}")
        task_parity = normalize_task_parity(task_parity)

        mask = build_parity_mask(task_parity, predictor)
        self._check_odd_task_mean(predictor, task_parity)

        super().__init__(message_passing, agg, predictor, batch_norm, **kwargs)

        self.odd_atom_start = odd_atom_start
        self.odd_bond_start = odd_bond_start
        self.task_parity = task_parity
        self.register_buffer("parity_mask", mask, persistent=False)

    @staticmethod
    def _as_start_index(name: str, value) -> int:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)

    @staticmethod
    def _check_odd_task_mean(predictor: Predictor, task_parity: list[int]) -> None:
        transform = predictor.output_transform
        if not isinstance(transform, UnscaleTransform):
            return
        mean = transform.mean.reshape(-1)
        bad = [i for i, p in enumerate(task_parity) if p == ODD and mean[i] != 0]
        if bad:
            raise ValueError(
                f"The predictor's output transform has a nonzero mean on odd task(s) {bad}. "
                "The mean cancels in the odd combination, which would bias those predictions "
                "by -mean. Zero the odd columns of the fitted scaler before normalizing the "
                "datasets and building the transform: `scaler = StandardScaler().fit(Y_train)`; "
                "`scaler.mean_[odd_cols] = 0.0`; optionally (recommended) `scaler.scale_"
                "[odd_cols] = np.sqrt(np.nanmean(Y_train[:, odd_cols] ** 2, axis=0))`, the RMS "
                "about zero; `dset.normalize_targets(scaler)` for the training and validation "
                "datasets; then `UnscaleTransform.from_standard_scaler(scaler)`. Note that "
                "StandardScaler(with_mean=False) does not work: it still sets `mean_`, which "
                "`from_standard_scaler` copies."
            )

    @classmethod
    def from_featurizer(
        cls,
        featurizer: ParityCGRFeaturizer,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        task_parity: Sequence[int],
        **kwargs,
    ) -> ParityEquivariantMPNN:
        """Build a model whose odd-block offsets are read from ``featurizer``.

        Raises
        ------
        ValueError
            if ``message_passing`` records input widths ``d_v``/``d_e`` that differ from the
            featurizer's ``atom_fdim``/``bond_fdim``.
        TypeError
            if ``featurizer`` is not a :class:`ParityCGRFeaturizer`. Chemprop's own CGR
            featurizers are not in the ``(s, d)`` basis, so a model built for them would
            silently lose exactness on real reversed reactions.
        """
        if not isinstance(featurizer, ParityCGRFeaturizer):
            raise TypeError(
                "from_featurizer requires a ParityCGRFeaturizer, got "
                f"{type(featurizer).__name__}. Chemprop's CondensedGraphOfReactionFeaturizer is "
                "not in the (s, d) basis that makes reversal a sign flip."
            )
        hparams = getattr(message_passing, "hparams", None) or {}
        for key, fdim in (("d_v", featurizer.atom_fdim), ("d_e", featurizer.bond_fdim)):
            dim = hparams.get(key)
            if dim is not None and dim != fdim:
                raise ValueError(
                    f"message_passing has {key} = {dim} but the featurizer emits {fdim} "
                    "features; build the message passing from featurizer.atom_fdim and "
                    "featurizer.bond_fdim."
                )
        return cls(
            message_passing,
            agg,
            predictor,
            featurizer.odd_atom_start,
            featurizer.odd_bond_start,
            task_parity,
            **kwargs,
        )

    def _paired_fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Fingerprints of the forward and reversed batch, stacked as ``(2B, d)``.

        Both run through the encoder in one pass over a ``2B``-graph batch, which is faster
        than two passes and computes the same thing.
        """
        both = stack_reversed_bmg(bmg, self.odd_atom_start, self.odd_bond_start)
        V_d = None if V_d is None else torch.cat((V_d, V_d))
        H = self.bn(self.agg(self.message_passing(both, V_d), both.batch))

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d).repeat(2, 1)), dim=1)

    def _combine(self, preds: Tensor) -> Tensor:
        """Combine ``(2B, ...)`` stacked predictions into ``(B, ...)`` using the parity mask."""
        B = preds.shape[0] // 2
        fwd, rev = preds[:B], preds[B:]
        m = self.parity_mask
        return torch.where(m == FREE, fwd, 0.5 * (fwd + m * rev))

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate parity-equivariant predictions for the input reactions"""
        return self._combine(self.predictor(self._paired_fingerprint(bmg, V_d, X_d)))

    def _train_preds(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Combined ``predictor.train_step`` output, as the training objective sees it."""
        return self._combine(self.predictor.train_step(self._paired_fingerprint(bmg, V_d, X_d)))

    # The parent's training/validation steps call `fingerprint` and `predictor.train_step`
    # directly, bypassing `forward`; these mirror them with only the preds computation changed.
    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self._train_preds(bmg, V_d, X_d)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)

        return l

    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self._train_preds(bmg, V_d, X_d)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True)
