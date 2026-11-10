"""An MPNN whose predictions are exactly parity-equivariant under reaction reversal."""

from __future__ import annotations

from collections.abc import Sequence

from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.models.model import BatchType
from chemprop.nn import Aggregation, MessagePassing, Predictor
from chemprop.nn.transforms import UnscaleTransform
import torch
from torch import Tensor, nn

from chemprop_contrib.reaction_parity.featurizers import ParityCGRFeaturizer, reverse_bmg
from chemprop_contrib.reaction_parity.parity import FREE, ODD, build_parity_mask


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
        if ``message_passing.graph_transform`` is not :class:`~torch.nn.Identity`; if the
        predictor is not one of the regression predictors supported by
        :func:`~chemprop_contrib.reaction_parity.parity.build_parity_mask`, or ``task_parity``
        is invalid for it; or if the predictor's output transform has a nonzero mean on an
        ``ODD`` task.

    Notes
    -----
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
        odd_atom_start = int(odd_atom_start)
        odd_bond_start = int(odd_bond_start)
        task_parity = [int(p) for p in task_parity]

        if not isinstance(message_passing.graph_transform, nn.Identity):
            raise ValueError(
                "ParityEquivariantMPNN requires message_passing.graph_transform to be "
                f"nn.Identity, got {type(message_passing.graph_transform).__name__}. Scaling "
                "atom/bond features before the encoder breaks parity equivariance; disable "
                "atom and bond feature scaling."
            )
        mask = build_parity_mask(task_parity, predictor)
        self._check_odd_task_mean(predictor, task_parity)

        super().__init__(message_passing, agg, predictor, batch_norm, **kwargs)

        self.odd_atom_start = odd_atom_start
        self.odd_bond_start = odd_bond_start
        self.task_parity = task_parity
        self.register_buffer("parity_mask", mask, persistent=False)

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
                "`scaler.mean_[odd_cols] = 0.0`; `dset.normalize_targets(scaler)` for the "
                "training and validation datasets; then "
                "`UnscaleTransform.from_standard_scaler(scaler)`. Note that "
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
        """Build a model whose odd-block offsets are read from ``featurizer``."""
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
        """Fingerprints of the forward and reversed batch, stacked as ``(2B, d)``."""
        bmg_rev = reverse_bmg(bmg, self.odd_atom_start, self.odd_bond_start)
        H_f = self.agg(self.message_passing(bmg, V_d), bmg.batch)
        H_r = self.agg(self.message_passing(bmg_rev, V_d), bmg.batch)
        H = self.bn(torch.cat([H_f, H_r], dim=0))

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
