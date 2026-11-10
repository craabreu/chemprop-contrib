"""Per-task parity masks for parity-equivariant reaction models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from chemprop.nn.predictors import EvidentialFFN, MveFFN, Predictor, QuantileFFN, RegressionFFN

ODD = -1
EVEN = 1
FREE = 0

_VALID_PARITIES = (ODD, EVEN, FREE)

# Exact types only: MveFFN, EvidentialFFN and QuantileFFN subclass RegressionFFN, and any other
# subclass (e.g. SpectralFFN, whose output is a normalised positive spectrum) must be rejected
# rather than admitted silently.
_ALLOWED_PREDICTORS = frozenset({RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN})


def build_parity_mask(task_parity: Sequence[int], predictor: Predictor) -> Tensor:
    """Build the mask applied to predictor outputs when combining forward and reversed passes.

    Component 0 of each task takes that task's parity; all later components (variance,
    evidential parameters, interval upper bound) are ``EVEN``, since an odd combination of a
    variance can be negative.

    Returns
    -------
    Tensor
        Float tensor of shape ``(n_tasks,)`` if ``predictor.n_targets == 1``, else
        ``(n_tasks, n_targets)``.
    """
    if type(predictor) not in _ALLOWED_PREDICTORS:
        allowed = ", ".join(sorted(c.__name__ for c in _ALLOWED_PREDICTORS))
        raise ValueError(
            f"Parity masks are only defined for {allowed}; got {type(predictor).__name__}."
        )
    task_parity = list(task_parity)
    if len(task_parity) != predictor.n_tasks:
        raise ValueError(
            f"task_parity has length {len(task_parity)} but predictor has "
            f"n_tasks={predictor.n_tasks}."
        )
    bad = [p for p in task_parity if p not in _VALID_PARITIES]
    if bad:
        raise ValueError(f"Invalid parity values {bad}; each must be one of ODD, EVEN, FREE.")

    mask = torch.tensor(task_parity, dtype=torch.get_default_dtype())
    if predictor.n_targets == 1:
        return mask
    full = torch.ones(predictor.n_tasks, predictor.n_targets, dtype=mask.dtype)
    full[:, 0] = mask
    return full
