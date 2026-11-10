# `reaction_parity`

This `chemprop-contrib` package makes reaction-property predictions **exactly antisymmetric (or symmetric) under reaction reversal**, using the Condensed Graph of Reaction (CGR).

Thermochemical quantities such as the reaction enthalpy, free energy or energy satisfy

```
ΔH(A→B) = −ΔH(B→A)
```

A standard Chemprop reaction model satisfies this only by accident, and nothing in the loss penalizes a violation. Other reaction properties are exactly symmetric under reversal, and others (activation energy) are neither. The requirement is therefore a *per-task parity*, which this package enforces by construction rather than by data augmentation.

| Parity | Constant | Behavior under reversal | Example target |
|---|---|---|---|
| odd | `ODD` (−1) | `f(reversed) = −f(x)` | ΔH, ΔG, ΔE |
| even | `EVEN` (+1) | `f(reversed) = +f(x)` | reorganization energy |
| free | `FREE` (0) | unconstrained | activation energy `Ea` |

## How it works

1. `ParityCGRFeaturizer` emits CGR features in an `(s, d)` basis, with `s` the mean of the reactant-side and product-side features and `d` their difference. In this basis, reversing a reaction is a sign flip on the contiguous `d` block, so no second featurization is needed. It is an invertible linear map of Chemprop's `REAC_DIFF` features, so no information is lost.
2. `ParityEquivariantMPNN` runs the same network (shared weights) on the batch and on its reversal (`reverse_bmg`), then combines the outputs per task: `½[g(x) − g(x̄)]` for odd, `½[g(x) + g(x̄)]` for even, `g(x)` for free. The guarantee holds for any network `g`.

The cost is two forward passes per batch.

## Minimal example

```python
import csv

import lightning.pytorch as pl
from chemprop.data import ReactionDatapoint, ReactionDataset, build_dataloader
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.nn.transforms import UnscaleTransform
import numpy as np
from sklearn.preprocessing import StandardScaler

from chemprop_contrib.reaction_parity import (
    EVEN, FREE, ODD, ParityCGRFeaturizer, ParityEquivariantMPNN,
)

# columns: dH (odd), reorganization energy (even), Ea (free)
task_parity = [ODD, EVEN, FREE]
cols = ["dH", "lambda", "Ea"]
with open("reactions.csv") as f:  # atom-mapped reaction SMILES in "smiles"
    rows = list(csv.DictReader(f))

featurizer = ParityCGRFeaturizer()  # not chemprop's CondensedGraphOfReactionFeaturizer
dps = [
    ReactionDatapoint.from_smi(
        r["smiles"], y=[float(r[c]) for c in cols], keep_h=True, add_h=False
    )
    for r in rows
]
n_train = int(0.8 * len(dps))
train_dset = ReactionDataset(dps[:n_train], featurizer=featurizer)
val_dset = ReactionDataset(dps[n_train:], featurizer=featurizer)

# odd columns need a zero-mean scaler (see below)
scaler = StandardScaler().fit(train_dset.Y)
odd_cols = [i for i, p in enumerate(task_parity) if p == ODD]
scaler.mean_[odd_cols] = 0.0
scaler.scale_[odd_cols] = np.sqrt(np.nanmean(train_dset.Y[:, odd_cols] ** 2, axis=0))
train_dset.normalize_targets(scaler)
val_dset.normalize_targets(scaler)
output_transform = UnscaleTransform.from_standard_scaler(scaler)

mp = BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim)
predictor = RegressionFFN(
    n_tasks=len(task_parity), input_dim=mp.output_dim, output_transform=output_transform
)
model = ParityEquivariantMPNN.from_featurizer(
    featurizer, mp, MeanAggregation(), predictor, task_parity, batch_norm=True
)

train_loader = build_dataloader(train_dset, batch_size=8, num_workers=0)
val_loader = build_dataloader(val_dset, batch_size=8, num_workers=0, shuffle=False)
trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False)
trainer.fit(model, train_loader, val_loader)

preds = trainer.predict(model, val_loader)  # list of (batch, n_tasks) tensors
```

Reactions must be atom-mapped and built with `keep_h=True`, and the datasets must use `ParityCGRFeaturizer`. `tests/test_models.py` contains a complete training setup on a bundled fixture.

## Things to know

**Use `ParityCGRFeaturizer`, not Chemprop's CGR featurizer.** Chemprop's `CondensedGraphOfReactionFeaturizer` produces features of the same size, so a model given such data cannot tell. Its features are not in the `(s, d)` basis, though, and predictions on real reversed reactions are then only approximately odd or even. `ParityEquivariantMPNN.from_featurizer` raises `TypeError` for any other featurizer; the plain constructor cannot check this.

**Odd-task targets need a zero-mean scaler.** The output transform unscales as `X·scale + mean`, and the mean cancels in the odd combination, so a nonzero mean on an odd column silently biases every odd prediction by `−mean`. Fit the scaler, zero its mean on the odd columns, and only then normalize the datasets and build the transform, as in the example above. The example also resets the odd columns' scale to their root mean square `sqrt(mean(y²))`. After zeroing the mean, `scale_` is still the standard deviation about the old mean, so for data that mostly runs in one direction (`|mean| ≫ std`) the scaled odd targets would be far from unit size. Predictions are unbiased either way; the RMS scale only keeps the training targets well-conditioned. `StandardScaler(with_mean=False)` does **not** work: it still sets `mean_`, which `UnscaleTransform.from_standard_scaler` copies. The model checks this at construction and raises a `ValueError` with this recipe. Even and free tasks keep their means.

**Exactness is for `eval()` only.** In train mode, batch-norm batch statistics and independent dropout masks across the two passes make the construction approximate. In `eval()` the guarantee is exact (to float precision). Call `model.eval()` before checking parity yourself; `trainer.predict` does so for you. Call `model.train()` again before any further `trainer.fit`: Lightning trains a module in the mode it was left in, and in eval mode the output transform is active, so the model would fit unscaled predictions to scaled targets and fail to learn.

**Extra descriptors must be direction-invariant.** `X_d` (and per-atom `V_d`, which `ReactionDataset` does not currently produce) is passed identically to the forward and reversed passes, so it is treated as invariant under reversal. Reaction conditions such as temperature or solvent are fine. Direction-dependent descriptors (for example reactant-side-only descriptors) break exactness unless you symmetrise them or fold them into the graph.

**Atom and bond feature scaling is fine.** A `graph_transform` on the message passing runs inside the encoder, after the reversal, so it is just part of the network `g` and exactness is unaffected.

**`fingerprint` is direction-dependent.** `model.fingerprint(...)` returns the forward-direction fingerprint only, with the usual shape. The parity structure lives in the combination of predictor outputs, not in the fingerprint.

**Loading saved models.** Use `ParityEquivariantMPNN.load_from_file` or `ParityEquivariantMPNN.load_from_checkpoint`. Chemprop's generic `MPNN.load_from_file`, `chemprop.models.utils.load_model` and the `chemprop predict` and `chemprop fingerprint` CLIs cannot rebuild this class (they raise a `TypeError` about `odd_atom_start`).

## Unsupported combinations

Construction raises `ValueError` for:

- predictors outside `RegressionFFN`, `MveFFN`, `EvidentialFFN` and `QuantileFFN`. Variance, evidential and quantile-interval components are combined as even, so they stay valid;
- a `task_parity` entry other than `ODD`, `EVEN` or `FREE` (floats and booleans are rejected, not truncated);
- a nonzero output-transform mean on an odd task.

Reaction-plus-molecule inputs (for example a reaction with a solvent, via `MulticomponentMPNN`) are not supported: `ParityEquivariantMPNN` takes a single reaction graph.

`CuikmolmakerCGRFeaturizer` is not supported, since it cannot express the `(s, d)` layout. There is no CLI integration; use the Python API.

## Future work

Activation energies satisfy `Ea_rev = Ea_fwd − ΔH`. Predicting an even head `B` and an odd head `ΔH` and forming `Ea_fwd = B + ΔH/2`, `Ea_rev = B − ΔH/2` satisfies this relation identically. This coupled head is not implemented yet.

## Tests

```
pytest chemprop_contrib/reaction_parity/
```

The bundled fixture `tests/test_data.csv` is derived from Chemprop's `rxn.csv`; see `tests/data_provenance.txt`.
