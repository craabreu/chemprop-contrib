# `reversal_parity`

This `chemprop-contrib` package makes reaction-property predictions **exactly antisymmetric (or symmetric) under reaction reversal**, using the Condensed Graph of Reaction (CGR). The same holds for alchemical transformations between two molecules, such as phenol → toluene, of which chemical reactions are a special case.

Thermochemical quantities such as the reaction enthalpy, free energy or energy satisfy

```
ΔH(A→B) = −ΔH(B→A)
```

A standard Chemprop reaction model satisfies this only by accident, and nothing in the loss penalizes a violation. Other reaction properties are exactly symmetric under reversal, and others are neither. The requirement is therefore a *per-column parity*, which this package enforces by construction rather than by data augmentation. You declare it with `ParityTargetTransform`, which also scales the targets:

| Declaration | Behavior under reversal | Example target |
|---|---|---|
| `odd=[i]` | `f(x̄) = −f(x)` | ΔH, ΔG, ΔE, ΔΔG_solv |
| `even=[i]` | `f(x̄) = +f(x)` | reorganization energy |
| `paired_reverse={i: j}` | swaps columns `i` and `j` | forward and reverse barriers |
| `paired_difference={i: j}` | column `i` becomes `i − j`; column `j` is odd | forward barrier and ΔE |
| not listed | unconstrained (free) | any other target |

## How it works

1. `ParityCGRFeaturizer` emits CGR features in an `(s, d)` basis, with `s` the mean of the reactant-side and product-side features and `d` their difference. In this basis, reversing a reaction is a sign flip on the contiguous `d` block, so no second featurization is needed. Every atom feature gets both parts, including the element, so a mapped atom may change element (see [Alchemical transformations](#alchemical-transformations)).
2. `ParityEquivariantMPNN` runs the same network (shared weights) on the batch and on its reversal, then combines the outputs per column: `½[g(x) − g(x̄)]` for odd, `½[g(x) + g(x̄)]` for even, `g(x)` for free. A pair is predicted from an even mean and an odd difference (see [Pairs](#pairs-barriers-and-other-two-direction-properties)). The guarantee holds for any network `g`.

Both directions go through the encoder in a single pass over a batch of twice as many graphs (`stack_reversed_bmg`). On RDB7 (small gas-phase reactions, batch size 64, one RTX 2080 SUPER), training takes about 15% longer than stock chemprop.

## Minimal example

```python
import csv

import lightning.pytorch as pl
from chemprop.data import ReactionDatapoint, ReactionDataset, build_dataloader
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

from chemprop_contrib.reversal_parity import (
    ParityCGRFeaturizer, ParityEquivariantMPNN, ParityTargetTransform,
)

# columns: forward barrier Ea, reaction energy dE (Ea - dE is the reverse barrier),
# reorganization energy (even)
cols = ["Ea", "dE", "lambda"]
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

# declare each column's parity, fit the target scaling, normalize both datasets with it
target = ParityTargetTransform.fit(train_dset.Y, paired_difference={0: 1}, even=[2])
train_dset.normalize_targets(target.to_standard_scaler())
val_dset.normalize_targets(target.to_standard_scaler())

mp = BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim)
predictor = RegressionFFN(n_tasks=len(cols), input_dim=mp.output_dim)  # no output_transform
model = ParityEquivariantMPNN.from_featurizer(
    featurizer, mp, MeanAggregation(), predictor, target, batch_norm=True
)

train_loader = build_dataloader(train_dset, batch_size=8, num_workers=0)
val_loader = build_dataloader(val_dset, batch_size=8, num_workers=0, shuffle=False)
trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False)
trainer.fit(model, train_loader, val_loader)

preds = trainer.predict(model, val_loader)  # list of (batch, n_tasks) tensors, unscaled
```

Reactions must be atom-mapped and built with `keep_h=True`, and the datasets must use `ParityCGRFeaturizer`. `tests/test_models.py` contains a complete training setup on a bundled fixture.

## Pairs: barriers and other two-direction properties

Reversal swaps a forward barrier with its reverse barrier: `Ea_r(x) = Ea_f(x̄)`, and their difference is the reaction energy. Declare such a property by mapping its forward column to whichever of the reverse column or the difference column your data has:

```python
paired_reverse={0: 1}      # forward and reverse barriers in columns 0 and 1
paired_difference={0: 1}   # forward barrier and reaction energy in columns 0 and 1
```

The model predicts the pair from an even mean `B` and an odd difference `D`, as

```
forward = B + ½ D      reverse = B − ½ D      difference = D
```

so forward − reverse = difference holds exactly, and predicting on the reversed reaction gives the reverse barrier as its forward value. Predictions come back as the columns of your dataset.

Training scores all three members of a pair. A row with both of its labels also scores the third member, computed from them, and a row with one label scores that one. Both forms therefore train the same model on the same data, and neither direction is favoured. Rows with a forward barrier alone, or with a reaction energy alone with `paired_difference`, still train. `fit` scales the two directions together, from both pooled, and the difference about zero.

The same structure fits any property that reversal maps to its partner: rate constants (`log k` forward and reverse, with `log K_eq` as the difference), free-energy barriers, or, for an alchemical pair `A>>B`, an endpoint property such as the solvation free energy of each end state, whose difference is the relative `ΔΔG_solv`. The sign convention is difference = forward − reverse.

Pairs need a plain `RegressionFFN`: uncertainty is not supported for them yet.

## Alchemical transformations

The input `A>>B` does not have to be a chemical reaction. Any atom-mapped pair of molecules works, including alchemical transformations in which mapped atoms change element. An odd task then predicts a relative property, such as the relative solvation free energy `ΔΔG_solv(A→B) = ΔG_solv(B) − ΔG_solv(A)`, which is exactly antisymmetric under swapping `A` and `B`.

The atom map alone defines the topology of the transformation, as in alchemical free-energy methods. For phenol → toluene (hydrogens as counts):

| Topology | `A>>B` | What the CGR contains |
|---|---|---|
| single | `[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH3:7]` | one node that turns from O into C in place |
| dual | `[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH3]` | an O node that vanishes and a C node that appears |
| hybrid | any mix, e.g. O mapped to C with explicit hydrogens partly mapped | both |

A chemical reaction is the special case in which no mapped atom changes element. Nothing checks atom maps for chemical plausibility, so a map that pairs atoms of different elements by mistake is featurized as an element change.

On the side where an atom is absent it is a zero dummy atom: Chemprop's atom featurizer has no null bit, and an all-zero vector (`atom_featurizer(None)`) is unambiguous, since every real atom has exactly one element bit set. A vanishing atom therefore has `d = −x_A` and an appearing atom `d = +x_B`, and the net element change summed over the graph is the same in every topology. With `ParityCGRFeaturizer(balanced=True)`, an atom absent from one side is instead copied across unchanged (Chemprop's meaning of an unrecorded spectator in an incomplete reaction).

## Things to know

**Use `ParityCGRFeaturizer`, not Chemprop's CGR featurizer.** Chemprop's `CondensedGraphOfReactionFeaturizer` is not in the `(s, d)` basis, and a model built for it gives predictions on real reversed reactions that are only approximately odd or even. `ParityEquivariantMPNN.from_featurizer` raises `TypeError` for any other featurizer; the plain constructor only sees dimensions and offsets and cannot check this.

**Scale targets with `ParityTargetTransform`, not the predictor.** The model combines the forward and reversed predictions in the scaled space and then unscales, so the predictor must not have its own `output_transform`, and the datasets must be normalized with `target.to_standard_scaler()`. Odd columns are scaled about zero (mean 0, root-mean-square scale): a shift is even, so it would break their parity. `fit` handles this, ignores missing values, and leaves even and free columns with their usual mean and standard deviation. The code cannot check that the datasets were normalized with the matching scaler, so build both from the same transform, as in the example. The model keeps its own copy of the transform, so one transform can serve several models, such as the members of an ensemble.

**Exactness is for `eval()` only.** In train mode, batch-norm batch statistics and independent dropout masks for the forward and reversed copies make the construction approximate. In `eval()` the guarantee is exact (to float precision). Call `model.eval()` before checking parity yourself; `trainer.predict` does so for you. Call `model.train()` again before any further `trainer.fit`: Lightning trains a module in the mode it was left in, and in eval mode the target transform unscales, so the model would fit unscaled predictions to scaled targets and fail to learn.

**Extra descriptors must be direction-invariant.** `X_d` (and per-atom `V_d`, which `ReactionDataset` does not currently produce) is passed identically to the forward and reversed passes, so it is treated as invariant under reversal. Reaction conditions such as temperature or solvent are fine. Direction-dependent descriptors (for example reactant-side-only descriptors) break exactness unless you symmetrise them or fold them into the graph.

**Atom and bond feature scaling is fine.** A `graph_transform` on the message passing runs inside the encoder, after the reversal, so it is just part of the network `g` and exactness is unaffected.

**`fingerprint` is direction-dependent.** `model.fingerprint(...)` returns the forward-direction fingerprint only, with the usual shape. The parity structure lives in the combination of predictor outputs, not in the fingerprint.

**Loading saved models.** Use `ParityEquivariantMPNN.load_from_file` or `ParityEquivariantMPNN.load_from_checkpoint`. Chemprop's generic `MPNN.load_from_file`, `chemprop.models.utils.load_model` and the `chemprop predict` and `chemprop fingerprint` CLIs cannot rebuild this class (they raise a `TypeError` about `odd_atom_start`). **Never use `MPNN.load_from_checkpoint`:** Lightning silently drops the parity hyperparameters and returns a plain `MPNN`, whose predictions differ and have no parity guarantee.

## Unsupported combinations

Construction raises `ValueError` for:

- predictors outside `RegressionFFN`, `MveFFN`, `EvidentialFFN` and `QuantileFFN`. Variance, evidential and quantile-interval components are combined as even, so they stay valid;
- a predictor with its own `output_transform`, or a different number of tasks than the target transform;
- pairs with `MveFFN`, `EvidentialFFN` or `QuantileFFN`: uncertainty is not supported for pairs yet.

`ParityTargetTransform` raises `ValueError` for indices out of range (floats and booleans are rejected, not truncated), a column given two roles (including a forward column in both `paired_reverse` and `paired_difference`), and a scaling that breaks the declaration (a nonzero mean on an odd or difference column, or different scalings for the two directions of a pair).

Training and prediction raise `TypeError` for a batch class that carries fields or attributes beyond those of chemprop's `BatchMolGraph`, since the stacked batch would silently leave them out.

Reaction-plus-molecule inputs (for example a reaction with a solvent, via `MulticomponentMPNN`) are not supported: `ParityEquivariantMPNN` takes a single reaction graph.

`CuikmolmakerCGRFeaturizer` is not supported, since it cannot express the `(s, d)` layout. There is no CLI integration; use the Python API.

## Future work

Extra descriptors that flip sign under reversal, such as a reaction enthalpy used as an input, are not supported yet.

## Tests

```
pytest chemprop_contrib/reversal_parity/
```

The bundled fixture `tests/test_data.csv` is derived from Chemprop's `rxn.csv`; see `tests/data_provenance.txt`.
