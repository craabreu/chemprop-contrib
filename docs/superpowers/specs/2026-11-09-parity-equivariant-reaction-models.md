# Spec: Parity-Equivariant Reaction Property Prediction

**Package:** `chemprop_contrib/reaction_parity`
**Date:** 2026-11-09 (revised 2026-11-10 after independent review)
**Target:** `chemprop/chemprop-contrib`, against chemprop `v2.3.1` (verified at `1b90e3e2`)

---

## 1. Problem

Chemprop featurizes reactions with the Condensed Graph of Reaction (CGR) and has no mechanism to
enforce — or encourage — any defined behaviour of predicted properties under reaction reversal. For
reaction thermochemistry (ΔH, ΔG, ΔS, ΔE) the required behaviour is exact antisymmetry:

```
f(A→B) = −f(B→A)
```

A trained chemprop model satisfies this only accidentally, and nothing in the loss penalises
violating it. Other reaction properties are exactly symmetric, and others still (activation energy)
are neither — so the general requirement is a **per-task parity**, not a single global constraint.

Verified absent in core as of `1b90e3e2`:

- All six `RxnMode` values are concatenations of reactant-side and product-side features. None is
  antisymmetric, and none makes reversal a sign flip on a known column block.
- `ReactionDatapoint` (`data/datapoints.py:207-211`) stores a fixed `(rct, pdt)` pair with no
  direction flag and no reversed-copy augmentation.
- The CLI exposes no symmetry-related option.
- `RegressionFFN.forward` is an unconstrained FFN with biases.
- A grep for `antisym|reversal|reverse|swap` finds only `rev_edge_index` docstrings, which are
  directed-message bookkeeping in the D-MPNN and unrelated to reaction direction.

## 2. Prior art and why architecture rather than augmentation

[DeepDelta](https://doi.org/10.1186/s13321-023-00769-x) (Fralish et al., *J. Cheminform.* 2023)
addresses the analogous problem for molecular pairs by **data augmentation**: it trains on the full
cross-merge of the training set, so both `(A,B)` and `(B,A)` appear with opposite-signed targets.
Antisymmetry is learned approximately, and the paper evaluates the residual as a self-consistency
check rather than enforcing it.

Evidence on augmentation versus architectural equivariance is mixed, and the two bodies apply to
different regimes:

- [Brehmer et al., *Does equivariance matter at scale?*](https://arxiv.org/abs/2410.23179) (TMLR
  2025) find augmentation largely closes the **data-efficiency** gap given sufficient epochs, while
  equivariance retains a **compute-efficiency** advantage at all budgets studied. This concerns
  E(3), where equivariant architectures pay a heavy expressivity and speed tax.
- [Lyle et al., *On the Benefits of Invariance in Neural Networks*](https://arxiv.org/abs/2005.00178)
  (2020) conclude feature averaging > data augmentation > nothing, and show that invariance learned
  by augmentation fails outside the training distribution. Formal statements are for convex losses.

The construction specified here is Reynolds averaging over ℤ/2. Unlike SE(3)-equivariant layers it
imposes **no restriction on the backbone** — activations, biases and batch norm are unconstrained,
and the encoder is an unmodified chemprop `MessagePassing` block. The cost is 2× forward passes.
The argument that made equivariance lose at scale does not apply.

**Non-goal:** this spec does not claim the architectural route wins empirically. §9 defines the A/B
that would settle it.

## 3. Approach

**3.1 Reversal is a fixed linear map on CGR features.** Writing `x_r` and `x_p` for the
reactant-side and product-side feature vectors of an atom or bond, and `d = x_p − x_r`:

| Mode | Emitted layout | Layout of the reversed reaction |
|---|---|---|
| `REAC_PROD` | `(x_r, x_p)` | `(x_p, x_r)` — block swap |
| `REAC_DIFF` | `(x_r, d)` | `(x_r + d, −d)` |
| `PROD_DIFF` | `(x_p, d)` | `(x_p − d, −d)` |
| **this spec** | `(s, d)` with `s = ½(x_r + x_p)` | `(s, −d)` — **sign flip** |

In the `(s, d)` basis, reversal is a sign flip on a known, contiguous column block. The basis change
is invertible, so no information is lost relative to `REAC_DIFF`.

**3.2 Reversal leaves the topology untouched.** Reversing permutes node order but produces the same
graph. Because message passing is permutation-equivariant and aggregation permutation-invariant, the
reversed graph can be represented by **reusing the forward `edge_index`, `rev_edge_index` and
`batch` tensors verbatim** and transforming only `V` and `E`. No second featurization, no second
graph construction, no second `O(N²)` atom-pair loop.

**3.3 Parity is a per-output property.** The two passes produce `g(x)` and `g(x̄)`. What is done with
that pair is chosen per output column, so a multitask model can mix parities.

## 4. Mathematical specification

Let `rev` be the reversal operator on CGR features, `rev ∘ rev = id`. Let `g` be the full chemprop
stack (message passing → aggregation → batch norm → predictor).

```
odd   (parity −1):  f(x) = ½[g(x) − g(x̄)]      f(x̄) = −f(x)
even  (parity +1):  f(x) = ½[g(x) + g(x̄)]      f(x̄) = +f(x)
free  (parity  0):  f(x) = g(x)                 unconstrained
```

In representation-theoretic terms the model is equivariant under ℤ/2: odd tasks transform under the
sign representation, even tasks under the trivial one. Both are exact for any `g`, with no
restriction on `g`.

**Weight sharing is mandatory.** For two independently parameterised networks,
`f(x) = ½[g₁(x) − g₂(x̄)]` is antisymmetric iff `g₁ + g₁∘rev = g₂ + g₂∘rev`, i.e. iff `g₁` and `g₂`
have identical even parts. That is not imposable on two separately trained D-MPNNs. The two passes
must evaluate one function at two points.

### 4.1 The target scaler must have zero mean on odd tasks

`UnscaleTransform.forward` is `X·scale + mean`, **and is the identity while `self.training` is
True** (`nn/transforms.py:47-48`). The consequence is a bias bug, not a convenience:

- **Training:** `output_transform` is identity, so the model fits `½[g − g_rev]` to the *scaled*
  target `(y − μ)/σ`.
- **Inference:** the transform is active, and the odd combination gives
  `½[(g·σ + μ) − (g_rev·σ + μ)] = ½[g − g_rev]·σ ≈ y − μ`.

The mean cancels, so **every odd-task prediction is biased by −μ**. Confirmed numerically during
review.

**Requirement:** odd-task columns of the target scaler must have `mean == 0` — fit with
`StandardScaler(with_mean=False)` on those columns, or zero them after fitting. The model validates
this at construction (R7). Even and free tasks are unaffected and keep their means.

`on_validation_model_eval` (`models/model.py:163-168`) explicitly puts `output_transform` back into
train mode, so validation loss is computed in scaled space, consistently with training.

### 4.2 The graph scaler must be identity

`GraphTransform` applies `(X − mean)/scale` to `V` and `E` *before* the nonlinear encoder, so a
nonzero mean on an odd column breaks equivariance and does not cancel. It defaults to
`nn.Identity()` (`nn/message_passing/base.py:88`) and is only active with atom/bond feature scaling
enabled. The model rejects a non-identity `graph_transform` (R7).

### 4.3 Coupled barrier heads (motivating case, deferred to v2)

Activation energies satisfy `Ea_rev = Ea_fwd − ΔH`, so `Ea` is neither odd nor even but is rigidly
tied to an odd quantity. Predicting an even head `B` and an odd head `ΔH` and forming
`Ea_fwd = B + ΔH/2`, `Ea_rev = B − ΔH/2` satisfies the relation identically. Data augmentation
cannot express this constraint at all. Out of scope for v1; see §8.

## 5. Component design

### 5.1 `ParityCGRFeaturizer` (`featurizers.py`)

Subclasses `chemprop.featurizers.molgraph.reaction.CondensedGraphOfReactionFeaturizer`. **No core
changes are required.**

It does **not** reimplement the parent's feature construction. It calls
`super()._calc_node_feature_matrix(...)` and `super()._calc_edge_feature(...)`, which yield the
`REAC_DIFF` layout `(x_r, d)`, and applies the basis change in place:

```
s = x_r + d/2        →  emit (s, d)
```

This is why: the parent's balanced and unbalanced branches differ
(`featurizers/molgraph/reaction.py:182-202`), so reimplementing either one silently breaks the
other. Delegating makes R1 true by construction and leaves no duplicated core logic to drift.
Verified working for both `balanced` settings during review.

A `balanced: bool = False` field selects the `_get_bonds` behaviour; `__post_init__` **passes** the
corresponding mode to the parent (`super().__post_init__(RxnMode.REAC_DIFF_BALANCE if self.balanced
else RxnMode.REAC_DIFF)`) rather than assigning `self.mode` first, because the parent's
`__post_init__` assigns `self.mode = mode_` at `reaction.py:76` and would overwrite it. A
user-supplied `mode_` raises `ValueError`: this featurizer defines its own layout and only the
balance behaviour is configurable.

With `F = len(atom_featurizer)`, `k = len(atom_featurizer.atomic_nums) + 1`,
`Bb = len(bond_featurizer)`:

**Node layout** (`atom_fdim = 2F − k`, unchanged from core):

| Columns | Content | Parity |
|---|---|---|
| `[0:k]` | atomic-number block, emitted once | shared |
| `[k:F]` | `½(x_r + x_p)[k:]` | even |
| `[F:2F−k]` | `(x_p − x_r)[k:]` | **odd** |

Emitting the atomic-number block once is valid because `num_only` (`featurizers/atom.py:113-123`)
sets only the atomic-number bit. For *unmapped* atoms this holds by construction; for *mapped*
atoms it holds provided the atom map is element-consistent, which RDKit does not enforce. The
featurizer asserts `x_r[:k] == x_p[:k]` rather than assuming it.

**Edge layout** (`bond_fdim = 2Bb`, unchanged from core):

| Columns | Content | Parity |
|---|---|---|
| `[0:Bb]` | `½(x_e_r + x_e_p)` | even |
| `[Bb:2Bb]` | `x_e_p − x_e_r` | **odd** |

No shared block for bonds: bit 0 of `MultiHotBondFeaturizer` is the null-bond indicator
(`featurizers/bond.py:66-68`) and legitimately differs between sides whenever a bond forms or breaks.

**Exposed properties:** `odd_atom_start -> int` (`F`) and `odd_bond_start -> int` (`Bb`). The odd
block is always trailing, so a start index determines it, and ints serialize into Lightning
hyperparameters where a `slice` would not.

### 5.2 `reverse_bmg` (`featurizers.py`)

```python
def reverse_bmg(bmg: BatchMolGraph, odd_atom_start: int, odd_bond_start: int) -> BatchMolGraph
```

Returns `copy.copy(bmg)` with `V` and `E` replaced by clones whose trailing blocks are negated,
sharing `edge_index`, `rev_edge_index` and `batch` with the input. `BatchMolGraph` is a
`slots=True` dataclass; `copy.copy` followed by field reassignment is the pattern core itself uses
in `GraphTransform.forward` (`nn/transforms.py:69-72`), and was verified to preserve `__len__`.

Takes start indices as arguments rather than hardcoding the layout, so a future cuik-molmaker path
can supply its own (§8).

### 5.3 `parity.py`

```python
ODD, EVEN, FREE = -1, 1, 0

def build_parity_mask(task_parity: Sequence[int], predictor: Predictor) -> Tensor
```

**Predictor outputs are rank-3 `(B, n_tasks, n_targets)`, not flat column blocks.** Multi-target
predictors `torch.chunk` the raw FFN output but **return** `torch.stack(..., dim=2)`
(`nn/predictors.py:186, 207, 226`). `_evaluate_batch` relies on this (`preds[..., 0]`,
`model.py:196-197`), as do `MVELoss` and `QuantileLoss` (`torch.unbind(preds, dim=-1)`). The mask
must match:

| Predictor | `n_targets` | Output rank | Components | Mask shape / value |
|---|---|---|---|---|
| `RegressionFFN` | 1 | `(B, n_tasks)` | mean | `(n_tasks,)` = `task_parity` |
| `MveFFN` | 2 | `(B, n_tasks, 2)` | mean, var | `(n_tasks, 2)` = `task_parity`, `EVEN` |
| `EvidentialFFN` | 4 | `(B, n_tasks, 4)` | mean, v, α, β | `(n_tasks, 4)` = `task_parity`, `EVEN`×3 |
| `QuantileFFN` | 2 | `(B, n_tasks, 2)` | **mean, interval** | `(n_tasks, 2)` = `task_parity`, `EVEN` |

Both shapes broadcast against their predictor's output. A single rule covers every supported
predictor: **component 0 takes the task's parity; all later components are `EVEN`.**

`QuantileFFN` needs no special case. It returns `(mean, interval)`, not `(lower, upper)`
(`predictors.py:223-226`). Under reversal of an odd task the mean negates and the interval width is
preserved, which is exactly `task_parity, EVEN`. An earlier draft of this spec specified a
swap-and-negate branch against a `(lower, upper)` basis chemprop does not emit; review showed the
two are algebraically identical and the generic rule subsumes it.

Forcing later components even is a **correctness requirement, not a convention**: an odd combination
of `MveFFN`'s variance can be negative, which is invalid and produces NaNs in the NLL.

**Supported predictors are an allowlist** — `RegressionFFN`, `MveFFN`, `EvidentialFFN`,
`QuantileFFN`. Anything else raises `ValueError`. An allowlist rather than a classification denylist
because `SpectralFFN` (`predictors.py:344-371`) is regression-registered but emits a normalised
positive spectrum for which odd parity is meaningless, and future predictor types would otherwise be
admitted silently.

### 5.4 `ParityEquivariantMPNN` (`models.py`)

Subclasses `chemprop.models.MPNN`. Adds `odd_atom_start: int`, `odd_bond_start: int`,
`task_parity: Sequence[int]`. A `from_featurizer` classmethod reads the two start indices off a
`ParityCGRFeaturizer`.

**`forward` is not the only entry point.** `training_step` (`model.py:155-156`) and
`validation_step` (`model.py:179-180`) call `self.fingerprint(...)` followed by
`self.predictor.train_step(...)` directly, bypassing `forward` entirely. Only `_evaluate_batch`
(`model.py:193`) and `predict_step` route through `self(...)`. Putting the parity combination solely
in `forward` would leave the constraint **absent from the training objective** and produce a
`(2B, ·)` vs `(B, ·)` shape mismatch against targets.

The method split:

| Method | Shape | Role |
|---|---|---|
| `_paired_fingerprint` | `(2B, d')` | both passes, stacked on the batch dim, after `bn` |
| `fingerprint` | `(B, d')` | **unchanged public contract** — the forward-direction half |
| `_combine` | `(2B, …) → (B, …)` | applies the parity mask |
| `forward` | `(B, …)` | `_combine(predictor(_paired_fingerprint(...)))` |
| `training_step`, `validation_step` | — | override to use `_paired_fingerprint` + `predictor.train_step` + `_combine`, mirroring the parent's masking and weighting |

Keeping `fingerprint` at `(B, d')` means `encoding` (`model.py:136-140`) and the `chemprop
fingerprint` CLI continue to work unchanged, and only three methods are overridden. The returned
fingerprint is direction-dependent; the parity structure lives in the combination, and the README
says so.

`_paired_fingerprint` computes:

```
H_f = agg(message_passing(bmg, V_d), bmg.batch)
H_r = agg(message_passing(reverse_bmg(bmg, ...), V_d), bmg.batch)
H   = bn(cat([H_f, H_r], dim=0))          # (2B, d)
```

then appends `X_d_transform(X_d).repeat(2, 1)` when `X_d` is given. The same `V_d` is forwarded to
both passes: node order is shared between them, so it is direction-independent.
(`ReactionDataset.__getitem__` always emits `V_d = None` today — `data/datasets.py:698` — so this is
currently moot, but the model should not silently drop it.)

**Why stack on the batch dimension.** In `eval()`, `bn` is a fixed per-feature affine map
(`model.py:94, 132`), so exactness holds whether the two passes are normalised together or
separately. The stack matters **in train mode**, where it gives the two passes identical batch
statistics — without it the training objective is not the parity-equivariant function the model
claims to be — and it doubles the sample count feeding the running statistics. The combination is
applied **after** `predictor` (which includes `output_transform`), per §4.1.

`V_d` and `X_d` aside, nothing else in the parent is touched.

## 6. Correctness requirements

| ID | Requirement |
|---|---|
| R1 | `ParityCGRFeaturizer` output is an invertible linear map of the corresponding `REAC_DIFF` output. True by construction given the delegation in §5.1; tested anyway. |
| R2 | For every mode and reaction class, `encoder(reverse_bmg(featurize(rxn)))` equals `encoder(featurize(reverse(rxn)))` at the **aggregated fingerprint** level. Compared post-aggregation because node ordering legitimately differs. |
| R3 | In `eval()`, odd-task outputs satisfy `f(x̄) = −f(x)` to `atol=1e-5`. |
| R4 | In `eval()`, even-task outputs satisfy `f(x̄) = +f(x)`; free-task outputs equal the forward pass exactly. |
| R5 | `MveFFN` variance and `EvidentialFFN` `v, α, β` remain strictly positive for odd tasks. |
| R6 | `QuantileFFN`'s interval component is combined as `EVEN`, so a nonnegative interval stays nonnegative. Core does not guarantee `interval ≥ 0` for an untrained model, so this is stated as parity preservation, not as an ordering assertion on random weights. |
| R7 | Construction raises if: `graph_transform` is not `nn.Identity`; the predictor is outside the §5.3 allowlist; or `output_transform.mean` is nonzero on any odd-task column. |
| R8 | The parity combination is present in `training_step` and `validation_step`, not only `forward`. Gradients reach all encoder parameters; a training smoke test reduces loss and preserves R3 afterwards. |
| R9 | `odd_atom_start`, `odd_bond_start` and `task_parity` survive a checkpoint round-trip, and a reloaded model reproduces predictions and R3 exactly. |

**Exactness holds in `eval()` only.** During training, batch-norm batch statistics and independent
dropout masks across the two passes make the construction approximate. At inference, batch norm uses
running statistics and dropout is off, so the guarantee is exact where it matters. Tests asserting
R3/R4 must call `model.eval()` first — and must do so anyway, because `ScaleTransform`,
`UnscaleTransform` and `GraphTransform` all short-circuit to identity while `self.training` is True,
so a train-mode assertion can pass for unrelated reasons.

## 7. Constraints

- chemprop `>= 2.3.1`. Python `>= 3.11`.
- **No new dependencies** (torch, numpy, chemprop, rdkit). The package joins the unconditional
  import block of `chemprop_contrib/__init__.py` alongside `moe_regressor`; `pyproject.toml` gains
  **no** `[project.optional-dependencies]` entry.
- **No modifications to core chemprop.** Everything by subclassing.
- MIT licence. Layout, LICENSE, README and tests per `chemprop-contrib`'s CONTRIBUTING section.

## 8. Out of scope for v1

- **Coupled ΔH/Ea barrier heads** (§4.3). High value, but a second output transform with its own
  design questions; it should land once the core construction is validated.
- **Single-pass parity channels.** Carrying even and odd hidden channels through the encoder removes
  the 2× forward cost but requires a new `MessagePassing` class and restricts nonlinearities — the
  regime where the Brehmer-style architectural tax argument does apply. (Both this and the shipped
  design are parity-equivariant; they differ in *where* the equivariance lives.)
- **cuik-molmaker support.** `CuikmolmakerCGRFeaturizer` passes a mode integer to a C++ library and
  cannot express a new layout. Since `(s,d)` is a linear map of `REAC_DIFF`, support could be added
  by applying the basis change to cuik output on the GPU.
- **Cycle consistency beyond 2-cycles.** `f(A→B) + f(B→C) + f(C→A) = 0` requires a state-function
  form `f = h(B) − h(A)`, which discards CGR's cross-side bond correspondence.
- **CLI integration.** Model construction is via the Python API.

## 9. Validation plan

Unit tests (§6) establish correctness. They do **not** establish that the approach helps:

- **Arm A** — unmodified `MPNN` on `REAC_DIFF`, trained with reversed-reaction augmentation.
- **Arm B** — `ParityEquivariantMPNN` on `ParityCGRFeaturizer`, no reversal augmentation, since
  parity is satisfied identically.

Arm B costs ~2× per step, so **match on wall-clock, not epochs**. Report accuracy and the
antisymmetry residual `mean |f(A→B) + f(B→A)|` — exactly zero for B, and if large for A, that is
itself the result.

Candidate datasets are the Grambow/RDB7-family reaction sets, which carry ΔH alongside forward and
reverse barriers and so exercise the mixed-parity path. Confirm their column contents before
committing.

Per the project's GPU-budget convention: one quick targeted A/B at reduced steps, then a single full
run (`--steps 1000`) on whichever arm is kept.

## 10. Known risks

1. **Unbalanced reactions.** `num_only` zero-padding reaches reactant-only and product-only atoms
   through different code paths. R2 must cover reactant-heavy and product-heavy cases explicitly.
2. **Silently incomplete training path.** The parity combination must be present in three places,
   not one (§5.4). R8 is the guard, and it belongs in the task that writes the model, not only in a
   late smoke test.
3. **Odd-task target means.** A standard `StandardScaler` silently biases every odd prediction
   (§4.1). R7 is the guard.
4. **Feature scaling.** Enabling atom/bond feature scaling silently breaks equivariance. R7 is the
   guard.
5. **Empty edge arrays.** Reactions with no bonds produce `np.empty((0, bond_fdim))`; `reverse_bmg`
   must handle a zero-row `E`.

**`*_BALANCE` reversal symmetry is not a risk.** An earlier draft rated this the highest risk and
recommended a pre-emptive upstream issue. The gate was run during review across all 6 modes × 9
reactions, post-aggregation and as a direct bag-of-rows comparison of `V`/`E`; all passed, and a case
analysis of `_get_bonds` (`reaction.py:220-263`) confirms each branch maps to its swap under
reversal in both balanced and unbalanced modes. The test is retained as a regression test against
future core changes; the alarm is withdrawn.

## 11. Packaging

```
chemprop_contrib/reaction_parity/
├── LICENSE                 # MIT
├── README.md               # usage, parity table, the eval()-only caveat
├── __init__.py             # public exports
├── featurizers.py          # ParityCGRFeaturizer, reverse_bmg
├── parity.py               # ODD/EVEN/FREE, build_parity_mask
├── models.py               # ParityEquivariantMPNN
└── tests/
    ├── conftest.py
    ├── test_core_symmetry.py   # R2 against core, regression
    ├── test_featurizer.py      # R1
    ├── test_reversal.py        # R2
    ├── test_parity.py          # R5 mask construction, allowlist
    ├── test_models.py          # R3–R9
    ├── make_test_data.py       # derives the fixture columns
    ├── test_data.csv
    └── data_provenance.txt
```

Plus one line in `chemprop_contrib/__init__.py`'s unconditional import block and one bullet in the
root `README.md`. `pyproject.toml` is unchanged.

**Test fixture.** Reaction SMILES and the `ea` column come from chemprop
`tests/data/regression/rxn/rxn.csv` (100 atom-mapped, fully H-mapped reactions; requires
`keep_h=True`). `ea` is a **free** task — activation energy is genuinely neither odd nor even, so it
is correctly typed rather than conveniently typed. Two further columns are *derived*, not measured:

```
y_odd  = MolLogP(pdt) − MolLogP(rct)      exactly odd by construction
y_even = MolLogP(pdt) + MolLogP(rct)      exactly even by construction
```

They are generated by a committed script so the fixture is reproducible and auditable rather than an
opaque CSV, and `data_provenance.txt` states plainly that they are chemically meaningless and exist
only to exercise parity. Deriving them this way — rather than using random targets — is what makes
"loss decreases" a real signal: the target genuinely has the parity its column claims, so the test
shows a constrained model can fit a function of the parity it exists for.

Do **not** use molecular weight for the derived columns: for a balanced reaction `MW(pdt) ==
MW(rct)`, so `y_odd` would be identically zero and the test would pass vacuously. Crippen logP does
not degenerate.

Submission follows the contrib convention: fork, branch, PR using the provided template.

**Recommended before writing code:** open an issue against `chemprop/chemprop` asking whether the
maintainers want the `(s,d)` layout upstreamed as a seventh `RxnMode` — it is useful independently
of parity — with the model staying in contrib. The subclassing design means a "no" costs nothing.
