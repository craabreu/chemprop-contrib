# Parity-Equivariant Reaction Property Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `chemprop-contrib` package that makes reaction-property predictions exactly parity-equivariant under reaction reversal — odd, even or unconstrained per task — with no changes to core chemprop.

**Architecture:** A `CondensedGraphOfReactionFeaturizer` subclass delegates to the parent and applies a basis change so CGR features land in a `(sum, diff)` basis, making reversal a sign flip on a trailing column block that reuses the forward graph topology verbatim. An `MPNN` subclass evaluates the shared encoder at both orbit points, stacks the fingerprints on the batch dimension, and combines predictor outputs per task.

**Tech Stack:** Python ≥3.11, chemprop ≥2.3.1, PyTorch, NumPy, RDKit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-parity-equivariant-reaction-models.md`

## Global Constraints

- chemprop `>= 2.3.1`; Python `>= 3.11`. Verified against chemprop at `1b90e3e2`.
- **No new dependencies.** torch, numpy, chemprop, rdkit only. `pyproject.toml` is not modified.
- **No modifications to core chemprop.** Everything by subclassing.
- **Self-contained.** The `(s,d)` layout and everything else stays in `chemprop_contrib/reaction_parity/`; no upstream issue, PR or `RxnMode` proposal is part of this work.
- All code under `chemprop_contrib/reaction_parity/`. MIT licence.
- Parity constants: `ODD = -1`, `EVEN = 1`, `FREE = 0`.
- Every parity assertion runs after `model.eval()`. `ScaleTransform`, `UnscaleTransform` and `GraphTransform` all short-circuit to identity while `self.training` is True, so a train-mode assertion can pass for unrelated reasons.
- Tolerance for float comparisons: `atol=1e-5`.
- Test reactions: copy the `rxn_smis` list from chemprop `tests/unit/featurizers/test_cgr.py:51-61` into `tests/conftest.py` with a provenance comment naming that file and commit `1b90e3e2`. It covers balanced+mapped, reordered atom indices, partially mapped on each side, product-heavy, reactant-heavy, and a zero-edge reaction.
- Build molecules with `chemprop.utils.make_mol(smi, keep_h=True, add_h=False)`, as `test_cgr.py:140-143` does. `Chem.MolFromSmiles` drops the unmapped `[H]` in two fixtures.
- `AVAILABLE_RXN_MODE_NAMES` lives in chemprop's *test* tree and is not importable from an install — use `RxnMode.keys()`.
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MYqeKqjp62cW5TbGdfp7MD
  ```

## Review Focus

1. **Unbalanced reactions** — reactant-only and product-only atoms reach `num_only` through different code paths, so symmetry must be checked on reactant-heavy and product-heavy cases, not only balanced ones. Covered in Tasks 1 and 3.
2. **Odd-task target means** — a stock `StandardScaler` leaves `mean ≠ 0`, which cancels in the odd combination and biases every odd prediction by `−μ` with no error anywhere. Covered in Task 5.
3. **Zero-edge reactions** — `"[CH4:1]>>[CH3:1].[H:2]"` yields `E` of shape `(0, bond_fdim)`; `reverse_bmg` must return a correctly-shaped empty tensor rather than raising or broadcasting. Covered in Task 3.
4. **Enabled atom/bond feature scaling** — a non-identity `graph_transform` subtracts a mean *before* the encoder, which does not cancel and silently destroys equivariance. Covered in Task 5.
5. **Checkpoint round-trip** — if `task_parity`, `odd_atom_start` and `odd_bond_start` do not survive, a reloaded model either fails to load or silently loses its constraint. Covered in Task 7.

---

### Task 1: Core reversal-symmetry regression test

Pins the assumption the whole package rests on: that core's CGR of a reversed reaction is the mirror of the forward one. **This was run during spec review and passed on all 6 modes × 9 reactions**, so it is a regression test against future core changes, not an open question. Uses no package code — it applies the known `REAC_DIFF` linear map to core's own output.

**Files:**
- Create: `chemprop_contrib/reaction_parity/tests/conftest.py`
- Create: `chemprop_contrib/reaction_parity/tests/test_core_symmetry.py`

**Interfaces:**
- Consumes: nothing.
- Produces, all from `conftest.py` and reused by Tasks 2, 3 and 5:
  - `rxn_smi` — a pytest fixture parametrized over the 7 `rxn_smis`.
  - `make_rxn(smi: str) -> tuple[Chem.Mol, Chem.Mol]` — splits on `>>`, returns the `(reactant, product)` pair `__call__` expects, via `make_mol(..., keep_h=True, add_h=False)`.
  - `reverse_rxn_smi(smi: str) -> str`
  - `encode(mg: MolGraph | BatchMolGraph) -> Tensor` — a seeded untrained `BondMessagePassing` + `MeanAggregation` returning a `(1, d)` graph-level fingerprint. Must accept both types: Task 3 passes it a `BatchMolGraph`.

- [ ] **Step 1: Write `conftest.py` helpers**

Comparison is post-aggregation because node ordering legitimately differs between directions and aggregation is permutation-invariant.

- [ ] **Step 2: Write the failing test**

```python
@pytest.mark.parametrize("mode", list(RxnMode.keys()))
def test_core_cgr_reversal_is_mirror(rxn_smi, mode):
    f = CondensedGraphOfReactionFeaturizer(mode_=mode)
    mg_fwd = f(make_rxn(rxn_smi))
    mg_rev = f(make_rxn(reverse_rxn_smi(rxn_smi)))
    assert torch.allclose(encode(apply_reversal(mg_fwd, f, mode)), encode(mg_rev), atol=1e-5)
```

`apply_reversal` implements the spec §3.1 table and lives in the test file, not the package — it exists only to interrogate core. Node blocks are `[k:F]` ("mid") and `[F:2F−k]` ("tail"); edge blocks are `[0:Bb]` and `[Bb:2Bb]`. For `REAC_DIFF`: `mid += tail; tail *= -1`. For `PROD_DIFF`: `mid -= tail; tail *= -1`. For `REAC_PROD`: swap the two *trailing* blocks, leaving the shared `[0:k]` head alone (`reaction.py:210` emits it once). `MolGraph` is a `NamedTuple` (`data/molgraph.py:6`), so build a new one with `._replace` rather than assigning fields.

- [ ] **Step 3: Run the test**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_core_symmetry.py -v`
Expected: PASS, 7 reactions × 6 modes. A failure means core changed; stop and investigate before continuing.

- [ ] **Step 4: Commit**

```bash
git add chemprop_contrib/reaction_parity/tests/
git commit -m "test: pin CGR reversal symmetry in core chemprop"
```

---

### Task 2: `ParityCGRFeaturizer`

**Files:**
- Create: `chemprop_contrib/reaction_parity/__init__.py`, `featurizers.py`
- Create: `chemprop_contrib/reaction_parity/tests/test_featurizer.py`

**Interfaces:**
- Consumes: `encode`, `make_rxn`, `rxn_smi` (Task 1).
- Produces:
  ```python
  @dataclass
  class ParityCGRFeaturizer(CondensedGraphOfReactionFeaturizer):
      balanced: bool = False
      @property
      def odd_atom_start(self) -> int: ...   # len(self.atom_featurizer)
      @property
      def odd_bond_start(self) -> int: ...   # len(self.bond_featurizer)
  ```

- [ ] **Step 1: Write the failing tests**

```python
def test_dimensions_match_core(): ...
    # atom_fdim == 2F - k, bond_fdim == 2B, identical to REAC_DIFF

@pytest.mark.parametrize("balanced", [False, True])
def test_is_invertible_map_of_reac_diff(rxn_smi, balanced): ...
    # R1: recovering (x_r, d) via s - d/2 reproduces the REAC_DIFF V and E exactly

def test_rejects_explicit_mode(): ...
    # mode_= supplied by the user -> pytest.raises(ValueError)

def test_asserts_shared_atomic_number_block(): ...
    # x_r[:k] == x_p[:k] is checked, not assumed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_featurizer.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `ParityCGRFeaturizer` in `featurizers.py`**

Override `_calc_node_feature_matrix` and `_calc_edge_feature` to **call `super()`** and apply the basis change `mid += 0.5 * tail` to the returned `REAC_DIFF` layout. Do not reimplement the parent's construction: its balanced and unbalanced branches differ (`reaction.py:182-202`), so reimplementing one breaks the other, and delegating makes R1 true by construction.

`__post_init__` must **pass** the mode to the parent, not assign it first — the parent assigns `self.mode = mode_` at `reaction.py:76` and would overwrite an earlier assignment:

```python
def __post_init__(self, mode_):
    if mode_ is not None: raise ValueError(...)
    super().__post_init__(RxnMode.REAC_DIFF_BALANCE if self.balanced else RxnMode.REAC_DIFF)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_featurizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chemprop_contrib/reaction_parity/
git commit -m "feat: add ParityCGRFeaturizer emitting CGR features in (sum, diff) basis"
```

---

### Task 3: `reverse_bmg`

**Files:**
- Modify: `chemprop_contrib/reaction_parity/featurizers.py`
- Create: `chemprop_contrib/reaction_parity/tests/test_reversal.py`

**Interfaces:**
- Consumes: `ParityCGRFeaturizer.odd_atom_start` / `.odd_bond_start` (Task 2); `encode`, `make_rxn`, `reverse_rxn_smi`, `rxn_smi` (Task 1).
- Produces: `reverse_bmg(bmg: BatchMolGraph, odd_atom_start: int, odd_bond_start: int) -> BatchMolGraph`.

- [ ] **Step 1: Write the failing tests**

```python
def test_shares_topology_tensors(): ...
    # out.edge_index IS bmg.edge_index (identity); same for rev_edge_index, batch
def test_does_not_mutate_input(): ...
def test_is_an_involution(): ...
def test_handles_zero_edge_reaction(): ...
    # "[CH4:1]>>[CH3:1].[H:2]": E is (0, bond_fdim) in and out
def test_preserves_len(): ...

@pytest.mark.parametrize("balanced", [False, True])
def test_matches_refeaturized_reverse(rxn_smi, balanced): ...
    # R2: encode(reverse_bmg(featurize(rxn))) == encode(featurize(reverse(rxn)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_reversal.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `reverse_bmg`**

`copy.copy(bmg)`, then assign cloned `V` and `E` with trailing blocks negated. `BatchMolGraph` is a `slots=True` dataclass; `copy.copy` + field reassignment is the pattern core uses in `nn/transforms.py:69-72`. Take start indices as arguments so a future cuik-molmaker path can supply its own.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_reversal.py -v`
Expected: PASS, 7 reactions × 2 balance settings.

- [ ] **Step 5: Commit**

```bash
git add chemprop_contrib/reaction_parity/
git commit -m "feat: add reverse_bmg sign-flip reversal reusing graph topology"
```

---

### Task 4: Parity mask construction

**Files:**
- Create: `chemprop_contrib/reaction_parity/parity.py`
- Create: `chemprop_contrib/reaction_parity/tests/test_parity.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ODD`, `EVEN`, `FREE`, and `build_parity_mask(task_parity: Sequence[int], predictor: Predictor) -> Tensor` of shape `(n_tasks,)` when `n_targets == 1`, else `(n_tasks, n_targets)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_regression_mask_is_1d(): ...
    # RegressionFFN(n_tasks=3), [ODD, EVEN, FREE] -> shape (3,), tensor([-1, 1, 0])

def test_mve_mask_is_2d_with_even_uncertainty(): ...
    # MveFFN(n_tasks=2), [ODD, ODD] -> shape (2, 2), [[-1, 1], [-1, 1]]

def test_evidential_mask_shape_and_values(): ...
    # EvidentialFFN(n_tasks=1), [ODD] -> shape (1, 4), [[-1, 1, 1, 1]]

def test_quantile_uses_the_generic_rule(): ...
    # QuantileFFN(n_tasks=1), [ODD] -> shape (1, 2), [[-1, 1]]
    # it returns (mean, interval), so no special case is needed

def test_mask_broadcasts_against_predictor_output(): ...
    # for each predictor: (mask * predictor(Z)).shape == predictor(Z).shape

def test_rejects_predictors_outside_allowlist(): ...
    # BinaryClassificationFFN, MulticlassClassificationFFN, SpectralFFN -> ValueError

def test_rejects_wrong_length_task_parity(): ...
def test_rejects_invalid_parity_value(): ...
```

The shapes follow from multi-target predictors returning `torch.stack(..., dim=2)` — `(B, n_tasks, n_targets)` — while `RegressionFFN` returns `(B, n_tasks)` (`predictors.py:166-167, 186, 207, 226`). Forcing later components `EVEN` is a correctness requirement: an odd combination of `MveFFN`'s variance can be negative and produces NaNs in the NLL.

`SpectralFFN` is regression-registered but emits a normalised positive spectrum, so use an allowlist of `{RegressionFFN, MveFFN, EvidentialFFN, QuantileFFN}` rather than a classification denylist.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_parity.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `parity.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_parity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chemprop_contrib/reaction_parity/
git commit -m "feat: add per-task parity mask construction"
```

---

### Task 5: `ParityEquivariantMPNN`

**Files:**
- Create: `chemprop_contrib/reaction_parity/models.py`
- Create: `chemprop_contrib/reaction_parity/tests/test_models.py`

**Interfaces:**
- Consumes: `reverse_bmg`, `ParityCGRFeaturizer` (Tasks 2–3); `build_parity_mask`, `ODD`/`EVEN`/`FREE` (Task 4).
- Produces:
  ```python
  class ParityEquivariantMPNN(MPNN):
      def __init__(self, message_passing, agg, predictor, odd_atom_start: int,
                   odd_bond_start: int, task_parity: Sequence[int],
                   batch_norm: bool = False, **kwargs): ...
      @classmethod
      def from_featurizer(cls, featurizer: ParityCGRFeaturizer, message_passing,
                          agg, predictor, task_parity, **kwargs) -> "ParityEquivariantMPNN": ...
  ```

- [ ] **Step 1: Write the failing tests**

```python
def test_odd_task_is_exactly_antisymmetric(rxn_smi): ...   # R3, in eval()
def test_even_task_is_exactly_symmetric(rxn_smi): ...      # R4
def test_free_task_equals_forward_pass(rxn_smi): ...       # R4

def test_training_step_applies_the_parity_combination(): ...
    # R8, the critical one: training_step must NOT inherit the parent's path.
    # Assert its preds have batch dim B (not 2B) and that an odd task's
    # training-mode preds satisfy f(x_rev) == -f(x) up to the batch-norm caveat
def test_validation_step_applies_the_parity_combination(): ...
def test_fingerprint_keeps_batch_dim_B(): ...
    # the public contract `encoding` and the fingerprint CLI depend on
def test_encoding_is_unchanged(): ...

def test_rejects_nonzero_odd_task_target_mean(): ...
    # Review Focus 2: UnscaleTransform with mean != 0 on an odd column -> ValueError
def test_zero_mean_odd_task_is_unbiased(): ...
    # with_mean=False on odd columns: eval prediction has no -mu offset
def test_rejects_non_identity_graph_transform(): ...        # Review Focus 4
def test_extra_descriptors_reach_both_passes(): ...
def test_v_d_is_forwarded_to_both_passes(): ...
def test_antisymmetry_holds_with_batch_norm(rxn_smi): ...
    # guards the train-mode property: both passes share batch statistics
def test_output_shape_matches_plain_mpnn(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_models.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `ParityEquivariantMPNN`**

`__init__` validates per R7 — `graph_transform` is `nn.Identity`, predictor is in the allowlist, and `output_transform.mean` is zero on odd-task columns — then registers the mask as a **non-persistent** buffer (`register_buffer(..., persistent=False)`) so it does not duplicate the `task_parity` hyperparameter in `state_dict`.

Four methods, per spec §5.4:

| Method | Shape | Role |
|---|---|---|
| `_paired_fingerprint` | `(2B, d')` | both passes, stacked on dim 0, after `bn` |
| `fingerprint` | `(B, d')` | **unchanged contract** — forward-direction half |
| `_combine` | `(2B, …) → (B, …)` | applies the mask |
| `forward` | `(B, …)` | `_combine(predictor(_paired_fingerprint(...)))` |

Then override `training_step` and `validation_step`. This is load-bearing: they call `self.fingerprint` then `self.predictor.train_step` directly (`model.py:155-156`, `179-180`), bypassing `forward`. Leaving them inherited would omit the constraint from the training objective and mismatch shapes against targets. Mirror the parent's masking, weighting and logging exactly; only the preds computation changes. `encoding` and `_evaluate_batch` need no override because `fingerprint` keeps its `(B, d')` shape.

Stack on the **batch** dimension, not the feature dimension: in `eval()` either works, but in train mode the stack is what gives both passes identical batch statistics.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chemprop_contrib/reaction_parity/
git commit -m "feat: add ParityEquivariantMPNN with per-task parity combination"
```

---

### Task 6: Uncertainty predictors

**Files:**
- Modify: `chemprop_contrib/reaction_parity/tests/test_models.py`

**Interfaces:**
- Consumes: `ParityEquivariantMPNN` (Task 5). No new public symbols.

No production code is expected. `MveFFN`, `EvidentialFFN` and `QuantileFFN` are already covered by the Task 4 mask; this task proves it, and exists as a separate reviewer gate because a silent sign error here produces NaNs only once a loss is computed.

- [ ] **Step 1: Write the tests**

```python
def test_mve_variance_stays_positive_for_odd_task(): ...        # R5
def test_evidential_params_stay_positive_for_odd_task(): ...    # R5: v, alpha, beta > 0
def test_quantile_mean_is_odd_and_interval_is_even(): ...       # R6
def test_mve_loss_is_finite_for_odd_task(): ...
    # the failure this task exists to catch: NaN NLL from a negative variance
```

- [ ] **Step 2: Run them**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_models.py -k "mve or evidential or quantile" -v`
Expected: PASS without new production code. **If any fail, the Task 4 mask is wrong** — fix it there, not with a special case here.

- [ ] **Step 3: Commit**

```bash
git add chemprop_contrib/reaction_parity/
git commit -m "test: cover MVE, evidential and quantile predictors under parity"
```

---

### Task 7: Fixture data, training and checkpoint round-trip

**Files:**
- Create: `chemprop_contrib/reaction_parity/tests/make_test_data.py`, `test_data.csv`, `data_provenance.txt`
- Modify: `chemprop_contrib/reaction_parity/tests/test_models.py`

**Interfaces:**
- Consumes: `ParityEquivariantMPNN` (Tasks 5–6). Produces nothing.

- [ ] **Step 1: Write `make_test_data.py` and generate the fixture**

Reads chemprop `tests/data/regression/rxn/rxn.csv` (100 atom-mapped, fully H-mapped reactions; needs `keep_h=True`), keeps `smiles` and `ea`, and derives two columns with RDKit:

```
y_odd  = MolLogP(pdt) - MolLogP(rct)      exactly odd
y_even = MolLogP(pdt) + MolLogP(rct)      exactly even
```

`ea` is the **free** task — genuinely neither odd nor even, so correctly typed rather than conveniently typed. Do **not** use molecular weight: for a balanced reaction `MW(pdt) == MW(rct)`, so `y_odd` would be identically zero and every test would pass vacuously.

`data_provenance.txt` names the source file and commit `1b90e3e2`, states `ea` is as published, and states plainly that `y_odd`/`y_even` are derived, chemically meaningless, and exist only to exercise parity.

- [ ] **Step 2: Write the failing tests**

```python
def test_gradients_reach_all_encoder_parameters(): ...
def test_training_reduces_loss_on_an_exactly_odd_target(): ...
    # meaningful because y_odd genuinely has odd parity: this shows a
    # constrained model can fit a function of the parity it exists for
def test_antisymmetry_survives_training(): ...              # R8
def test_checkpoint_round_trip_preserves_parity(): ...      # R9, Review Focus 5
    # save_checkpoint -> load_from_checkpoint: task_parity, odd_atom_start and
    # odd_bond_start survive; reloaded model reproduces predictions and R3
```

- [ ] **Step 3: Run tests to verify they fail, then make them pass**

Run: `pytest chemprop_contrib/reaction_parity/tests/test_models.py -k "training or gradient or checkpoint" -v`

Note on R9: Lightning's `collect_init_args` walks up the `__init__` frames, so `MPNN.__init__`'s existing `save_hyperparameters` call (`model.py:82`) already captures subclass arguments — verified during review. **The subclass must not call `save_hyperparameters` again.** If the round-trip fails, the cause is elsewhere.

- [ ] **Step 4: Run the full suite**

Run: `pytest chemprop_contrib/reaction_parity/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chemprop_contrib/reaction_parity/
git commit -m "test: add derived fixture data, training and checkpoint round-trip tests"
```

---

### Task 8: Package integration

**Files:**
- Create: `chemprop_contrib/reaction_parity/LICENSE`, `README.md`
- Modify: `chemprop_contrib/__init__.py`, root `README.md`

**Interfaces:**
- Consumes: all public symbols from Tasks 2–5.
- Produces: `from chemprop_contrib import reaction_parity`.

- [ ] **Step 1: Write the failing test**

```python
def test_package_is_importable_from_namespace():
    from chemprop_contrib import reaction_parity
    assert {"ParityCGRFeaturizer", "reverse_bmg", "ParityEquivariantMPNN",
            "build_parity_mask", "ODD", "EVEN", "FREE"} <= set(reaction_parity.__all__)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest chemprop_contrib/reaction_parity/tests/ -k importable -v`
Expected: FAIL

- [ ] **Step 3: Wire up the package**

Export the public symbols from `reaction_parity/__init__.py`. Add `reaction_parity` to the **unconditional** import block and `__all__` in `chemprop_contrib/__init__.py` — it has no extra dependencies, so it belongs with `moe_regressor`, not in a `try/except ImportError`. Add MIT `LICENSE`. Add one bullet to the root `README.md`. `pyproject.toml` is **not** modified.

- [ ] **Step 4: Run the full suite**

Run: `pytest chemprop_contrib/ -v`
Expected: PASS, with no other package's tests broken.

- [ ] **Step 5: Write `reaction_parity/README.md`**

Covers: the physical motivation, leading with antisymmetry since that is what readers will search for; the parity table with example targets (ΔH odd, reorganization energy even, `Ea` free); a minimal end-to-end example via `from_featurizer`; **the `with_mean=False` requirement for odd-task targets**, which is a silent bias if missed; the **eval-only** exactness caveat; that `fingerprint` returns the direction-dependent forward half; unsupported combinations (predictors outside the allowlist, non-identity `graph_transform`, cuik-molmaker); and the deferred coupled ΔH/Ea recipe `Ea_fwd = B + ΔH/2`, `Ea_rev = B − ΔH/2` as future work.

- [ ] **Step 6: Commit**

```bash
git add chemprop_contrib/ README.md
git commit -m "docs: add reaction_parity README, licence and exports"
```

---

## Post-plan

The spec's §9 A/B benchmark is **not** part of this plan — it depends on a dataset choice, and the unit tests establish correctness without it. Run it before opening the PR if the PR is to make an empirical claim; open the PR as a correctness-only contribution otherwise.

The work is self-contained in chemprop-contrib (spec §11). Do not open issues or PRs against `chemprop/chemprop`, and do not propose the `(s,d)` layout as a core `RxnMode`; nothing in this plan waits on upstream.
