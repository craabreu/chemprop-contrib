"""Derive bench_lc.py from the user's bench2.py by exact, checked substitutions (bench2.py is not edited)."""
import pathlib

src = pathlib.Path("/home/charlles/data/rdb7/bench2.py").read_text()
reps = [
    ('"""RDB7 benchmark on the protocol',
     '"""Learning-curve variant of bench2.py: trains on a random fraction of the training set.\n\n'
     "Differences from bench2.py (generated from it by make_bench_lc.py):\n"
     "- trains on the first round(FRACTION * n) reactions of a seeded permutation of the fold's training\n"
     "  set (seed 1000 + fold), so the subsets are nested across fractions and identical for all models;\n"
     "  stock_aug adds the reverses of the same reactions. Validation and test sets are the full ones.\n"
     "- constant number of gradient updates: with K = round(1 / FRACTION), trains for EPOCHS * K epochs,\n"
     "  warms up for 5 * K epochs and validates every K epochs, so the learning-rate schedule and the\n"
     "  number of validation checks match the full-data run update for update.\n"
     "- imports chemprop_contrib from an export of commit 42887e4 (the (m, d) feature basis), not from\n"
     "  the worktree, and records the package path in result.json\n"
     "Usage: python bench_lc.py MODEL FOLD SPLIT EPOCHS FRACTION OUTDIR\n\n"
     "Original docstring:\n\nRDB7 benchmark on the protocol"),
    ("kind, fold, split, epochs, outdir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), "
     "pathlib.Path(sys.argv[5])\n",
     "kind, fold, split, epochs, frac, outdir = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),\n"
     "                                            float(sys.argv[5]), pathlib.Path(sys.argv[6]))\n"
     "K = round(1 / frac)\n"),
    ("here = pathlib.Path(__file__).parent\n", 'here = pathlib.Path("/home/charlles/data/rdb7")\n'),
    ("import lightning.pytorch as pl\n",
     "sys.path.insert(0, \"/tmp/claude-1000/-home-charlles-github-repos-chemprop-contrib--claude-worktrees-\"\n"
     "                \"reaction-parity/d614dc31-7a4a-4fe1-9021-e4ced7195e61/scratchpad/lcurve/md_42887e4\")\n"
     "import lightning.pytorch as pl\n"),
    ('    "train_time_s": round(train_time, 1),\n',
     '    "train_time_s": round(train_time, 1),\n'
     '    "package": sys.modules["chemprop_contrib.reversal_parity.featurizers"].__file__,\n'),
    ('sets = json.loads((here / f"splits_{split}.json").read_text())[fold]\n',
     'sets = json.loads((here / f"splits_{split}.json").read_text())[fold]\n'
     'n_full = len(sets["train"])\n'
     "perm = np.random.default_rng(1000 + int(fold)).permutation(n_full)\n"
     'sets["train"] = [sets["train"][j] for j in perm[: round(frac * n_full)]]\n'),
    ("lr = dict(warmup_epochs=5, init_lr=1e-4, max_lr=1e-3, final_lr=1e-5)\n",
     "lr = dict(warmup_epochs=5 * K, init_lr=1e-4, max_lr=1e-3, final_lr=1e-5)\n"),
    ("trainer = pl.Trainer(max_epochs=epochs, accelerator",
     "trainer = pl.Trainer(max_epochs=epochs * K, check_val_every_n_epoch=K, accelerator"),
    ('"best_epoch": int(best["epoch"]),\n',
     '"best_epoch": int(best["epoch"]),\n'
     '    "fraction": frac, "n_train_reactions": len(sets["train"]), "epochs_run": epochs * K, "val_every": K,\n'),
]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
pathlib.Path(__file__).with_name("bench_lc.py").write_text(src)
