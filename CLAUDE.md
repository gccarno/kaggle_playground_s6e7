# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a Kaggle **Playground Series S6E7** competition repo. The task: predict `health_condition`
(3-class: `unhealthy` / `at-risk` / `fit`) for students, scored on **balanced accuracy**, from a
heavily imbalanced training set (`at-risk` dominates — predicting it for everyone scores ~86% raw
accuracy but only ~33% balanced accuracy).

All the substantive work lives in Jupyter notebooks written to run as **Kaggle kernels** (paths like
`/kaggle/input/...`, `/kaggle/working/...`), not as a local Python package — `src/`, `notebooks/`,
and `experiments/` are currently empty placeholder directories.

## Repo layout

- `ply-s6e7-vibe-v2.ipynb` — 5-base-learner stack: XGBoost + CatBoost + LogisticRegression +
  **RealMLP** (`pytabkit`) + **TabTransformer** (`tab_transformer_pytorch`), both third-party deep
  tabular libraries.
- `ply-s6e7-ft.ipynb` — the newer variant. Same pipeline, but RealMLP + TabTransformer are replaced
  by a single **from-scratch PyTorch FT-Transformer** (no `pytabkit`/`tab-transformer-pytorch`
  dependency, since `pytabkit` doesn't install offline on Kaggle). Prefer this one as the base for
  further edits unless told otherwise.
- `kernel-metadata.json` — Kaggle kernel push config (`kaggle kernels push`). `code_file` currently
  points at `ply-s6e7-ft.ipynb`; update it if the active notebook changes. Runs on GPU T4 x2,
  competition source `playground-series-s6e7`, no internet access during kernel execution (so any
  new pip installs must work from Kaggle's bundled package index).

## Running / iterating on the notebooks

There's no local `train.csv`/`test.csv` and no requirements file — these notebooks are designed to
execute inside a Kaggle kernel environment where the competition data is mounted at
`/kaggle/input/competitions/playground-series-s6e7/`. To push a new version to Kaggle:

```
kaggle kernels push -p .
```

(requires the Kaggle CLI configured with API credentials). When editing notebook cells, keep the
`!pip install -q ...` cell in sync with whatever libraries the notebook actually imports — Kaggle
kernels run with `enable_internet: false`, so any package not preinstalled in the `kaggle/python`
Docker image must be verified to install/import successfully offline, or avoided (this is exactly
why the `ft` notebook dropped `pytabkit`).

## Pipeline architecture (shared by both notebooks)

Both notebooks follow the same end-to-end structure; understanding it holistically matters more than
any single cell, since later stages depend on column-set/dtype decisions made early on:

1. **Load + target encode** (`cls_to_int`/`int_to_cls`, sorted class order) — raw numeric columns
   (`RAW_NUM`) and raw categorical columns (`RAW_CAT`) are listed explicitly up front.
2. **Feature engineering** (`engineer_features`, all unsupervised/leakage-safe): ratio & interaction
   features, nonlinear transforms, plus two families of categoricals *derived from* numerics —
   domain-threshold clinical bins (`*_category`) and train-fitted quantile bins (`*_qbin`, via
   `fit_quantile_edges`/`apply_quantile_bins`). Intentionally redundant; pruning happens in step 4.
3. **Column-type detection** (`detect_column_types`) runs **after** FE so derived bins are routed as
   categorical. Categorical columns get cast to `str` (missingness becomes an explicit `"nan"`
   level rather than a hole).
4. **Multiclass OOF target encoding** (`oof_target_encode_multiclass`): each categorical expands
   into `K` per-class-rate columns, fit OOF within each usage site to stay leakage-safe. Used
   repeatedly (feature selection, HPO, XGB/LR base learners) — always re-fit per train/val split
   passed in, never fit once globally.
5. **SHAP-based feature selection**: a quick class-balanced XGBoost is fit on a 75/25 split (target
   encoded), `TreeExplainer` SHAP values are computed on the held-out 25%, per-encoded-column
   importances are folded back onto their *source* feature (summing `col__te0..teK`), and the
   smallest feature set reaching 95% cumulative |SHAP| (floor `MIN_FEATURES=12`) is kept — `X`,
   `X_test`, `CAT_COLS`, `NUM_COLS` are all narrowed to this set for the rest of the notebook.
6. **Optuna HPO** on a 20% stratified sample, 3-fold CV, optimizing **balanced accuracy** directly —
   only for XGBoost/CatBoost/LogisticRegression (`obj_xgb`/`obj_cat`/`obj_lr`); the neural learner(s)
   use fixed hyperparameters (per-fold HPO on a net inside the outer stack would be too slow).
7. **Base learner fit functions** (`fit_xgb`, `fit_cat`, `fit_lr`, `fit_ft` or
   `fit_realmlp`/`fit_tabtransformer`) — each takes `(Xt, yt, Xv, Xte, yv)` and returns
   `(val_proba, test_proba)` of shape `(n, K)`. Class imbalance is handled per-learner: sample
   weights for XGB/LR, CatBoost's `auto_class_weights="Balanced"`, resampling for RealMLP, and
   balanced cross-entropy weights for the from-scratch FT-Transformer/TabTransformer.
8. **Outer 5-fold stacking**: `StratifiedKFold(N_FOLDS=5)`, every base learner is fit per fold, OOF
   validation predictions build `oof_meta` (shape `n × (N_BASE·K)`), test predictions are averaged
   across folds into `test_meta`. `BASE_LEARNERS` list order must match the concatenation order fed
   into `oof_meta`/`test_meta` slots (`_slot(i)`).
9. **Meta-learner**: multinomial `LogisticRegression` trained on `oof_meta` (itself 5-fold CV'd) to
   produce final class probabilities; mean |coef| per base learner reports how much each learner
   contributes (near-zero = redundant with the trees, which is an acceptable outcome, not a bug).
   A **prior-correction toggle** (`meta_test_proba / priors`) is applied only if it improves OOF
   balanced accuracy — always re-validate this on OOF before trusting it, don't assume it's on.
10. **Submission**: writes `submission.csv` with columns `id`, `health_condition` (string labels via
    `int_to_cls`).

## Conventions to preserve when editing

- **Leakage discipline is the load-bearing constraint of this codebase.** Target encoders, quantile
  bin edges, scalers, imputers, and category vocabularies must always be *fit* on a train fold only
  and *applied* to the corresponding val/test fold — never fit on the full `X` before the outer CV
  loop. This pattern repeats identically in feature selection, HPO, and the final stack; match it in
  any new base learner.
- **Balanced accuracy is the metric to optimize everywhere** — CV scoring, Optuna objectives, and
  neural early stopping all use `balanced_accuracy_score`, not plain accuracy/logloss.
- **GPU detection is centralized**: `USE_GPU = torch.cuda.is_available()` gates `xgb_device_kwargs()`
  / `cat_device_kwargs()` / the torch device string, all defined once near the top of the notebook.
  New learners should follow the same pattern rather than hardcoding a device.
- Unseen categorical values at inference time (val/test) must map to a reserved "unknown" index/level
  rather than erroring — every categorical encoder in these notebooks (target encoding, FT-Transformer
  vocab, TabTransformer vocab) follows this rule.
