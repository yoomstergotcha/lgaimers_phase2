# LGAimers 

## Principle

- **Code** describes reusable behavior.
- **Experiment YAML** describes one hypothesis/run.
- **Run directory** stores immutable outputs and the frozen config.
- **Git commit + config hash + pipeline manifest** identify what produced a result.
- **Raw data never changes in place.**

## Directory structure

```text
.
├── configs/
│   ├── data/                 # paths / schema
│   ├── features/             # feature-set definitions
│   ├── models/               # reusable model defaults
│   ├── validation/           # temporal split / lock policy
│   └── experiments/          # ONLY place to define experiment variants
├── data/
│   ├── raw/                  # immutable source files
│   ├── interim/              # reconstructed flags etc.
│   ├── processed/            # model-ready tables
│   └── cache/                # disposable cached artifacts
├── src/lgaimers/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── pipeline/
│   └── utils/
├── notebooks/
│   ├── eda/                  # exploration only; no canonical pipeline logic
│   └── experiments/          # visualization/analysis only
├── tests/                    # regression/leakage tests
├── legacy/
│   └── model_pipeline_v4.py  # 실험 중인 코드
├── scripts/
│   └── run_experiment.py
└── runs/
    └── exp_XXX/              # immutable run artifacts
```

## Put the actual competition files here

The current anchor expects:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/trackman_history.csv   # optional for the current V1 model input
```


## Reproduce the current V4 anchor

```bash
python scripts/run_experiment.py \
  --config configs/experiments/exp_000_v4_repro.yaml \
  --stage prepare

python scripts/run_experiment.py \
  --config configs/experiments/exp_000_v4_repro.yaml \
  --stage dev

python scripts/run_experiment.py \
  --config configs/experiments/exp_000_v4_repro.yaml \
  --stage locked

python scripts/run_experiment.py \
  --config configs/experiments/exp_000_v4_repro.yaml \
  --stage final
```

The first invocation creates `runs/exp_000_v4_repro/config.frozen.yaml`. If you later edit the YAML but reuse the same experiment ID, the runner aborts. This prevents silent experiment mutation.

## How to change a feature

Do **not** create `model_pipeline_v5.py`.

Example: test `R-only seasonal history`.

1. Copy `configs/experiments/exp_000_v4_repro.yaml` to `exp_002_r_only.yaml`.
2. Change:

```yaml
experiment:
  id: exp_002_r_only
  parent: exp_000_v4_repro

features:
  temporal_history_scope: R
```

3. Run dev under the new config.

The code stays the same; only the hypothesis/config differs.

## What belongs in notebooks?

Notebooks may inspect distributions, compare run outputs, and visualize errors. They must **not** contain the canonical feature-generation implementation. Once an EDA idea is accepted, move it to `src/lgaimers/features/` and test it.

## Recommended experiment naming

```text
exp_000_v4_repro
exp_001_rel_level
exp_002_rel_trend
exp_003_rel_level_plus_trend
exp_004_reliability_only
exp_005_query_missing_embed
```

Keep the numeric ID permanent. Add `parent:` to record which experiment it changed from.

## Next migration step

Use `MIGRATION_MAP.md`. Migrate the V4 code one component at a time, always checking numerical parity against `legacy/model_pipeline_v4.py` before switching the runner.
