# V4 → modular source migration map

The legacy file is the reproduction anchor. Move one group at a time and verify identical outputs before deleting any legacy path.

| Current V4 responsibility | Target module |
|---|---|
| constants, Base263/Cat11 contract | `src/lgaimers/features/registry.py` |
| `add_base_features`, raw schema | `src/lgaimers/data/schema.py`, `features/base.py` |
| `reconstruct_pitch_flags` | `features/pitch_flags.py` |
| Flat priors / seasonal Flat | `features/flat.py` |
| Context Effect / Safe Context / HLogit | `features/context.py` |
| Current-season / preseason missing | `features/current.py` |
| skill snapshots / walk-forward latent skill | `features/latent_skill.py` |
| Prev1/2/3 / REL / trend / reliability | `features/temporal.py` |
| CatBoost arms + fit/predict | `models/catboost.py` |
| State schema / QueryPreprocessor | `models/state.py` |
| StructuredTransformer | `models/transformer.py` |
| FlattenMLP | `models/mlp.py` |
| blend sensitivity / diagnostics | `models/ensemble.py`, `evaluation/metrics.py` |
| dev/locked temporal masks | `evaluation/splits.py` |
| prepare/cache | `pipeline/prepare.py` |
| dev screening | `pipeline/dev.py` |
| 2024 lock | `pipeline/locked.py` |
| 2025 inference/submission | `pipeline/final.py` |

## Migration rule

1. Keep `legacy/model_pipeline_v4.py` unchanged.
2. Move one responsibility into `src/lgaimers/...`.
3. Add a regression test comparing old/new feature columns or predictions on a small fixture.
4. Only switch the experiment runner to the modular implementation after parity passes.
5. Never modify an existing `configs/experiments/exp_XXX.yaml`; create a child experiment instead.
