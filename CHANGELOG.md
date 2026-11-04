# Changelog

## 0.4.0

- Added a zero-training direct-lookup reference for episodic tasks with answers explicitly present in prompts.
- Added optional training-family early stopping, soft wall-time budgets, and CPU thread configuration.
- Added exact-configuration reuse with model/training-code and PyTorch-version checks.
- Logged executed versus requested optimizer steps, elapsed time, and a rough parameter-memory estimate.
- Added regression tests and GitHub Actions smoke checks for the cost-aware paths.
- Kept quality trade-offs explicit: reducing compute can reduce held-out accuracy.

## 0.3.0

- Added episodic in-context retrieval on unseen complete key/value mappings.
- Added counterfactual value-swap tests, input-masking controls and NLL metrics.
- Added multi-seed CLI and explicit mapping-partition checks.
- Kept prior continual learning benchmarks unchanged and added CI smoke coverage.

## 0.2.0

- Added a compositional rule benchmark with held-out operand pairs and deterministic splits.
- Added sequential character-level learning from a local UTF-8 corpus, with disjoint train/test character spans.
- Reported held-out accuracy, held-out NLL, and held-out forgetting after each task.
- Added a non-sequential joint-training reference to separate task difficulty from sequential-learning effects.
- Extended repeated-seed sweeps to include held-out metrics and paired comparisons.
- Bounded importance-estimation work to 128 samples per task for larger datasets.
- Added benchmark tests, a demo corpus, and standalone GitHub Actions smoke tests.

## 0.1.0

- Added a small decoder-only Transformer trained from scratch.
- Compared sequential fine-tuning, reservoir replay, and prioritized replay with EWC-style regularization on synthetic memorization tasks.
- Added checkpoints, a CLI, JSON metrics, and regression tests.
