# Changelog

## 0.4.0 — NOVA MOLT research prototype

- Added a bounded neural-to-symbolic skill compiler based on a finite, explicit integer DSL.
- Added separate compilation training, validation and untouched test splits; certification refuses unsupported, ambiguous, and budget-truncated cases.
- Added a router with domain checks, fallback, optional independent spot verification and automatic skill disabling on disagreement.
- Added JSON-only skill persistence with revalidation on load; no dynamic code execution.
- Added a tiny trainable Transformer teacher, CPU latency benchmark, handwritten Python baseline, compilation break-even estimate, and optional Linux RAPL readings.
- Added a standalone command-line demo, reproducible example report, and dedicated regression tests.
- Does not claim new algorithm discovery, general-purpose compression, or measured hardware energy savings.

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
