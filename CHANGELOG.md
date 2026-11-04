# Changelog

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
