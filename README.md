# NOVA

A small, reproducible continual-learning experiment with a **decoder-only Transformer trained from scratch**. The goal is to study how a model retains old associations when learning new ones, not to claim AGI or state-of-the-art results.

NOVA compares three methods on the same sequence of synthetic tasks:

- **`finetune`** — train only on the current task.
- **`replay`** — mix examples from a bounded reservoir of earlier tasks into each batch.
- **`nova`** — prioritize previously difficult examples for replay and add an EWC-style parameter-importance penalty to discourage changes to important earlier parameters.

The `nova` combination is an **experimental baseline**, not a novel research breakthrough. Both replay and EWC-style regularization are established continual-learning methods. Here importance is estimated from squared minibatch gradients; it is only an approximation to the empirical Fisher. A real research claim would need repeated seeds, better baselines, matched compute and tests on more realistic datasets.

## Quick start

Python 3.10+ and PyTorch required. CPU is fine for the default experiment. You do **not** need an API key, account, or pretrained model.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
pytest
python -m nova experiment --steps 150 --output runs/demo
python -m nova query --output runs/demo --method nova --task-id 0 --entity-id 4
```

The experiment writes `runs/demo/metrics.json` plus one `.pt` checkpoint per method. These are local and ignored by Git. To keep experiments fast, run `python -m nova experiment --steps 10 --d-model 32 --layers 1` first.

## Research question

> Can rehearsal and a parameter-importance penalty reduce forgetting when a small Transformer learns a sequence of conflicting synthetic associations?

Each task defines a random relation-specific lookup for shared entity IDs. A query is a four-token prompt, such as:

```text
<bos> relation_1 entity_7 <query>  →  value_3
```

The network predicts the next token. Task 1, 2, etc. reuse the same entities and output symbols but assign different relations and values, causing potential interference.

**Important limitation:** All evaluated associations were present in training. Accuracy here measures *memorization and retention*, **not** novel reasoning, understanding, open-domain language modeling, self-directed learning, or generalization to unseen facts.

## Training and evaluation

All three methods start from the same seeded random initialization and use the same number of optimizer steps and total examples per batch. `replay` and `nova` replace part of each current-task batch with earlier-task examples. NOVA also performs periodic extra inference passes over its memory and computes an EWC snapshot after each task; **compute is therefore not fully equalized**. Treat comparisons as educational, not conclusive.

After each task, the script records accuracy on **every** task, including tasks not yet trained (those are diagnostic only). `final_mean_accuracy` is the mean accuracy after the last task. `mean_forgetting` averages, for every earlier task, its best accuracy since being learned minus its final accuracy.

For a paired **multi-seed benchmark**, use `sweep`. This writes a `summary.json`
with mean and standard deviation across runs, plus the per-seed metrics:

```bash
python -m nova sweep --seeds 7,19,42 --steps 65 --output runs/sweep
```

A sweep avoids judging a method from a single favorable random seed. However,
three seeds are still too few for strong statistical claims. The experiment
reports descriptive variation rather than significance tests.

Try individual seeds and model sizes:

```bash
python -m nova experiment --seed 7 --steps 150 --output runs/seed7
python -m nova experiment --seed 19 --steps 150 --output runs/seed19
python -m nova experiment --d-model 192 --layers 4 --heads 6 --steps 300 --output runs/larger
```

Results depend on seed, task difficulty and hyperparameters; **NOVA may perform worse than a simpler baseline**. Compare the measured numbers, not the method names.

## Layout

```text
nova/
  cli.py          Command-line interface and checkpoint queries
  data.py         Deterministic synthetic fact generator
  model.py        Small decoder-only Transformer
  learning.py     Fine-tuning, replay, priority updates, EWC
  experiment.py   Seeded experiment runner and JSON metrics
  sweep.py        Paired multi-seed benchmark summary
  __main__.py     python -m nova entry point
tests/
  test_nova.py    Core regression and integration tests
```

## Follow-on experiments

1. Add multiple seeds and bootstrap confidence intervals before comparing methods.
2. Add evaluation on held-out paraphrases or transfer tasks to test actual generalization.
3. Compare against LoRA adapters, standard EWC, gradient replay, and an external-memory baseline.
4. Profile wall time and tokens processed; match total compute in controlled comparisons.
5. Replace synthetic facts with small, permissively licensed sequential text tasks.

The architecture is intentionally compact enough to inspect, modify and run without paid infrastructure.

## Example result (CPU, seed 7)

An observed run with **3 tasks × 24 entities, 6 values, 65 steps per task, and a 105,088-parameter model** produced:

| Method | Final mean accuracy | Mean forgetting |
| --- | ---: | ---: |
| Fine-tune | 0.472 | 0.792 |
| Replay | **0.958** | **0.062** |
| NOVA | **0.972** | **0.042** |

This is one small synthetic run, not a statistically supported model comparison. On this seed NOVA scored slightly higher, but another seed or configuration could reverse that result. Reproduce with:

```bash
python -m nova experiment --steps 65 --tasks 3 --entities 24 --values 6 --d-model 64 --layers 2 --heads 4 --batch-size 24 --output runs/smoke
```
