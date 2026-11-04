# NOVA

A small PyTorch research project for studying **continual learning** in decoder-only Transformers. NOVA trains a model from scratch, introduces tasks one at a time, and measures how well it retains earlier skills. Version 0.2 adds tests on inputs that were **not present during training**, along with a simple text benchmark and an offline reference model.

This is an experimental learning system, **not an AGI model**. The algorithms are established continual-learning techniques combined for comparison; the project does not claim an architectural breakthrough.

## What it compares

| Method | Training procedure | Online learner? |
| --- | --- | --- |
| `finetune` | Only the current task; no rehearsal | Yes |
| `replay` | Current examples plus a bounded memory of older examples | Yes |
| `nova` | Prioritized rehearsal and an EWC-style importance penalty | Yes |
| `joint_oracle` | Samples all tasks together from the start | **No** — upper-bound reference only |

The NOVA method estimates importance from squared minibatch gradients, not an exact Fisher information matrix. It spends additional compute refreshing replay priorities and capturing importance; equal optimizer-step counts do not mean equal compute budgets.

## Setup

Requirements: Python 3.10+, PyTorch, and a machine capable of running small CPU models. No API key, paid service, or pretrained model weights are needed.

```bash
git clone https://github.com/mahasmrtiadhuri/NOVA.git
cd NOVA
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest
```

Run a short experiment first:

```bash
python -m nova experiment --dataset rules --tasks 3 --entities 8 --values 4 \
  --steps 25 --d-model 32 --layers 1 --heads 4 --batch-size 16 \
  --output runs/quick-rules
```

On Windows PowerShell, either put the command on one line or use PowerShell's backtick for continuation instead of `\`.

## Benchmarks

### 1. Rules: unseen combinations of known operands

The model receives four tokens: `<bos> rule_0 number_2 number_5`, then predicts a value token. Each task represents a different mapping: modular addition, modular subtraction, modular maximum, or comparison (`left > right`). Tasks share the same operand tokens and output vocabulary.

A seeded split **reserves whole operand pairs** from every training task. Neither the model nor its replay memory nor its EWC importance estimator sees these pairs during training. After each task, the experiment reports accuracy on both the training pairs and the reserved pairs.

```bash
python -m nova experiment --dataset rules --tasks 3 --entities 12 --values 6 \
  --steps 100 --d-model 64 --layers 2 --heads 4 --output runs/rules
```

This evaluates generalization to **new combinations of seen symbols**, not unseen vocabulary, novel real-world rules, or autonomous skill acquisition. Modular arithmetic is deliberately difficult for a small Transformer; low accuracy is still a valid result.

### 2. Text: held-out next-character prediction

Provide a UTF-8 text file to build sequential next-character tasks. Each task is a chronological chunk of the file; that chunk is split into disjoint train and test character ranges **before** sliding windows are created. The benchmark reports top-1 accuracy and negative log-likelihood (NLL) on held-out text.

```bash
python -m nova experiment --dataset text --corpus examples/sample_text.txt \
  --context-length 8 --steps 25 --d-model 32 --layers 1 \
  --heads 4 --batch-size 16 --output runs/text
```

`examples/sample_text.txt` is a very small, original demo corpus; it is not a serious language-model evaluation dataset. Use a larger, properly licensed local corpus for meaningful research. The loader processes up to the first 12,000 characters to keep this educational model inexpensive. Contiguous splits prevent shared windows across boundaries; repeated *content* elsewhere in a corpus can still occur. The model does not tokenize words or learn to converse.

### 3. Facts: the original retention-only baseline

```bash
python -m nova experiment --dataset facts --steps 65 --output runs/facts
python -m nova query --output runs/facts --method nova --task-id 0 --entity-id 4
```

The original benchmark trains on randomly assigned relation/entity/value facts, then evaluates those **same** facts. It measures memorization and forgetting only. Keep it as a sanity check, not evidence of generalization.

## Run across multiple seeds

```bash
python -m nova sweep --dataset rules --seeds 7,19,42 \
  --tasks 3 --entities 8 --values 4 --steps 25 \
  --d-model 32 --layers 1 --heads 4 --batch-size 16 \
  --output runs/rule-sweep
```

A sweep writes each seed's `metrics.json` and a `summary.json` with means, sample standard deviations, and paired NOVA-minus-baseline differences. Three seeds provide useful variation, **not** statistical significance. Different methods use the same task splits and initial model weights for a given seed. Offline `joint_oracle` sees future tasks and should not be ranked as an online learner.

The following results were observed locally with the exact three-seed command above:

| Method | Mean held-out accuracy | Sample standard deviation |
| --- | ---: | ---: |
| Fine-tune | 0.285 | 0.048 |
| Replay | 0.264 | 0.084 |
| NOVA | 0.278 | 0.043 |
| Joint oracle (not sequential) | 0.285 | 0.098 |

**NOVA did not outperform fine-tuning or the oracle in this small run.** These results are not a claim about the methods' general performance; larger experiments, matching total compute, and stronger task families are needed.

## Outputs

`runs/<name>/metrics.json` contains per-task learning trajectories, model size, training and held-out accuracy, NLL, and forgetting where applicable. Each `experiment` also saves `.pt` checkpoints for the methods. Runs, checkpoints, Python caches, and local secrets are excluded by `.gitignore`.

Rule checkpoints can be queried with:

```bash
python -m nova query --output runs/quick-rules --method nova \
  --task-id 1 --left 2 --right 3
```

For text experiments, evaluate NLL instead of using the fact/rule query command. Never load checkpoints you do not trust.

## Project layout

```text
nova/
  model.py        Decoder-only Transformer
  data.py         Original synthetic fact world
  benchmarks.py   Held-out rule and text datasets
  learning.py     Fine-tuning, replay, consolidation
  experiment.py   Sequential and joint-oracle training
  sweep.py        Multiple seeds and paired summaries
  cli.py          Experiment, sweep and query commands
tests/
  test_nova.py    Regression and benchmark leakage tests
examples/
  sample_text.txt Demonstration text corpus
  seed7_smoke_metrics.json    Original v0.1 retention run
```

## Limits and next experiments

1. Quantify learning on more than one task family, with larger and independent holdout sets.
2. Equalize compute, not just optimizer steps, and profile the cost of consolidation.
3. Add stronger baselines: per-task adapters, external memory, and conventional regularization.
4. Report uncertainty across more seeds and study data contamination from duplicated text.
5. Move toward open-ended environments only after the model demonstrates consistent transfer on controlled tasks.

NOVA v0.2 is a reproducible research prototype. It does not learn arbitrary skills by itself, solve general reasoning tasks, or approach frontier-scale AI performance.
