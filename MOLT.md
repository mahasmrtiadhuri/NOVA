# NOVA MOLT

NOVA MOLT explores **neural-to-symbolic skill compilation**: turning repeated
answers from a small model into a bounded, auditable program, then routing
future in-domain requests through that program instead of invoking the model.

This is an educational, experimentally measurable implementation of a known
research direction. It does not invent a new general-purpose reasoning engine.
It is **not** a replacement for an LLM or a guarantee of lower energy use.

## How it works

1. Train a tiny `TinyTransformer` on all inputs in a finite integer task.
2. Split the domain into compilation examples, independent compilation
   validation examples, and a final test set. The *teacher* sees all inputs
   during its own training; only the *compiler* is held out from validation/test.
3. Query the teacher for answers on training and validation examples.
4. Search a **fixed** interpreter-backed language of numeric rules, including
   addition, XOR, max/min and small affine functions. There is no Python `eval`
   and no generated code execution.
5. Reject compilation when no rule fits, evidence is insufficient, candidates
   disagree outside observed examples, or the search budget is exhausted.
6. Store an approved skill as JSON together with validation probes.
7. Route in-domain requests to the skill; fall back to the reference model for
   unsupported tasks or out-of-domain inputs.
8. Optionally shadow-check a fraction of calls against an independent verifier
   and disable a mismatching skill.

**The acceptance test verifies only observed examples.** Even if all finite-domain
candidate programs agree, the true unknown function could behave differently
on unobserved inputs. We therefore measure independent test accuracy and
report limitations; we do not claim formal correctness or generalization.

## Quick start

Python 3.10+, local PyTorch and pytest:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m nova.molt demo --operator max --steps 300 --modulus 8 \
  --requests 300 --output runs/molt-max
python -m nova.molt query --skill runs/molt-max/skill.json --left 5 --right 3
```

Try a different task:

```bash
python -m nova.molt demo --operator xor --steps 300 --output runs/molt-xor
python -m nova.molt demo --operator add_mod --steps 300 --output runs/molt-add
```

The last experiment may fail to compile. That's an intended outcome when a
teacher makes even a small number of inconsistent predictions: unsupported or
unreliable skills are not silently promoted.

## Integrate the compiler

```python
from nova.molt import Example, MoltRouter, compile_skill
from nova.molt.store import save_skill, load_skill

# These are input-output examples obtained from a model or another solver.
train = [Example(a, b, max(a, b)) for a in range(8) for b in range(4)]
valid = [Example(a, b, max(a, b)) for a in range(8) for b in range(4, 7)]
result = compile_skill('maximum', train, valid, modulus=8)
assert result.skill is not None
save_skill(result.skill, 'skill.json')

# The fallback is your model (or your application's reference solver).
router = MoltRouter(lambda task, a, b: max(a, b))
router.install(load_skill('skill.json'))
print(router.predict('maximum', 7, 2))
print(router.predict('maximum', 10, 2))  # outside certified domain: fallback
```

If you have a trusted independent reference, set
`MoltRouter(fallback, verifier=reference, verify_every=10)` to check every tenth
compiled request. A failing check disables the skill. Verifier calls also cost
resources, so a production deployment needs to include them in cost analysis.

## Benchmarks and measurement

`runs/molt-max/report.json` contains:

- teacher's training duration and full-domain accuracy;
- compilation wall-clock time and number of tested candidates;
- classification accuracy on independent *compiler-held-out* input pairs;
- mean, median and 95th-percentile per-request latency for teacher, MOLT
  router and an intentionally included handwritten Python baseline;
- an **estimated** compilation break-even request count based on observed
  per-request latency savings; this excludes teacher training and storage;
- optional CPU package energy (Linux RAPL), otherwise `null`.

Timing is descriptive and highly hardware/OS dependent; models are small and
requests are sequential. There is no GPU performance claim, no strong baseline
against optimized production inference, and no credible claim of 80% lower
physical energy without supporting sensors and proper matched experiments.

The machine-learning *teacher is trained on the entire finite input domain*.
That makes this a study of **compressing an already-learned skill**, not
learning new tasks from out-of-distribution examples or surpassing the teacher.
The handcrafted Python baseline will typically be faster than the router,
which is exactly why it's important to include it.

## What still needs research

- Compiling useful operations from real model traces with more expressive,
  well-controlled program search.
- Trustworthy uncertainty calibration and data-drift detection in open domains.
- Cost-efficient batching, CPU/GPU power instrumentation, and repeated
  measurements with error bars on realistic workloads.
- Comparing against caching, ordinary distillation, model routing, and
  optimized operator implementations.
- Demonstrating net **lifetime** benefits after compilation and verification
  costs, not just isolated inference speedups.

All dependencies are open-source. No paid APIs are required, but actual hardware
usage and electricity are not free. Do not execute untrusted generated code.
