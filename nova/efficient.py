"""Compute-light references for tasks with an exact known solution.

These are deliberately non-learning baselines. Routing a solvable task to a
small deterministic program avoids wasting neural-network training compute.
"""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from .episodes import EpisodeConfig, EpisodeGenerator


def lookup_demonstrated_value(prompts: torch.Tensor) -> torch.Tensor:
    """Return the value attached to each queried key in an episodic prompt.

    A key must appear exactly once in the demonstrations. This procedure does
    not predict an unseen relation, infer an unknown value, or learn weights.
    """
    if prompts.ndim != 2 or prompts.shape[1] < 7 or prompts.shape[1] % 2 == 0:
        raise ValueError("Expected [batch, 2 * examples + 3] token IDs")
    if not torch.all(prompts[:, 0] == EpisodeGenerator.bos) or not torch.all(
        prompts[:, -2] == EpisodeGenerator.query
    ):
        raise ValueError("Invalid episode delimiters")
    keys = prompts[:, 1:-2:2]
    values = prompts[:, 2:-2:2]
    matches = keys == prompts[:, -1].unsqueeze(1)
    if not torch.all(matches.sum(dim=1) == 1):
        raise ValueError("Query must match exactly one demonstrated key")
    return values.gather(1, matches.long().argmax(dim=1, keepdim=True)).squeeze(1)


def run_lite_baseline(config: EpisodeConfig, output_dir: Path) -> dict:
    """Evaluate exact in-context lookup using zero trainable parameters."""
    config.validate()
    generator = EpisodeGenerator(
        symbols=config.symbols, examples=config.examples, seed=config.seed + 7919,
    )
    prompts, targets = generator.batch(config.eval_episodes, "test")
    start = perf_counter()
    answers = lookup_demonstrated_value(prompts)
    swapped, swapped_targets = generator.counterfactual(prompts)
    swapped_answers = lookup_demonstrated_value(swapped)
    elapsed = perf_counter() - start
    report = {
        "method": "direct_context_lookup",
        "config": asdict(config),
        "parameters": 0,
        "optimizer_steps": 0,
        "heldout_accuracy": round(float((answers == targets).float().mean().item()), 5),
        "counterfactual_accuracy": round(
            float((swapped_answers == swapped_targets).float().mean().item()), 5,
        ),
        "elapsed_seconds": round(elapsed, 6),
        "queries": len(targets),
        "scope": "Only lookup of an explicitly demonstrated key; not a general learner",
        "cost_note": "No paid API or model training; CPU time and electricity are not zero.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lite_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report
