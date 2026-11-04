"""Reproducible sequential-task experiments and machine-readable outputs."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import torch

from .data import FactWorld, make_world
from .learning import Consolidation, Memory, evaluate, train_one_task
from .model import TinyTransformer


@dataclass(frozen=True)
class ExperimentConfig:
    tasks: int = 3
    entities: int = 32
    values: int = 8
    steps: int = 150
    batch_size: int = 32
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    learning_rate: float = 0.002
    replay_fraction: float = 0.3
    memory_capacity: int = 128
    consolidation_strength: float = 2.0
    seed: int = 7
    device: str = "cpu"


def run_method(world: FactWorld, config: ExperimentConfig, method: str) -> tuple[dict, TinyTransformer]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = TinyTransformer(
        world.vocab_size, d_model=config.d_model, layers=config.layers, heads=config.heads
    ).to(device)
    memory = Memory(config.memory_capacity, seed=config.seed)
    consolidation = Consolidation()
    trajectory: list[dict] = []
    for task_id in range(config.tasks):
        losses = train_one_task(
            model, world, task_id, method=method, steps=config.steps,
            batch_size=config.batch_size, learning_rate=config.learning_rate,
            replay_fraction=config.replay_fraction, memory=memory,
            consolidation=consolidation, consolidation_strength=config.consolidation_strength,
            device=device, seed=config.seed,
        )
        accuracy = evaluate(model, world, device=device)
        trajectory.append({
            "after_task": task_id,
            "accuracy_by_task": accuracy,
            "mean_seen_accuracy": round(sum(accuracy[:task_id + 1]) / (task_id + 1), 5),
            "last_training_loss": round(losses[-1], 5),
        })
    final_accuracy = trajectory[-1]["accuracy_by_task"]
    forgetting = [
        round(max(row["accuracy_by_task"][task_id] for row in trajectory[task_id:]) - final_accuracy[task_id], 5)
        for task_id in range(config.tasks - 1)
    ]
    return {
        "method": method,
        "parameter_count": model.parameter_count,
        "trajectory": trajectory,
        "final_mean_accuracy": round(sum(final_accuracy) / config.tasks, 5),
        "mean_forgetting": round(sum(forgetting) / len(forgetting), 5) if forgetting else 0,
    }, model


def run_experiment(config: ExperimentConfig, output_dir: Path, *, save_checkpoints: bool = True) -> dict:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    world = make_world(tasks=config.tasks, entities=config.entities, values=config.values, seed=config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in ("finetune", "replay", "nova"):
        metrics, model = run_method(world, config, method)
        results[method] = metrics
        if save_checkpoints:
            torch.save({"state_dict": model.state_dict(), "config": asdict(config), "vocabulary": world.vocabulary}, output_dir / f"{method}.pt")
    result = {
        "config": asdict(config),
        "dataset": "Synthetic overlapping entity-relation-value associations; training-fact retention, not out-of-distribution generalization",
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
