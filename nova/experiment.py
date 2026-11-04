"""Reproducible sequential-task experiments with held-out evaluation."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .benchmarks import Benchmark, make_rule_benchmark, make_text_benchmark
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
    dataset: str = "facts"
    test_fraction: float = 0.25
    corpus: str | None = None
    context_length: int = 8


def load_benchmark(config: ExperimentConfig) -> Benchmark | None:
    if config.dataset == "facts":
        return None
    if config.dataset == "rules":
        return make_rule_benchmark(
            tasks=config.tasks, operands=config.entities, values=config.values,
            seed=config.seed, test_fraction=config.test_fraction,
        )
    if config.dataset == "text":
        if not config.corpus:
            raise ValueError("--corpus is required for the text dataset")
        return make_text_benchmark(
            config.corpus, tasks=config.tasks, context_length=config.context_length,
            test_fraction=config.test_fraction,
        )
    raise ValueError(f"Unknown dataset: {config.dataset}")


def _new_model(vocab_size: int, config: ExperimentConfig) -> TinyTransformer:
    return TinyTransformer(
        vocab_size, d_model=config.d_model, layers=config.layers,
        heads=config.heads, max_context=max(8, config.context_length),
    ).to(config.device)


@torch.inference_mode()
def evaluate_loss(model: TinyTransformer, world: FactWorld, *, device: torch.device) -> list[float]:
    """Mean cross-entropy per held-out sample, evaluated in fixed-size chunks."""
    was_training = model.training
    model.eval()
    results = []
    for task in world.tasks:
        total = 0.0
        for i in range(0, len(task.targets), 128):
            batch_prompts = task.prompts[i:i + 128].to(device)
            batch_targets = task.targets[i:i + 128].to(device)
            total += F.cross_entropy(model(batch_prompts), batch_targets, reduction="sum").item()
        results.append(round(total / len(task.targets), 5))
    model.train(was_training)
    return results


def run_method(
    world: FactWorld, config: ExperimentConfig, method: str, *,
    heldout: FactWorld | None = None,
) -> tuple[dict, TinyTransformer]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = _new_model(world.vocab_size, config)
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
        row = {
            "after_task": task_id,
            "accuracy_by_task": accuracy,
            "mean_seen_accuracy": round(sum(accuracy[:task_id + 1]) / (task_id + 1), 5),
            "last_training_loss": round(losses[-1], 5),
        }
        if heldout is not None:
            heldout_accuracy = evaluate(model, heldout, device=device)
            row.update({
                "heldout_accuracy_by_task": heldout_accuracy,
                "heldout_nll_by_task": evaluate_loss(model, heldout, device=device),
                "mean_seen_heldout_accuracy": round(
                    sum(heldout_accuracy[:task_id + 1]) / (task_id + 1), 5,
                ),
            })
        trajectory.append(row)
    final_accuracy = trajectory[-1]["accuracy_by_task"]
    forgetting = [
        round(max(row["accuracy_by_task"][task_id] for row in trajectory[task_id:]) - final_accuracy[task_id], 5)
        for task_id in range(config.tasks - 1)
    ]
    result = {
        "method": method,
        "parameter_count": model.parameter_count,
        "trajectory": trajectory,
        "optimizer_steps": config.steps * config.tasks,
        "final_mean_accuracy": round(sum(final_accuracy) / config.tasks, 5),
        "mean_forgetting": round(sum(forgetting) / len(forgetting), 5) if forgetting else 0,
    }
    if heldout is not None:
        last = trajectory[-1]
        final_heldout = last["heldout_accuracy_by_task"]
        heldout_forgetting = [
            round(max(row["heldout_accuracy_by_task"][task_id] for row in trajectory[task_id:])
                  - final_heldout[task_id], 5)
            for task_id in range(config.tasks - 1)
        ]
        result.update({
            "final_heldout_accuracy": round(sum(final_heldout) / config.tasks, 5),
            "final_heldout_nll": round(sum(last["heldout_nll_by_task"]) / config.tasks, 5),
            "mean_heldout_forgetting": round(
                sum(heldout_forgetting) / len(heldout_forgetting), 5,
            ) if heldout_forgetting else 0,
        })
    return result, model


def run_joint_oracle(
    world: FactWorld, config: ExperimentConfig, heldout: FactWorld,
) -> tuple[dict, TinyTransformer]:
    """Offline upper bound: can train on all tasks from the first update.

    It is not a valid online-learning competitor because it sees future tasks.
    It does use the same total number of optimizer steps and batch size.
    """
    torch.manual_seed(config.seed)
    model = _new_model(world.vocab_size, config)
    device = torch.device(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(config.seed + 808)
    model.train()
    for _ in range(config.tasks * config.steps):
        sampled_tasks = torch.randint(config.tasks, (config.batch_size,), generator=generator)
        prompts, targets = [], []
        for task_id in sampled_tasks.tolist():
            task = world.tasks[task_id]
            index = int(torch.randint(len(task.targets), (1,), generator=generator))
            prompts.append(task.prompts[index])
            targets.append(task.targets[index])
        inputs = torch.stack(prompts).to(device)
        labels = torch.stack(targets).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    train_acc = evaluate(model, world, device=device)
    test_acc = evaluate(model, heldout, device=device)
    test_loss = evaluate_loss(model, heldout, device=device)
    return {
        "method": "joint_oracle",
        "note": "Offline reference only: trains on all tasks at once, including future tasks",
        "parameter_count": model.parameter_count,
        "optimizer_steps": config.steps * config.tasks,
        "final_mean_accuracy": round(sum(train_acc) / len(train_acc), 5),
        "final_heldout_accuracy": round(sum(test_acc) / len(test_acc), 5),
        "final_heldout_nll": round(sum(test_loss) / len(test_loss), 5),
        "train_accuracy_by_task": train_acc,
        "heldout_accuracy_by_task": test_acc,
    }, model


def run_experiment(config: ExperimentConfig, output_dir: Path, *, save_checkpoints: bool = True) -> dict:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    benchmark = load_benchmark(config)
    world = benchmark.train if benchmark else make_world(
        tasks=config.tasks, entities=config.entities, values=config.values, seed=config.seed,
    )
    heldout = benchmark.test if benchmark else None
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in ("finetune", "replay", "nova"):
        metrics, model = run_method(world, config, method, heldout=heldout)
        results[method] = metrics
        if save_checkpoints:
            torch.save({
                "state_dict": model.state_dict(), "config": asdict(config),
                "vocabulary": world.vocabulary,
            }, output_dir / f"{method}.pt")
    if benchmark is not None:
        metrics, model = run_joint_oracle(world, config, heldout)
        results["joint_oracle"] = metrics
        if save_checkpoints:
            torch.save({
                "state_dict": model.state_dict(), "config": asdict(config),
                "vocabulary": world.vocabulary,
            }, output_dir / "joint_oracle.pt")
    result = {
        "config": asdict(config),
        "dataset": benchmark.description if benchmark else (
            "Synthetic overlapping entity-relation-value associations; "
            "training-fact retention, not out-of-distribution generalization"
        ),
        "benchmark": benchmark.details if benchmark else None,
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
