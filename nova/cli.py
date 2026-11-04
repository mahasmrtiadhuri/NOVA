"""Command-line entry point."""

import argparse
from pathlib import Path

import torch

from .data import make_world
from .experiment import ExperimentConfig, run_experiment
from .model import TinyTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA continual-learning experiments")
    parser.add_argument("command", choices=["experiment", "query", "sweep"])
    parser.add_argument("--output", default="runs/default", help="Results directory")
    parser.add_argument("--steps", type=int, default=150, help="Optimizer steps per task")
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--entities", type=int, default=32)
    parser.add_argument("--values", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--replay-fraction", type=float, default=0.3)
    parser.add_argument("--memory-capacity", type=int, default=128)
    parser.add_argument("--consolidation-strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seeds", default="7,19,42", help="Comma-separated seeds for the sweep command")
    parser.add_argument("--method", choices=["finetune", "replay", "nova"], default="nova")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--entity-id", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.command in {"experiment", "sweep"}:
        cfg = ExperimentConfig(
            tasks=args.tasks, entities=args.entities, values=args.values,
            steps=args.steps, batch_size=args.batch_size,
            d_model=args.d_model, layers=args.layers, heads=args.heads,
            learning_rate=args.learning_rate, replay_fraction=args.replay_fraction,
            memory_capacity=args.memory_capacity,
            consolidation_strength=args.consolidation_strength,
            seed=args.seed, device=args.device,
        )
        if args.command == "sweep":
            from .sweep import run_sweep

            try:
                seeds = [int(value.strip()) for value in args.seeds.split(",")]
            except ValueError:
                parser.error("--seeds must contain comma-separated integers")
            if len(seeds) < 2 or len(set(seeds)) != len(seeds):
                parser.error("--seeds needs at least two distinct values")
            report = run_sweep(cfg, seeds, output_dir)
            for method, metrics in report["summary"].items():
                print(f"{method:9} accuracy={metrics['final_accuracy_mean']:.3f} "
                      f"±{metrics['final_accuracy_std']:.3f}")
            print(f"Wrote {output_dir / 'summary.json'}")
            return
        data = run_experiment(cfg, output_dir)
        print(f"Parameters per model: {data['results']['nova']['parameter_count']:,}")
        for method, result in data["results"].items():
            print(f"{method:9} final accuracy={result['final_mean_accuracy']:.3f} "
                  f"forgetting={result['mean_forgetting']:.3f}")
        print(f"Wrote {output_dir / 'metrics.json'}")
        return

    # Only load checkpoint files the user created or trusts: torch.load uses a
    # restricted weights-only loader here, but model files remain untrusted input.
    checkpoint = torch.load(output_dir / f"{args.method}.pt", map_location=args.device, weights_only=True)
    cfg = checkpoint["config"]
    world = make_world(tasks=cfg["tasks"], entities=cfg["entities"], values=cfg["values"], seed=cfg["seed"])
    model = TinyTransformer(world.vocab_size, d_model=cfg["d_model"], layers=cfg["layers"], heads=cfg["heads"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    query = world.encode_query(args.task_id, args.entity_id)
    with torch.inference_mode():
        scores = model(query.unsqueeze(0).to(args.device)).softmax(dim=-1).squeeze(0)
    value_indices = [world.token_to_id[f"value_{i}"] for i in range(cfg["values"])]
    best = max(value_indices, key=lambda idx: scores[idx].item())
    predicted = world.vocabulary[best]
    expected = world.vocabulary[int(world.tasks[args.task_id].targets[args.entity_id])]
    print(f"relation_{args.task_id} entity_{args.entity_id} -> {predicted} "
          f"(expected: {expected}; model probability: {scores[best].item():.3f})")


if __name__ == "__main__":
    main()
