"""Command-line entry point for NOVA experiments."""

import argparse
from pathlib import Path

import torch

from .benchmarks import rule_value
from .data import make_world
from .experiment import ExperimentConfig, run_experiment
from .model import TinyTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA continual-learning benchmarks")
    parser.add_argument("command", choices=["experiment", "query", "sweep"])
    parser.add_argument("--dataset", choices=["facts", "rules", "text"], default="facts")
    parser.add_argument("--corpus", help="UTF-8 text file for the text benchmark")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--output", default="runs/default", help="Results directory")
    parser.add_argument("--steps", type=int, default=150, help="Optimizer steps per task")
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--entities", type=int, default=32, help="Entities or operands per rule")
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
    parser.add_argument("--seeds", default="7,19,42", help="Comma-separated seeds for sweep")
    parser.add_argument("--method", choices=["finetune", "replay", "nova", "joint_oracle"], default="nova")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--entity-id", type=int, default=0)
    parser.add_argument("--left", type=int, default=0, help="Left operand for rule query")
    parser.add_argument("--right", type=int, default=0, help="Right operand for rule query")
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
            seed=args.seed, device=args.device, dataset=args.dataset,
            corpus=args.corpus, test_fraction=args.test_fraction,
            context_length=args.context_length,
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
                key = "final_heldout_accuracy" if cfg.dataset != "facts" else "final_accuracy"
                print(f"{method:12} {key}={metrics[key + '_mean']:.3f} "
                      f"±{metrics[key + '_std']:.3f}")
            print(f"Wrote {output_dir / 'summary.json'}")
            return
        data = run_experiment(cfg, output_dir)
        print(f"Parameters per model: {data['results']['nova']['parameter_count']:,}")
        for method, result in data["results"].items():
            line = f"{method:12} train accuracy={result['final_mean_accuracy']:.3f}"
            if "final_heldout_accuracy" in result:
                line += (f" heldout accuracy={result['final_heldout_accuracy']:.3f}"
                         f" heldout NLL={result['final_heldout_nll']:.3f}")
            elif "mean_forgetting" in result:
                line += f" forgetting={result['mean_forgetting']:.3f}"
            print(line)
        print(f"Wrote {output_dir / 'metrics.json'}")
        return

    checkpoint = torch.load(
        output_dir / f"{args.method}.pt", map_location=args.device, weights_only=True,
    )
    cfg = checkpoint["config"]
    if cfg.get("dataset") == "text":
        parser.error("Query is for fact and rule experiments; use held-out NLL to assess text")
    world = make_world(
        tasks=cfg["tasks"], entities=cfg["entities"],
        values=cfg["values"], seed=cfg["seed"],
    ) if cfg.get("dataset", "facts") == "facts" else None
    vocab = checkpoint["vocabulary"]
    model = TinyTransformer(
        len(vocab), d_model=cfg["d_model"], layers=cfg["layers"],
        heads=cfg["heads"], max_context=max(8, cfg.get("context_length", 8)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    if world is not None:
        query = world.encode_query(args.task_id, args.entity_id)
        expected = world.vocabulary[int(world.tasks[args.task_id].targets[args.entity_id])]
    else:
        if not 0 <= args.task_id < cfg["tasks"]:
            parser.error("Unknown rule task")
        if not (0 <= args.left < cfg["entities"] and 0 <= args.right < cfg["entities"]):
            parser.error("Operands must be in range")
        lookup = {word: index for index, word in enumerate(vocab)}
        query = torch.tensor([
            lookup["<bos>"], lookup[f"rule_{args.task_id}"],
            lookup[f"number_{args.left}"], lookup[f"number_{args.right}"],
        ])
        expected = f"value_{rule_value(args.task_id, args.left, args.right, cfg['values'])}"
    with torch.inference_mode():
        scores = model(query.unsqueeze(0).to(args.device)).softmax(dim=-1).squeeze(0)
    value_indices = [vocab.index(f"value_{i}") for i in range(cfg["values"])]
    best = max(value_indices, key=lambda idx: scores[idx].item())
    print(f"Predicted: {vocab[best]} (expected: {expected}; "
          f"model probability: {scores[best].item():.3f})")


if __name__ == "__main__":
    main()
