"""Repeated-seed benchmarks for paired continual-learning comparisons."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean, stdev

from .experiment import ExperimentConfig, run_experiment


def run_sweep(config: ExperimentConfig, seeds: list[int], output_dir: Path) -> dict:
    """Run all methods on the same task worlds for each random seed.

    Experiments are paired by seed, not by individual minibatch samples. Results
    are descriptive; confidence intervals or significance claims need larger
    pre-registered experiments.
    """
    if len(seeds) < 2:
        raise ValueError("A sweep requires at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in seeds:
        run_config = replace(config, seed=seed)
        runs[str(seed)] = run_experiment(
            run_config, output_dir / f"seed_{seed}", save_checkpoints=False,
        )["results"]

    methods = ("finetune", "replay", "nova")
    summary = {}
    for method in methods:
        scores = [runs[str(seed)][method]["final_mean_accuracy"] for seed in seeds]
        forgetting = [runs[str(seed)][method]["mean_forgetting"] for seed in seeds]
        summary[method] = {
            "final_accuracy_mean": round(mean(scores), 5),
            "final_accuracy_std": round(stdev(scores), 5),
            "forgetting_mean": round(mean(forgetting), 5),
            "forgetting_std": round(stdev(forgetting), 5),
        }

    paired_differences = {}
    for baseline in ("finetune", "replay"):
        differences = [
            runs[str(seed)]["nova"]["final_mean_accuracy"]
            - runs[str(seed)][baseline]["final_mean_accuracy"]
            for seed in seeds
        ]
        paired_differences[f"nova_minus_{baseline}"] = {
            "mean": round(mean(differences), 5),
            "std": round(stdev(differences), 5),
        }

    report = {
        "config": asdict(config),
        "seeds": seeds,
        "summary": summary,
        "paired_accuracy_differences": paired_differences,
        "note": "Descriptive statistics on synthetic training facts, not proof of generalization or statistical significance.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report
