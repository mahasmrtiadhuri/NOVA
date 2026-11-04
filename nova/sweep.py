"""Paired-seed evaluation with descriptive variance estimates."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean, stdev

from .experiment import ExperimentConfig, run_experiment


def run_sweep(config: ExperimentConfig, seeds: list[int], output_dir: Path) -> dict:
    """Run all methods on identical tasks and splits within each seed.

    Report sample standard deviation, not statistical significance. The
    offline joint oracle is deliberately not a fair online-learning method.
    """
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("A sweep requires at least two distinct seeds")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in seeds:
        cfg = replace(config, seed=seed)
        runs[str(seed)] = run_experiment(
            cfg, output_dir / f"seed_{seed}", save_checkpoints=False,
        )["results"]

    methods = tuple(runs[str(seeds[0])])
    metric_names = ["final_mean_accuracy"]
    if config.dataset == "facts":
        metric_names += ["mean_forgetting"]
    else:
        metric_names += ["final_heldout_accuracy", "final_heldout_nll"]

    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in metric_names:
            if metric not in runs[str(seeds[0])][method]:
                continue
            values = [runs[str(seed)][method][metric] for seed in seeds]
            summary[method][metric + "_mean"] = round(mean(values), 5)
            summary[method][metric + "_std"] = round(stdev(values), 5)

    comparison_metric = "final_mean_accuracy" if config.dataset == "facts" else "final_heldout_accuracy"
    paired_differences = {}
    for baseline in ("finetune", "replay"):
        differences = [
            runs[str(seed)]["nova"][comparison_metric]
            - runs[str(seed)][baseline][comparison_metric]
            for seed in seeds
        ]
        paired_differences[f"nova_minus_{baseline}"] = {
            "metric": comparison_metric,
            "mean": round(mean(differences), 5),
            "std": round(stdev(differences), 5),
        }

    report = {
        "config": asdict(config), "seeds": seeds, "summary": summary,
        "paired_accuracy_differences": paired_differences,
        "note": "Descriptive statistics only. Rule pairs and text spans are held out; "
                "the joint oracle has access to all tasks from the beginning.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report
