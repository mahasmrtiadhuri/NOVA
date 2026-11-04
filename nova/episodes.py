"""In-context associative learning on unseen randomly assigned symbol tables.

A fresh environment supplies key/value examples and asks for the value of a
previously demonstrated key. The Transformer must learn to use the examples in
its context; weights are not updated during evaluation.
"""

from dataclasses import asdict, dataclass, replace
import json
import itertools
import math
from statistics import mean, stdev
from pathlib import Path
import random
import zlib

import torch
from torch import nn
from torch.nn import functional as F

from .model import TinyTransformer


@dataclass(frozen=True)
class EpisodeConfig:
    symbols: int = 8
    examples: int = 4
    steps: int = 800
    batch_size: int = 64
    learning_rate: float = 0.001
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    seed: int = 7
    eval_episodes: int = 512
    device: str = "cpu"

    def validate(self) -> None:
        if not (4 <= self.symbols <= 32):
            raise ValueError("symbols must be between 4 and 32")
        if not (2 <= self.examples <= self.symbols):
            raise ValueError("examples must be between 2 and symbols")
        if min(self.steps, self.batch_size, self.eval_episodes) < 1:
            raise ValueError("steps, batch_size, eval_episodes must be positive")
        if self.d_model < 8 or self.d_model % self.heads:
            raise ValueError("d_model must be >=8 and divisible by heads")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")


class EpisodeGenerator:
    """Create context examples from disjoint families of complete mappings.

    Full permutations belong to train (80%) or holdout (20%) based on a stable
    checksum. Demonstrated subsets can overlap across those families; only the
    *complete mappings* are guaranteed disjoint.
    """

    bos = 0
    query = 1
    mask = 2

    def __init__(self, *, symbols: int = 8, examples: int = 4, seed: int = 7):
        if symbols < 4 or examples < 2 or examples > symbols or symbols > 32:
            raise ValueError("Require 4..32 symbols and 2..symbols examples")
        self.symbols = symbols
        self.examples = examples
        self.key_start = 3
        self.value_start = self.key_start + symbols
        self.vocab_size = self.value_start + symbols
        self.context_length = 2 * examples + 3
        self.rng = random.Random(seed)

    @staticmethod
    def partition(mapping: tuple[int, ...]) -> str:
        bucket = zlib.crc32(bytes(mapping)) % 5
        return "test" if bucket == 0 else "train"

    def draw(self, split: str) -> tuple[list[int], int, tuple[int, ...]]:
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        for _ in range(10000):
            mapping = list(range(self.symbols))
            self.rng.shuffle(mapping)
            full_mapping = tuple(mapping)
            if self.partition(full_mapping) == split:
                break
        else:
            raise RuntimeError("Could not sample a mapping for the requested split")

        keys = self.rng.sample(range(self.symbols), self.examples)
        query_key = self.rng.choice(keys)
        prompt = [self.bos]
        for key in keys:
            prompt.extend((self.key_start + key, self.value_start + mapping[key]))
        prompt.extend((self.query, self.key_start + query_key))
        return prompt, self.value_start + mapping[query_key], full_mapping

    def batch(self, count: int, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if count < 1:
            raise ValueError("Batch must be nonempty")
        pairs = [self.draw(split) for _ in range(count)]
        prompts = torch.tensor([p for p, _, _ in pairs], dtype=torch.long)
        targets = torch.tensor([t for _, t, _ in pairs], dtype=torch.long)
        return prompts, targets

    def counterfactual(self, prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap the queried value with a different demonstration's value.

        Every resulting context is another valid one-to-one mapping. The
        alternative complete mapping may fall into the training partition;
        this is a controlled sensitivity probe, not a second held-out split.
        """
        if prompts.ndim != 2 or prompts.shape[1] != self.context_length:
            raise ValueError("Expected a batch of complete episode prompts")
        modified = prompts.clone()
        next_targets = []
        for row in modified:
            query_key = int(row[-1].item())
            query_index = next(
                offset for offset in range(1, 2 * self.examples + 1, 2)
                if int(row[offset].item()) == query_key
            )
            other_index = next(
                offset for offset in range(1, 2 * self.examples + 1, 2)
                if offset != query_index
            )
            original = row[query_index + 1].clone()
            row[query_index + 1] = row[other_index + 1]
            row[other_index + 1] = original
            next_targets.append(int(row[query_index + 1].item()))
        return modified, torch.tensor(next_targets, dtype=torch.long)

    def hide_values(self, prompts: torch.Tensor) -> torch.Tensor:
        """Ablation removes demonstrations' answers, not keys or query."""
        result = prompts.clone()
        result[:, 2:2 * self.examples + 1:2] = self.mask
        return result


@torch.inference_mode()
def evaluate_episodes(
    model: TinyTransformer, generator: EpisodeGenerator,
    inputs: torch.Tensor, targets: torch.Tensor, *, device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    """Report held-out accuracy with context and with demonstration values masked."""
    was_training = model.training
    model.eval()
    correct = 0
    without_examples = 0
    nll = 0.0
    counterfactual_correct = 0
    both_correct = 0
    for first in range(0, len(targets), batch_size):
        last = min(first + batch_size, len(targets))
        x = inputs[first:last].to(device)
        y = targets[first:last].to(device)
        logits = model(x)
        masked = model(generator.hide_values(x))
        changed, changed_targets = generator.counterfactual(inputs[first:last])
        changed_scores = model(changed.to(device))
        changed_ok = changed_scores.argmax(-1) == changed_targets.to(device)
        original_ok = logits.argmax(-1) == y
        both_correct += int((original_ok & changed_ok).sum().item())
        counterfactual_correct += int(changed_ok.sum().item())
        correct += int(original_ok.sum().item())
        without_examples += int((masked.argmax(-1) == y).sum().item())
        nll += float(F.cross_entropy(logits, y, reduction="sum").item())
    model.train(was_training)
    return {
        "heldout_accuracy": round(correct / len(targets), 5),
        "masked_context_accuracy": round(without_examples / len(targets), 5),
        "heldout_nll": round(nll / len(targets), 5),
        "counterfactual_accuracy": round(counterfactual_correct / len(targets), 5),
        "paired_counterfactual_success": round(both_correct / len(targets), 5),
    }


def run_episode_experiment(config: EpisodeConfig, output_dir: Path) -> dict:
    """Train from scratch and test on unseen complete key/value assignments."""
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    torch.manual_seed(config.seed)
    train = EpisodeGenerator(symbols=config.symbols, examples=config.examples, seed=config.seed)
    test = EpisodeGenerator(symbols=config.symbols, examples=config.examples, seed=config.seed + 7919)
    eval_inputs, eval_targets = test.batch(config.eval_episodes, "test")
    model = TinyTransformer(
        train.vocab_size, d_model=config.d_model, layers=config.layers,
        heads=config.heads, max_context=train.context_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    trajectory = []
    checkpoints = sorted({0, config.steps // 4, config.steps // 2, config.steps})

    def measure(step: int, loss: float | None) -> None:
        scores = evaluate_episodes(model, test, eval_inputs, eval_targets, device=device)
        trajectory.append({"step": step, "training_loss": loss, **scores})

    measure(0, None)
    for step in range(1, config.steps + 1):
        model.train()
        x, y = train.batch(config.batch_size, "train")
        logits = model(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step in checkpoints:
            measure(step, round(float(loss.item()), 5))

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.symbols <= 8:
        heldout_count = sum(
            train.partition(tuple(mapping)) == "test"
            for mapping in itertools.permutations(range(config.symbols))
        )
    else:
        heldout_count = None
    report = {
        "benchmark": "new-mapping in-context retrieval",
        "config": asdict(config),
        "parameter_count": model.parameter_count,
        "trajectory": trajectory,
        "final": trajectory[-1],
        "random_guess_accuracy": round(1 / config.symbols, 5),
        "symbolic_lookup_accuracy": 1.0,
        "complete_mapping_count": math.factorial(config.symbols),
        "heldout_complete_mapping_count": heldout_count,
        "split_rule": "complete permutations partitioned by CRC32 modulo 5; held-out demonstrations may share subsets",
        "limitations": [
            "Predicting a demonstrated value is associative retrieval, not rule induction or AGI.",
            "Train and held-out complete assignments are disjoint; individual example pairs can recur.",
            "Masked-value ablation is out of distribution, so it is not a calibrated causal estimate.",
            "Swapped contexts are valid mappings, but their full mappings are not held out by the original partition.",
            "Model weights do not change on evaluation examples.",
            "One random seed does not establish a robust improvement.",
        ],
    }
    (output_dir / "meta_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    torch.save({"state_dict": model.cpu().state_dict(), "config": asdict(config)}, output_dir / "meta.pt")
    return report


def run_episode_sweep(config: EpisodeConfig, seeds: list[int], output_dir: Path) -> dict:
    """Compare repeated runs using the same split rule, different random seeds."""
    config.validate()
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("A meta sweep needs at least two distinct seeds")
    runs = {}
    for seed in seeds:
        report = run_episode_experiment(
            replace(config, seed=seed), output_dir / f"seed_{seed}"
        )
        runs[str(seed)] = report["final"]

    summary = {}
    for metric in (
        "heldout_accuracy", "masked_context_accuracy", "heldout_nll",
        "counterfactual_accuracy", "paired_counterfactual_success",
    ):
        values = [runs[str(seed)][metric] for seed in seeds]
        summary[metric] = {
            "mean": round(mean(values), 5),
            "std": round(stdev(values), 5),
        }
    result = {
        "config": asdict(config), "seeds": seeds, "summary": summary,
        "runs": runs,
        "limits": "Unseen complete mappings, not novel algorithms or a measure of AGI.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
