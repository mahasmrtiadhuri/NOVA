"""Held-out tasks for testing generalization, not just memory retention."""

from dataclasses import dataclass
from pathlib import Path
import random

import torch

from .data import FactTask, FactWorld


@dataclass(frozen=True)
class Benchmark:
    train: FactWorld
    test: FactWorld
    description: str
    context_length: int
    details: dict


def rule_value(task_id: int, left: int, right: int, values: int) -> int:
    """Deterministic rules with shared operands and task-specific targets."""
    operation = task_id % 4
    if operation == 0:
        return (left + right) % values
    if operation == 1:
        return (left - right) % values
    if operation == 2:
        return max(left, right) % values
    return int(left > right)


def make_rule_benchmark(
    *, tasks: int, operands: int, values: int, seed: int, test_fraction: float,
) -> Benchmark:
    """Reserve the same operand pairs across every task before training.

    Every operand occurs in the training split; a held-out pair never appears
    in any training task. Only combinations of known symbols are unseen.
    """
    if tasks < 2 or operands < 4 or values < 2:
        raise ValueError("Rules require >=2 tasks, >=4 operands, >=2 values")
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")

    vocab = (
        ("<bos>",)
        + tuple(f"rule_{i}" for i in range(tasks))
        + tuple(f"number_{i}" for i in range(operands))
        + tuple(f"value_{i}" for i in range(values))
    )
    lookup = {word: index for index, word in enumerate(vocab)}
    pairs = [(a, b) for a in range(operands) for b in range(operands)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    requested = max(1, round(len(pairs) * test_fraction))

    # Keep a candidate test pair only if all operand symbols remain in train.
    counts = {n: 2 * operands for n in range(operands)}
    heldout: list[tuple[int, int]] = []
    for left, right in pairs:
        removals = 2 if left == right else 1
        if len(heldout) >= requested:
            break
        if counts[left] <= removals or counts[right] <= removals:
            continue
        counts[left] -= removals if left == right else 1
        if left != right:
            counts[right] -= 1
        heldout.append((left, right))
    test_pairs = set(heldout)
    train_pairs = [pair for pair in pairs if pair not in test_pairs]
    if not test_pairs or not train_pairs:
        raise ValueError("Split produced an empty partition")

    def build(pairs_to_encode: list[tuple[int, int]]) -> FactWorld:
        result = []
        for task_id in range(tasks):
            prompts = torch.tensor([
                [lookup["<bos>"], lookup[f"rule_{task_id}"],
                 lookup[f"number_{left}"], lookup[f"number_{right}"]]
                for left, right in pairs_to_encode
            ], dtype=torch.long)
            targets = torch.tensor([
                lookup[f"value_{rule_value(task_id, left, right, values)}"]
                for left, right in pairs_to_encode
            ], dtype=torch.long)
            result.append(FactTask(task_id, prompts, targets))
        return FactWorld(tuple(result), vocab, lookup)

    return Benchmark(
        train=build(train_pairs), test=build(sorted(test_pairs)),
        description="Modular arithmetic and comparison rules; unseen operand pairs",
        context_length=4,
        details={
            "rules": ["add_mod", "subtract_mod", "max_mod", "greater_than"],
            "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
            "test_fraction_actual": round(len(test_pairs) / len(pairs), 6),
            "split_seed": seed,
        },
    )


def make_text_benchmark(
    path: str | Path, *, tasks: int, context_length: int,
    test_fraction: float, max_chars: int = 12000,
) -> Benchmark:
    """Next-character prediction on disjoint chronological text segments.

    Each segment is one sequential training task. Splits are made on raw
    character positions before generating sliding windows, so no window can
    straddle the train/test boundary. This is a small LM benchmark, not AGI.
    """
    if tasks < 2 or context_length < 2:
        raise ValueError("Text requires >=2 tasks and context_length >=2")
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    text = Path(path).read_text(encoding="utf-8")[:max_chars]
    if len(text) < tasks * (context_length + 2) * 4:
        raise ValueError("Corpus too short for the requested task count and context length")
    alphabet = tuple(sorted(set(text)))
    if len(alphabet) < 2:
        raise ValueError("Corpus must contain at least two unique characters")
    lookup = {char: i for i, char in enumerate(alphabet)}
    encoded = [lookup[char] for char in text]
    train_tasks: list[FactTask] = []
    test_tasks: list[FactTask] = []
    partition_sizes = []
    for task_id in range(tasks):
        start = task_id * len(encoded) // tasks
        end = (task_id + 1) * len(encoded) // tasks
        segment = encoded[start:end]
        train_len = int(len(segment) * (1 - test_fraction))
        train_chars, test_chars = segment[:train_len], segment[train_len:]

        def windows(chars: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
            inputs = [chars[i:i + context_length] for i in range(len(chars) - context_length)]
            outputs = [chars[i + context_length] for i in range(len(chars) - context_length)]
            if not inputs:
                raise ValueError("Not enough text in one of the partitions")
            return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)

        train_prompts, train_targets = windows(train_chars)
        test_prompts, test_targets = windows(test_chars)
        train_tasks.append(FactTask(task_id, train_prompts, train_targets))
        test_tasks.append(FactTask(task_id, test_prompts, test_targets))
        partition_sizes.append({"train_examples": len(train_targets), "test_examples": len(test_targets)})
    return Benchmark(
        train=FactWorld(tuple(train_tasks), alphabet, lookup),
        test=FactWorld(tuple(test_tasks), alphabet, lookup),
        description="Character-level next-token prediction on held-out text segments",
        context_length=context_length,
        details={"corpus": str(path), "characters_used": len(text),
                 "vocab_size": len(alphabet), "partition_sizes": partition_sizes,
                 "splitting": "Contiguous train/test character spans per task; no shared sliding windows"},
    )
