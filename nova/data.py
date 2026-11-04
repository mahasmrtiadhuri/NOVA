"""Controlled synthetic facts with a stable, explicit vocabulary."""

from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class FactTask:
    task_id: int
    prompts: torch.Tensor  # [entities, tokens]
    targets: torch.Tensor  # [entities]


@dataclass(frozen=True)
class FactWorld:
    tasks: tuple[FactTask, ...]
    vocabulary: tuple[str, ...]
    token_to_id: dict[str, int]

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def encode_query(self, task_id: int, entity_id: int) -> torch.Tensor:
        if not 0 <= task_id < len(self.tasks):
            raise ValueError("Unknown task")
        if not 0 <= entity_id < self.tasks[task_id].targets.shape[0]:
            raise ValueError("Unknown entity")
        return self.tasks[task_id].prompts[entity_id].clone()


def make_world(*, tasks: int = 3, entities: int = 32, values: int = 8, seed: int = 7) -> FactWorld:
    """Create conflicting relation-specific lookups to measure interference.

    All tasks share entity and value tokens, but each relation defines its own
    random mapping. The same keys are evaluated throughout training: this tests
    retention of learned associations, not generalization to unseen facts.
    """
    if tasks < 2 or entities < 2 or values < 2:
        raise ValueError("Need at least 2 tasks, 2 entities and 2 values")
    vocabulary = (
        ("<bos>", "<query>")
        + tuple(f"relation_{i}" for i in range(tasks))
        + tuple(f"entity_{i}" for i in range(entities))
        + tuple(f"value_{i}" for i in range(values))
    )
    lookup = {token: idx for idx, token in enumerate(vocabulary)}
    rng = random.Random(seed)
    result = []
    for task_id in range(tasks):
        prompts = torch.tensor(
            [[lookup["<bos>"], lookup[f"relation_{task_id}"], lookup[f"entity_{entity}"], lookup["<query>"]]
             for entity in range(entities)],
            dtype=torch.long,
        )
        labels = [rng.randrange(values) for _ in range(entities)]
        targets = torch.tensor([lookup[f"value_{label}"] for label in labels], dtype=torch.long)
        result.append(FactTask(task_id, prompts, targets))
    return FactWorld(tuple(result), vocabulary, lookup)
