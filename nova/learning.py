"""Sequential fine-tuning, random replay, and a provisional NOVA variant."""

from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.nn import functional as F

from .data import FactWorld
from .model import TinyTransformer


@dataclass
class Experience:
    prompt: torch.Tensor
    target: torch.Tensor
    priority: float = 1.0


class Memory:
    """Bounded rehearsal memory; data remain on CPU until sampled."""

    def __init__(self, capacity: int = 128, *, seed: int = 7) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.items: list[Experience] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def remember(self, prompts: torch.Tensor, targets: torch.Tensor) -> None:
        """Reservoir sampling gives each observed example equal inclusion odds."""
        for prompt, target in zip(prompts, targets):
            self.seen += 1
            item = Experience(prompt.detach().cpu().clone(), target.detach().cpu().clone())
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                position = self.rng.randrange(self.seen)
                if position < self.capacity:
                    self.items[position] = item

    def sample(self, batch_size: int, *, prioritized: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.items:
            raise ValueError("Can't sample empty memory")
        weights = [max(item.priority, 0.01) for item in self.items] if prioritized else None
        indices = self.rng.choices(range(len(self.items)), weights=weights, k=batch_size)
        return (
            torch.stack([self.items[i].prompt for i in indices]),
            torch.stack([self.items[i].target for i in indices]),
        )

    @torch.no_grad()
    def update_priorities(self, model: TinyTransformer, *, device: torch.device) -> None:
        """Rehearse old associations that the current model is getting wrong."""
        if not self.items:
            return
        was_training = model.training
        model.eval()
        prompts = torch.stack([item.prompt for item in self.items]).to(device)
        targets = torch.stack([item.target for item in self.items]).to(device)
        losses = F.cross_entropy(model(prompts), targets, reduction="none").cpu().tolist()
        for item, loss in zip(self.items, losses):
            item.priority = min(20.0, 0.1 + float(loss))
        model.train(was_training)


class Consolidation:
    """EWC-style penalty using squared minibatch gradients as importance."""

    def __init__(self) -> None:
        self.snapshots: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

    def capture(self, model: TinyTransformer, prompts: torch.Tensor, targets: torch.Tensor) -> None:
        was_training = model.training
        model.eval()
        fisher = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
        # Small batches give a cheap, approximate diagonal importance estimate.
        for first in range(0, len(targets), 8):
            last = min(first + 8, len(targets))
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(prompts[first:last]), targets[first:last])
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square() * (last - first)
        for name in fisher:
            fisher[name] /= len(targets)
        anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.snapshots.append((anchor, fisher))
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    def penalty(self, model: TinyTransformer) -> torch.Tensor:
        penalty = next(model.parameters()).new_zeros(())
        for anchor, fisher in self.snapshots:
            for name, parameter in model.named_parameters():
                penalty = penalty + (fisher[name] * (parameter - anchor[name]).square()).sum()
        return penalty


def train_one_task(
    model: TinyTransformer, world: FactWorld, task_id: int, *,
    method: str, steps: int, batch_size: int, learning_rate: float,
    replay_fraction: float, memory: Memory, consolidation: Consolidation,
    consolidation_strength: float, device: torch.device, seed: int,
) -> list[float]:
    """All methods use the same number of optimizer steps and total batch size."""
    if method not in {"finetune", "replay", "nova"}:
        raise ValueError(f"Unknown method: {method}")
    if steps < 1 or batch_size < 2:
        raise ValueError("Need steps >= 1 and batch_size >= 2")
    if not 0 <= replay_fraction < 1:
        raise ValueError("replay_fraction must be in [0, 1)")
    task = world.tasks[task_id]
    generator = torch.Generator().manual_seed(seed + task_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        n_replay = 0
        if method != "finetune" and memory.items:
            n_replay = min(batch_size - 1, round(batch_size * replay_fraction))
        n_new = batch_size - n_replay
        indices = torch.randint(len(task.targets), (n_new,), generator=generator)
        prompts, targets = task.prompts[indices], task.targets[indices]
        if n_replay:
            old_prompts, old_targets = memory.sample(n_replay, prioritized=(method == "nova"))
            prompts = torch.cat([prompts, old_prompts])
            targets = torch.cat([targets, old_targets])
        logits = model(prompts.to(device))
        loss = F.cross_entropy(logits, targets.to(device))
        if method == "nova" and consolidation.snapshots:
            loss = loss + consolidation_strength * consolidation.penalty(model)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if method == "nova" and memory.items and (step + 1) % max(1, steps // 5) == 0:
            memory.update_priorities(model, device=device)
    if method == "nova":
        consolidation.capture(model, task.prompts.to(device), task.targets.to(device))
    if method != "finetune":
        memory.remember(task.prompts, task.targets)
    return losses


@torch.inference_mode()
def evaluate(model: TinyTransformer, world: FactWorld, *, device: torch.device) -> list[float]:
    was_training = model.training
    model.eval()
    result = []
    for task in world.tasks:
        predictions = model(task.prompts.to(device)).argmax(dim=-1)
        accuracy = (predictions == task.targets.to(device)).float().mean()
        result.append(round(float(accuracy.item()), 5))
    model.train(was_training)
    return result
