import json

import pytest
import torch

from nova.data import make_world
from nova.experiment import ExperimentConfig, run_experiment
from nova.learning import Consolidation, Memory, evaluate, train_one_task
from nova.model import TinyTransformer


def test_world_is_seeded_and_tasks_conflict():
    first = make_world(tasks=3, entities=20, values=6, seed=11)
    second = make_world(tasks=3, entities=20, values=6, seed=11)
    assert all(torch.equal(a.targets, b.targets) for a, b in zip(first.tasks, second.tasks))
    assert not torch.equal(first.tasks[0].targets, first.tasks[1].targets)
    assert first.tasks[0].prompts.shape == (20, 4)
    assert first.encode_query(1, 0).tolist() == first.tasks[1].prompts[0].tolist()
    with pytest.raises(ValueError):
        first.encode_query(9, 0)


def test_causal_attention_ignores_future_tokens():
    torch.manual_seed(0)
    model = TinyTransformer(40, d_model=32, layers=1, heads=4)
    model.eval()
    one = model(torch.tensor([[1]]))
    outputs = model(torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5]]), return_all=True)
    assert one.shape == (1, 40)
    assert outputs.shape == (2, 4, 40)
    # Changing the fourth token cannot change logits at earlier positions.
    assert torch.allclose(outputs[0, :3], outputs[1, :3], atol=1e-5)
    assert torch.isfinite(outputs).all()
    with pytest.raises(ValueError):
        model(torch.ones(1, 9, dtype=torch.long))


def test_memory_is_bounded_and_priority_update_works():
    world = make_world(tasks=2, entities=8, values=3)
    memory = Memory(capacity=3, seed=4)
    memory.remember(world.tasks[0].prompts, world.tasks[0].targets)
    assert len(memory.items) == 3
    assert memory.seen == 8
    x, y = memory.sample(5)
    assert x.shape == (5, 4) and y.shape == (5,)
    model = TinyTransformer(world.vocab_size, d_model=32, layers=1, heads=4)
    memory.update_priorities(model, device=torch.device("cpu"))
    assert all(item.priority > 0 for item in memory.items)


def test_training_changes_weights_and_evaluates():
    torch.manual_seed(3)
    world = make_world(tasks=2, entities=8, values=3)
    model = TinyTransformer(world.vocab_size, d_model=32, layers=1, heads=4)
    original = model.head.weight.detach().clone()
    memory = Memory(8)
    consolidation = Consolidation()
    for task in range(2):
        losses = train_one_task(
            model, world, task, method="nova", steps=3, batch_size=4,
            learning_rate=0.01, replay_fraction=0.25, memory=memory,
            consolidation=consolidation, consolidation_strength=1.0,
            device=torch.device("cpu"), seed=7,
        )
        assert len(losses) == 3
    assert len(consolidation.snapshots) == 2
    assert not torch.equal(original, model.head.weight.detach())
    accuracy = evaluate(model, world, device=torch.device("cpu"))
    assert len(accuracy) == 2 and all(0 <= value <= 1 for value in accuracy)


def test_experiment_writes_metrics_and_checkpoints(tmp_path):
    cfg = ExperimentConfig(
        tasks=2, entities=6, values=3, steps=2, batch_size=4,
        d_model=32, layers=1, heads=4,
    )
    results = run_experiment(cfg, tmp_path)
    assert set(results["results"]) == {"finetune", "replay", "nova"}
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["results"]["nova"]["parameter_count"] > 0
    assert (tmp_path / "nova.pt").is_file()
    checkpoint = torch.load(tmp_path / "nova.pt", weights_only=True)
    assert "state_dict" in checkpoint


def test_sweep_outputs_paired_seed_metrics(tmp_path):
    from nova.sweep import run_sweep

    cfg = ExperimentConfig(
        tasks=2, entities=6, values=3, steps=2, batch_size=4,
        d_model=32, layers=1, heads=4,
    )
    report = run_sweep(cfg, [3, 11], tmp_path)
    assert report["seeds"] == [3, 11]
    assert set(report["summary"]) == {"finetune", "replay", "nova"}
    assert "nova_minus_replay" in report["paired_accuracy_differences"]
    assert (tmp_path / "seed_3" / "metrics.json").exists()
    assert not (tmp_path / "seed_3" / "nova.pt").exists()
    assert (tmp_path / "summary.json").exists()
    with pytest.raises(ValueError):
        run_sweep(cfg, [3, 3], tmp_path)
