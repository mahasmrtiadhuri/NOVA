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


def test_rule_split_is_seeded_and_heldout_pairs_are_never_trained():
    from nova.benchmarks import make_rule_benchmark, rule_value

    first = make_rule_benchmark(tasks=4, operands=8, values=5, seed=31, test_fraction=0.25)
    second = make_rule_benchmark(tasks=4, operands=8, values=5, seed=31, test_fraction=0.25)
    for train_task, test_task, second_test in zip(first.train.tasks, first.test.tasks, second.test.tasks):
        train_prompts = {tuple(row) for row in train_task.prompts.tolist()}
        test_prompts = {tuple(row) for row in test_task.prompts.tolist()}
        assert not train_prompts.intersection(test_prompts)
        assert torch.equal(test_task.prompts, second_test.prompts)
        assert len(train_prompts) + len(test_prompts) == 64
    assert first.details["test_pairs"] > 0
    lookup = first.test.token_to_id
    task = first.test.tasks[0]
    a = int(first.test.vocabulary[int(task.prompts[0, 2])].split("_")[1])
    b = int(first.test.vocabulary[int(task.prompts[0, 3])].split("_")[1])
    assert task.targets[0].item() == lookup[f"value_{rule_value(0, a, b, 5)}"]


def test_rule_experiment_reports_heldout_and_joint_reference(tmp_path):
    cfg = ExperimentConfig(
        dataset="rules", tasks=2, entities=6, values=3,
        steps=2, batch_size=4, d_model=32, layers=1, heads=4,
    )
    data = run_experiment(cfg, tmp_path)
    assert set(data["results"]) == {"finetune", "replay", "nova", "joint_oracle"}
    assert data["benchmark"]["train_pairs"] > 0
    for result in data["results"].values():
        assert 0 <= result["final_heldout_accuracy"] <= 1
        assert result["final_heldout_nll"] > 0
    assert "mean_heldout_forgetting" in data["results"]["nova"]
    assert "mean_heldout_forgetting" not in data["results"]["joint_oracle"]
    assert (tmp_path / "joint_oracle.pt").exists()


def test_text_corpus_split_is_bounded_and_has_out_of_sample_windows(tmp_path):
    from nova.benchmarks import make_text_benchmark

    file = tmp_path / "corpus.txt"
    file.write_text("The city wakes slowly at sunrise. " * 40, encoding="utf-8")
    benchmark = make_text_benchmark(
        file, tasks=2, context_length=8, test_fraction=0.25,
    )
    assert benchmark.context_length == 8
    for train, test in zip(benchmark.train.tasks, benchmark.test.tasks):
        assert train.prompts.shape[1] == test.prompts.shape[1] == 8
        assert len(train.targets) > len(test.targets) > 0
    cfg = ExperimentConfig(
        dataset="text", corpus=str(file), tasks=2, context_length=8,
        steps=2, batch_size=4, d_model=32, layers=1, heads=4,
    )
    report = run_experiment(cfg, tmp_path / "results", save_checkpoints=False)
    assert report["results"]["nova"]["final_heldout_nll"] > 0
    assert not (tmp_path / "results" / "nova.pt").exists()
    with pytest.raises(ValueError):
        make_text_benchmark(file, tasks=2, context_length=1024, test_fraction=0.25)


def test_rule_sweep_paired_heldout_metrics(tmp_path):
    from nova.sweep import run_sweep

    cfg = ExperimentConfig(
        dataset="rules", tasks=2, entities=5, values=3, steps=2,
        batch_size=4, d_model=32, layers=1, heads=4,
    )
    result = run_sweep(cfg, [4, 9], tmp_path)
    assert "final_heldout_accuracy_mean" in result["summary"]["nova"]
    assert "final_heldout_nll_mean" in result["summary"]["joint_oracle"]
    assert result["paired_accuracy_differences"]["nova_minus_replay"]["metric"] == "final_heldout_accuracy"


def test_meta_episode_is_consistent_and_mapping_families_are_disjoint():
    from nova.episodes import EpisodeGenerator

    generator = EpisodeGenerator(symbols=7, examples=4, seed=13)
    train_mappings, test_mappings = set(), set()
    for split, collection in (("train", train_mappings), ("test", test_mappings)):
        for _ in range(100):
            prompt, target, mapping = generator.draw(split)
            collection.add(mapping)
            assert generator.partition(mapping) == split
            assert len(prompt) == 11
            query_key = prompt[-1] - generator.key_start
            displayed = {
                prompt[pos] - generator.key_start: prompt[pos + 1]
                for pos in range(1, 9, 2)
            }
            assert target == displayed[query_key]
    assert train_mappings.isdisjoint(test_mappings)


def test_meta_counterfactual_changes_answer_not_query():
    from nova.episodes import EpisodeGenerator

    generator = EpisodeGenerator(symbols=8, examples=4, seed=19)
    prompts, targets = generator.batch(16, "test")
    changed, changed_targets = generator.counterfactual(prompts)
    assert changed.shape == prompts.shape
    assert torch.equal(changed[:, 1::2], prompts[:, 1::2])
    assert (changed_targets != targets).all()
    for row, label in zip(changed, changed_targets):
        query_key = row[-1].item()
        assert any(row[i].item() == query_key and row[i + 1].item() == label.item()
                   for i in range(1, 9, 2))


def test_meta_smoke_run_and_outputs(tmp_path):
    from nova.episodes import EpisodeConfig, run_episode_experiment

    config = EpisodeConfig(
        symbols=6, examples=3, steps=2, batch_size=4,
        d_model=32, layers=1, heads=4, eval_episodes=20,
    )
    report = run_episode_experiment(config, tmp_path)
    assert report["benchmark"] == "new-mapping in-context retrieval"
    assert report["final"]["step"] == 2
    assert 0 <= report["final"]["heldout_accuracy"] <= 1
    assert 0 <= report["final"]["paired_counterfactual_success"] <= 1
    assert (tmp_path / "meta_metrics.json").exists()
    assert (tmp_path / "meta.pt").exists()
    checkpoint = torch.load(tmp_path / "meta.pt", weights_only=True)
    assert "state_dict" in checkpoint


def test_meta_rejects_invalid_config(tmp_path):
    from nova.episodes import EpisodeConfig, run_episode_experiment

    with pytest.raises(ValueError, match="examples"):
        run_episode_experiment(EpisodeConfig(examples=9), tmp_path)


def test_meta_sweep_reports_mean_and_rejects_duplicates(tmp_path):
    from nova.episodes import EpisodeConfig, run_episode_sweep

    config = EpisodeConfig(
        symbols=4, examples=3, steps=1, batch_size=4,
        d_model=32, layers=1, heads=4, eval_episodes=12,
    )
    report = run_episode_sweep(config, [3, 9], tmp_path)
    assert report["seeds"] == [3, 9]
    assert "std" in report["summary"]["heldout_accuracy"]
    assert (tmp_path / "meta_summary.json").exists()
    with pytest.raises(ValueError):
        run_episode_sweep(config, [3, 3], tmp_path)


def test_direct_lookup_matches_context_and_counterfactual():
    from nova.episodes import EpisodeGenerator
    from nova.efficient import lookup_demonstrated_value

    generator = EpisodeGenerator(symbols=6, examples=3, seed=23)
    prompts, expected = generator.batch(32, "test")
    assert torch.equal(lookup_demonstrated_value(prompts), expected)
    changed, changed_expected = generator.counterfactual(prompts)
    assert torch.equal(lookup_demonstrated_value(changed), changed_expected)
    with pytest.raises(ValueError, match="match exactly one"):
        malformed = prompts.clone()
        malformed[0, -1] = generator.key_start + generator.symbols + 12
        lookup_demonstrated_value(malformed)
    with pytest.raises(ValueError, match="token IDs"):
        lookup_demonstrated_value(prompts[:, :6])


def test_lite_path_uses_no_training_or_checkpoints(tmp_path):
    from nova.episodes import EpisodeConfig
    from nova.efficient import run_lite_baseline

    config = EpisodeConfig(symbols=4, examples=3, steps=2, eval_episodes=32)
    report = run_lite_baseline(config, tmp_path)
    assert report["heldout_accuracy"] == 1.0
    assert report["counterfactual_accuracy"] == 1.0
    assert report["optimizer_steps"] == report["parameters"] == 0
    assert (tmp_path / "lite_metrics.json").exists()
    assert not list(tmp_path.glob("*.pt"))


def test_early_stop_uses_training_family_and_restores_best_weights(tmp_path):
    from nova.episodes import EpisodeConfig, run_episode_experiment

    config = EpisodeConfig(
        symbols=4, examples=3, steps=30, batch_size=4, d_model=32,
        layers=1, heads=4, eval_episodes=12,
        patience=1, validate_every=2, min_delta=1000.0,
    )
    report = run_episode_experiment(config, tmp_path)
    usage = report["resource_usage"]
    assert usage["stop_reason"] == "training_partition_plateau"
    assert usage["executed_optimizer_steps"] == 4
    assert usage["steps_avoided"] == 26
    assert usage["weights_selected_at_step"] == 2
    assert report["final"]["selected_weight_step"] == 2
    checkpoint = torch.load(tmp_path / "meta.pt", weights_only=True)
    assert checkpoint["config"]["patience"] == 1


def test_budgeted_training_and_exact_reuse(tmp_path):
    from nova.episodes import EpisodeConfig, run_episode_experiment

    config = EpisodeConfig(
        symbols=4, examples=3, steps=10, batch_size=4,
        d_model=32, layers=1, heads=4, eval_episodes=12,
        max_seconds=0.000001,
    )
    first = run_episode_experiment(config, tmp_path, reuse=True)
    assert first["cache_hit"] is False
    assert first["resource_usage"]["stop_reason"] == "time_budget"
    assert first["resource_usage"]["executed_optimizer_steps"] == 1
    checkpoint_time = (tmp_path / "meta.pt").stat().st_mtime_ns
    second = run_episode_experiment(config, tmp_path, reuse=True)
    assert second["cache_hit"] is True
    assert (tmp_path / "meta.pt").stat().st_mtime_ns == checkpoint_time
    changed = run_episode_experiment(
        EpisodeConfig(**{**vars(config), "steps": 11}), tmp_path, reuse=True,
    )
    assert changed["cache_hit"] is False


def test_rejects_invalid_compute_limits():
    from nova.episodes import EpisodeConfig

    with pytest.raises(ValueError, match="patience"):
        EpisodeConfig(patience=-1).validate()
    with pytest.raises(ValueError, match="max_seconds"):
        EpisodeConfig(max_seconds=0).validate()
    with pytest.raises(ValueError, match="validate_every"):
        EpisodeConfig(validate_every=0).validate()
