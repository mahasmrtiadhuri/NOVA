import json
from dataclasses import replace

import pytest

from nova.molt import Example, MoltRouter, Program, compile_skill
from nova.molt.benchmark import DemoConfig, run_demo
from nova.molt.energy import EnergyCounter
from nova.molt.store import load_skill, save_skill


def example_data(fn, modulus=8):
    points = [(a, b) for a in range(modulus) for b in range(modulus)]
    return [Example(a, b, fn(a, b)) for a, b in points[:30]], [
        Example(a, b, fn(a, b)) for a, b in points[30:52]
    ], [Example(a, b, fn(a, b)) for a, b in points[52:]]


def test_compiler_learns_modular_rule_on_heldout_inputs():
    train, valid, test = example_data(lambda a, b: (a + b) % 8)
    result = compile_skill("sum", train, valid, modulus=8)
    assert result.status == "compiled"
    assert result.skill is not None
    assert result.skill.validation_accuracy == 1.0
    assert all(result.skill.predict(ex.left, ex.right) == ex.output for ex in test)
    assert result.skill.program.op == "add_mod"


def test_affine_skill_is_learned_from_examples():
    train, valid, test = example_data(lambda a, b: (3 * a + 2 * b + 1) % 8)
    result = compile_skill("affine", train, valid, modulus=8)
    assert result.skill is not None
    assert result.skill.program.op == "affine_mod"
    assert all(result.skill.predict(ex.left, ex.right) == ex.output for ex in test)


def test_compiler_rejects_unsupported_and_ambiguous_rules():
    # Rule deliberately outside the DSL, not another affine function.
    train, valid, _ = example_data(lambda a, b: (a * b + a + (a & b)) % 8)
    result = compile_skill("outside", train, valid, modulus=8)
    assert result.skill is None and result.status in {"no_valid_program", "ambiguous_program"}
    a, b, _ = example_data(lambda x, y: max(x, y))
    limited = compile_skill("max", a, b, modulus=8, max_candidates=2)
    assert limited.skill is None and limited.status == "search_budget_exhausted"


def test_compiler_prevents_validation_leakage_and_bad_inputs():
    train, valid, _ = example_data(lambda a, b: max(a, b))
    with pytest.raises(ValueError, match="disjoint"):
        compile_skill("bad", train, valid + [train[0]], modulus=8)
    with pytest.raises(ValueError, match="Duplicate"):
        compile_skill("bad", train + [train[0]], valid, modulus=8)
    with pytest.raises(ValueError, match="Invalid"):
        compile_skill("bad", train + [Example(99, 1, 0)], valid, modulus=8)
    assert compile_skill("short", train[:2], valid, modulus=8).status == "insufficient_examples"


def test_router_falls_back_on_unknown_or_outside_domain():
    train, valid, _ = example_data(lambda a, b: max(a, b))
    skill = compile_skill("max", train, valid, modulus=8).skill
    assert skill is not None
    calls = []

    def fallback(name, a, b):
        calls.append((name, a, b))
        return max(a, b)

    router = MoltRouter(fallback)
    router.install(skill)
    assert router.predict("max", 5, 1) == 5
    assert router.predict("max", 9, 1) == 9
    assert router.predict("unknown", 2, 7) == 7
    assert router.stats.paths["compiled"] == 1
    assert router.stats.paths["fallback"] == 2
    assert len(calls) == 2


def test_router_spotcheck_disables_bad_skill():
    train, valid, _ = example_data(lambda a, b: max(a, b))
    skill = compile_skill("max", train, valid, modulus=8).skill
    assert skill is not None
    router = MoltRouter(
        lambda task, a, b: a + b,
        verifier=lambda task, a, b: a + b,
        verify_every=1,
    )
    router.install(skill)
    assert router.predict("max", 1, 2) == 3
    assert "max" not in router.skills
    assert router.stats.paths["verification_failures"] == 1
    assert router.predict("max", 1, 3) == 4
    with pytest.raises(ValueError, match="verify_every"):
        MoltRouter(lambda task, a, b: a + b, verify_every=2)


def test_json_skill_store_revalidates_and_disallows_unknown_ops(tmp_path):
    train, valid, test = example_data(lambda a, b: a ^ b)
    skill = compile_skill("xor", train, valid, modulus=8).skill
    assert skill is not None
    output = tmp_path / "skill.json"
    save_skill(skill, output)
    restored = load_skill(output)
    assert all(restored.predict(ex.left, ex.right) == ex.output for ex in test)
    data = json.loads(output.read_text())
    data["program"]["op"] = "__import__"
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unknown operation"):
        load_skill(output)
    data["program"]["op"] = "max"
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="failed"):
        load_skill(output)


def test_program_rejects_invalid_inputs_and_arbitrary_parameters():
    with pytest.raises(ValueError):
        Program("add_mod", 8, (1, 0, 0))
    with pytest.raises(TypeError):
        Program("max", 8)(True, 1)
    with pytest.raises(ValueError):
        Program("max", 8)(8, 1)


def test_energy_counter_wrap_and_no_measurement(tmp_path):
    p = tmp_path / "energy_uj"
    p.write_text("7")
    meter = EnergyCounter(p, 100)
    assert meter.read() == 7
    assert meter.delta_joules(98, 3) == pytest.approx(5e-6)
    assert meter.delta_joules(None, 3) is None
    p.unlink()
    assert meter.read() is None


def test_neural_to_symbolic_demo_creates_reproducible_report(tmp_path):
    report = run_demo(DemoConfig(operator="max", modulus=8, steps=100, seed=7,
                                 benchmark_requests=24, output_dir=str(tmp_path)))
    assert report["training"]["teacher_parameters"] > 1000
    assert report["splits"]["train"] > 0
    assert (tmp_path / "report.json").exists()
    assert 0 <= report["results"]["molt_router"]["accuracy"] <= 1
    assert "handwritten_python" in report["results"]
    if report["compiler"]["status"] == "compiled":
        assert (tmp_path / "skill.json").exists()
    with pytest.raises(ValueError):
        run_demo(replace(DemoConfig(output_dir=str(tmp_path)), operator="invalid"))
