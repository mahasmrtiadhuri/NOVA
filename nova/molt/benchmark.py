"""Small reproducible teacher-to-skill compression benchmark.

The teacher is a TinyTransformer trained on the full finite task domain.
Training the teacher on all inputs means that held-out inputs are held out
from *compilation*, not from the original teacher's supervised learning.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter, perf_counter_ns
from typing import Callable

import torch
from torch.nn import functional as F

from nova.model import TinyTransformer

from .compiler import compile_skill
from .energy import EnergyCounter
from .language import Example
from .runtime import MoltRouter
from .store import save_skill


OPERATORS: dict[str, Callable[[int, int, int], int]] = {
    "add_mod": lambda a, b, m: (a + b) % m,
    "xor": lambda a, b, m: (a ^ b) % m,
    "max": lambda a, b, m: max(a, b),
    "affine_mod": lambda a, b, m: (3 * a + 2 * b + 1) % m,
}


@dataclass(frozen=True)
class DemoConfig:
    operator: str = "max"
    modulus: int = 8
    steps: int = 300
    seed: int = 7
    benchmark_requests: int = 256
    output_dir: str = "runs/molt"


class NeuralTeacher:
    def __init__(self, modulus: int, seed: int) -> None:
        torch.manual_seed(seed)
        self.modulus = modulus
        self.model = TinyTransformer(1 + 2 * modulus, d_model=48, layers=1, heads=4, max_context=3)
        self.model.eval()

    def _tokens(self, points: list[tuple[int, int]]) -> torch.Tensor:
        return torch.tensor([[0, 1 + a, 1 + b] for a, b in points], dtype=torch.long)

    def fit(self, oracle: Callable[[int, int, int], int], steps: int) -> dict:
        points = [(a, b) for a in range(self.modulus) for b in range(self.modulus)]
        x = self._tokens(points)
        y = torch.tensor([1 + self.modulus + oracle(a, b, self.modulus) for a, b in points])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.012, weight_decay=0.0)
        start = perf_counter()
        self.model.train()
        # The teacher learns every pair; the compiler later receives only a subset.
        for _ in range(steps):
            prediction = self.model(x)
            loss = F.cross_entropy(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.model.eval()
        with torch.inference_mode():
            all_predictions = self.model(x).argmax(-1)
        exact = int((all_predictions == y).sum().item())
        return {
            "train_seconds": round(perf_counter() - start, 6),
            "teacher_full_domain_accuracy": round(exact / len(points), 6),
            "teacher_parameters": self.model.parameter_count,
            "teacher_parameter_bytes_fp32": self.model.parameter_count * 4,
        }

    def __call__(self, task: str, left: int, right: int) -> int:
        if not 0 <= left < self.modulus or not 0 <= right < self.modulus:
            raise ValueError("Input outside neural teacher's domain")
        with torch.inference_mode():
            logits = self.model(self._tokens([(left, right)]))[0]
            # Only output tokens are legitimate answers.
            result = int(logits[1 + self.modulus:1 + 2 * self.modulus].argmax().item())
        return result


def _time_calls(fn: Callable[[int, int], int], points: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    results = []
    ns = []
    for a, b in points:
        start = perf_counter_ns()
        value = fn(a, b)
        ns.append(perf_counter_ns() - start)
        results.append(value)
    return results, ns


def _latency_ms(samples_ns: list[int]) -> dict:
    if not samples_ns:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    ordered = sorted(samples_ns)
    return {
        "mean_ms": round(mean(samples_ns) / 1e6, 7),
        "median_ms": round(median(samples_ns) / 1e6, 7),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))] / 1e6, 7),
    }


def run_demo(config: DemoConfig) -> dict:
    if config.operator not in OPERATORS:
        raise ValueError(f"Unknown task: {config.operator}")
    if not 5 <= config.modulus <= 12:
        raise ValueError("The neural demo supports moduli between 5 and 12")
    if config.steps < 1 or not 1 <= config.benchmark_requests <= 50_000:
        raise ValueError("steps and benchmark_requests must be positive and bounded")

    # Avoid nondeterministic thread oversubscription in tiny CPU experiments.
    # Machine-dependent wall times are descriptive only, not portable benchmarks.
    torch.set_num_threads(1)
    rng = random.Random(config.seed)
    oracle = OPERATORS[config.operator]
    domain = [(a, b) for a in range(config.modulus) for b in range(config.modulus)]
    shuffled = list(domain)
    rng.shuffle(shuffled)
    train_n = max(8, int(0.4 * len(domain)))
    validation_n = max(12, int(0.3 * len(domain)))
    train_pairs = shuffled[:train_n]
    validation_pairs = shuffled[train_n:train_n + validation_n]
    test_pairs = shuffled[train_n + validation_n:]
    if not test_pairs:
        raise ValueError("Not enough test inputs")

    teacher = NeuralTeacher(config.modulus, config.seed)
    training = teacher.fit(oracle, config.steps)

    # The program is induced from *the neural teacher's outputs*.
    # Independent oracle labels are used only for the final benchmark report.
    def teacher_samples(points: list[tuple[int, int]]) -> list[Example]:
        return [Example(a, b, teacher(config.operator, a, b)) for a, b in points]

    start_compile = perf_counter()
    compiled = compile_skill(
        config.operator, teacher_samples(train_pairs), teacher_samples(validation_pairs),
        modulus=config.modulus, max_candidates=5000,
    )
    compile_seconds = perf_counter() - start_compile
    router = MoltRouter(teacher)
    if compiled.skill is not None:
        router.install(compiled.skill)

    # Warm up inference to reduce initialization effects on timings.
    for a, b in test_pairs[:min(5, len(test_pairs))]:
        teacher(config.operator, a, b)
    requests = [test_pairs[i % len(test_pairs)] for i in range(config.benchmark_requests)]
    expected = [oracle(a, b, config.modulus) for a, b in requests]
    power = EnergyCounter.discover()

    before = power.read() if power else None
    teacher_output, neural_ns = _time_calls(lambda a, b: teacher(config.operator, a, b), requests)
    after = power.read() if power else None
    teacher_joules = power.delta_joules(before, after) if power else None

    before = power.read() if power else None
    routed_output, compiled_ns = _time_calls(lambda a, b: router.predict(config.operator, a, b), requests)
    after = power.read() if power else None
    routed_joules = power.delta_joules(before, after) if power else None

    handwritten_output, handwritten_ns = _time_calls(lambda a, b: oracle(a, b, config.modulus), requests)
    mean_neural = mean(neural_ns)
    mean_routed = mean(compiled_ns)
    saved_ns_per_query = mean_neural - mean_routed
    break_even = (
        int((compile_seconds * 1e9) / saved_ns_per_query) + 1
        if compiled.skill and saved_ns_per_query > 0 else None
    )
    report = {
        "benchmark": "MOLT finite-integer skill compression",
        "config": asdict(config),
        "training": training,
        "splits": {"train": len(train_pairs), "validation": len(validation_pairs), "test": len(test_pairs)},
        "compiler": {
            "status": compiled.status,
            "candidate_evaluations": compiled.checked_candidates,
            "wall_seconds": round(compile_seconds, 6),
            "skill": compiled.skill.program.to_dict() if compiled.skill else None,
            "certificate_validation_examples": compiled.skill.validation_count if compiled.skill else 0,
        },
        "results": {
            "neural_reference": {
                "accuracy": round(sum(a == b for a, b in zip(teacher_output, expected)) / len(requests), 6),
                **_latency_ms(neural_ns),
                "joules_unisolated_cpu_package": teacher_joules,
            },
            "molt_router": {
                "accuracy": round(sum(a == b for a, b in zip(routed_output, expected)) / len(requests), 6),
                **_latency_ms(compiled_ns),
                "joules_unisolated_cpu_package": routed_joules,
                "routing": dict(router.stats.paths),
            },
            "handwritten_python": {
                "accuracy": round(sum(a == b for a, b in zip(handwritten_output, expected)) / len(requests), 6),
                **_latency_ms(handwritten_ns),
            },
        },
        "estimated_compile_break_even_requests": break_even,
        "measurement_notes": [
            "Teacher was trained on the complete finite task domain; the test split was unseen by compilation only.",
            "Neural reference is intentionally tiny, not a competitive optimized production model.",
            "All performance numbers are single-threaded CPU request latencies on this machine; no uncertainty interval.",
            "Handwritten Python is a relevant lower-overhead baseline; compiled symbolic rules are not automatically better than known code.",
            "Break-even amortizes compilation only against subsequent neural inference, not teacher training, storage or energy.",
            "RAPL readings, if present, measure a CPU package and include unrelated activity; null means unavailable.",
            "Compiling a known rule from a small DSL is not general reasoning or novel algorithm discovery.",
        ],
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if compiled.skill:
        save_skill(compiled.skill, output_dir / "skill.json")
    return report
