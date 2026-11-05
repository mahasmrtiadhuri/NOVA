"""Compile observed behavior into a validated, bounded symbolic skill."""

from dataclasses import dataclass
from .language import Example, Program, candidates


@dataclass(frozen=True)
class Skill:
    """A program certified on *held-out validation examples*, not all inputs."""

    name: str
    program: Program
    train_count: int
    validation_count: int
    validation_accuracy: float
    validation_probes: tuple[Example, ...]

    def predict(self, left: int, right: int) -> int:
        return self.program(left, right)


@dataclass(frozen=True)
class CompilationResult:
    skill: Skill | None
    checked_candidates: int
    status: str
    observed_count: int


def _validate_samples(samples: list[Example], modulus: int, name: str) -> set[tuple[int, int]]:
    pairs = set()
    for ex in samples:
        if not isinstance(ex, Example) or any(
            type(v) is not int or not 0 <= v < modulus
            for v in (ex.left, ex.right, ex.output)
        ):
            raise ValueError(f"Invalid {name} example")
        key = (ex.left, ex.right)
        if key in pairs:
            raise ValueError(f"Duplicate {name} example for {key}")
        pairs.add(key)
    return pairs


def compile_skill(
    name: str,
    train: list[Example],
    validation: list[Example],
    *,
    modulus: int,
    max_candidates: int = 5000,
    min_train: int = 8,
    min_validation: int = 12,
) -> CompilationResult:
    """Use examples to choose an interpretable program, refusing uncertainty.

    Candidate selection sees training labels first and held-out validation only
    for certification. A separate test split is required for honest reporting.
    If surviving programs disagree on unobserved domain inputs, decline to
    compile instead of silently guessing which rule is correct.
    """
    if not name or len(name) > 80:
        raise ValueError("Skill needs a short name")
    if not 2 <= modulus <= 64:
        raise ValueError("modulus must be in [2, 64]")
    if max_candidates < 1 or max_candidates > 100_000:
        raise ValueError("max_candidates must be in [1, 100000]")
    if not 1 <= min_train <= 4096 or not 1 <= min_validation <= 4096:
        raise ValueError("Invalid minimum sample count")
    if len(train) > 4096 or len(validation) > 4096:
        raise ValueError("Too many samples")
    observed_count = len(train) + len(validation)
    if len(train) < min_train or len(validation) < min_validation:
        return CompilationResult(None, 0, "insufficient_examples", observed_count)
    train_pairs = _validate_samples(train, modulus, "training")
    validation_pairs = _validate_samples(validation, modulus, "validation")
    if train_pairs & validation_pairs:
        raise ValueError("Training and validation inputs must be disjoint")
    winners = []
    checked = 0
    for program in candidates(modulus):
        if checked >= max_candidates:
            break
        checked += 1
        if all(program(ex.left, ex.right) == ex.output for ex in train) and all(
            program(ex.left, ex.right) == ex.output for ex in validation
        ):
            winners.append(program)
    if checked < 11 + modulus ** 3:
        # A partial search cannot rule out contradictory programs that would
        # satisfy the observed examples. Never certify from a truncated search.
        return CompilationResult(None, checked, "search_budget_exhausted", observed_count)
    if not winners:
        return CompilationResult(None, checked, "no_valid_program", observed_count)

    # A finite-domain ambiguity test is cheap enough for the toy DSL. It is not
    # a proof that the selected program matches the original unknown function.
    signature = tuple(
        winners[0](left, right)
        for left in range(modulus)
        for right in range(modulus)
    )
    for winner in winners[1:]:
        if any(
            winner(left, right) != signature[left * modulus + right]
            for left in range(modulus)
            for right in range(modulus)
        ):
            return CompilationResult(None, checked, "ambiguous_program", observed_count)
    result = Skill(name, winners[0], len(train), len(validation), 1.0, tuple(validation))
    return CompilationResult(result, checked, "compiled", observed_count)
