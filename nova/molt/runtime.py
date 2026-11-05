"""Select certified skills and fall back to a reference solver."""

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Callable

from .compiler import Skill


@dataclass
class RoutingStats:
    paths: Counter = field(default_factory=Counter)
    timings_ns: dict[str, list[int]] = field(default_factory=lambda: {"compiled": [], "fallback": []})

    def as_dict(self) -> dict:
        return {
            "paths": dict(self.paths),
            "timings_ns": {key: list(values) for key, values in self.timings_ns.items()},
        }


class MoltRouter:
    """Dispatch named, validated skills with optional independent spot checks.

    A validation score is not a guarantee of universal correctness. Unknown
    tasks and out-of-domain inputs always use the supplied fallback. An
    independent verifier can detect some violations during deployment and
    disable a bad skill; verifying every call can erase the cost advantage.
    """

    def __init__(
        self,
        fallback: Callable[[str, int, int], int],
        *,
        verifier: Callable[[str, int, int], int] | None = None,
        verify_every: int = 0,
    ) -> None:
        if verify_every < 0 or (verify_every and verifier is None):
            raise ValueError("verify_every requires a positive interval and a verifier")
        self.fallback = fallback
        self.verifier = verifier
        self.verify_every = verify_every
        self.skills: dict[str, Skill] = {}
        self.stats = RoutingStats()
        self._compiled_seen = 0

    def install(self, skill: Skill) -> None:
        if len(skill.validation_probes) != skill.validation_count:
            raise ValueError("Invalid validation certificate")
        if any(skill.predict(ex.left, ex.right) != ex.output for ex in skill.validation_probes):
            raise ValueError("Stored skill does not pass its validation probes")
        if skill.validation_accuracy != 1.0:
            raise ValueError("Only fully validated skills may be installed")
        self.skills[skill.name] = skill

    def predict(self, name: str, left: int, right: int) -> int:
        start = perf_counter_ns()
        skill = self.skills.get(name)
        if skill is not None:
            try:
                result = skill.predict(left, right)
            except (ValueError, TypeError):
                pass
            else:
                self._compiled_seen += 1
                if self.verify_every and self._compiled_seen % self.verify_every == 0:
                    # The verifier should be independent of the model that
                    # supplied training labels; otherwise this is only agreement.
                    assert self.verifier is not None
                    verified = self.verifier(name, left, right)
                    self.stats.paths["verification_calls"] += 1
                    if result != verified:
                        self.skills.pop(name, None)
                        self.stats.paths["verification_failures"] += 1
                        self.stats.paths["fallback"] += 1
                        self.stats.timings_ns["fallback"].append(perf_counter_ns() - start)
                        return verified
                self.stats.paths["compiled"] += 1
                self.stats.timings_ns["compiled"].append(perf_counter_ns() - start)
                return result
        result = self.fallback(name, left, right)
        self.stats.paths["fallback"] += 1
        self.stats.timings_ns["fallback"].append(perf_counter_ns() - start)
        return result
