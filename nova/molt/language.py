"""A deliberately small, interpretable language for finite integer operations.

Programs are data, not Python source code. Nothing here executes user-supplied
code, and nothing attempts to synthesize arbitrary Python expressions.
"""

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Example:
    left: int
    right: int
    output: int


OPS = frozenset({
    "left", "right", "add_mod", "sub_mod", "mul_mod", "xor", "and",
    "or", "max", "min", "absdiff", "affine_mod",
})


@dataclass(frozen=True)
class Program:
    op: str
    modulus: int
    coefficients: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"Unknown operation: {self.op}")
        if not 2 <= self.modulus <= 64:
            raise ValueError("modulus must be between 2 and 64")
        if len(self.coefficients) != 3 or any(
            type(value) is not int or not 0 <= value < self.modulus
            for value in self.coefficients
        ):
            raise ValueError("Invalid coefficients")
        if self.op != "affine_mod" and self.coefficients != (0, 0, 0):
            raise ValueError("Only affine programs accept coefficients")

    def __call__(self, left: int, right: int) -> int:
        if type(left) is not int or type(right) is not int:
            raise TypeError("Operands must be integers")
        if not 0 <= left < self.modulus or not 0 <= right < self.modulus:
            raise ValueError("Inputs fall outside the certified domain")
        m = self.modulus
        if self.op == "left":
            return left
        if self.op == "right":
            return right
        if self.op == "add_mod":
            return (left + right) % m
        if self.op == "sub_mod":
            return (left - right) % m
        if self.op == "mul_mod":
            return (left * right) % m
        if self.op == "xor":
            return (left ^ right) % m
        if self.op == "and":
            return (left & right) % m
        if self.op == "or":
            return (left | right) % m
        if self.op == "max":
            return max(left, right)
        if self.op == "min":
            return min(left, right)
        if self.op == "absdiff":
            return abs(left - right)
        a, b, c = self.coefficients
        return (a * left + b * right + c) % m

    def to_dict(self) -> dict:
        return {"op": self.op, "modulus": self.modulus, "coefficients": list(self.coefficients)}

    @classmethod
    def from_dict(cls, value: dict) -> "Program":
        if not isinstance(value, dict) or set(value) != {"op", "modulus", "coefficients"}:
            raise ValueError("Invalid program specification")
        if not isinstance(value["coefficients"], list):
            raise ValueError("Coefficients must be an array")
        if type(value["modulus"]) is not int:
            raise ValueError("Invalid modulus")
        return cls(value["op"], value["modulus"], tuple(value["coefficients"]))


def candidates(modulus: int) -> Iterator[Program]:
    """Enumerate programs in a deterministic simple-to-complex order.

    The search space is intentionally bounded. Affine parameters cover a small
    family of functions; this does not discover new operator families.
    """
    for op in (
        "left", "right", "add_mod", "sub_mod", "mul_mod", "xor",
        "and", "or", "min", "max", "absdiff",
    ):
        yield Program(op, modulus)
    for a in range(modulus):
        for b in range(modulus):
            for c in range(modulus):
                yield Program("affine_mod", modulus, (a, b, c))
