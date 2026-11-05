"""Best-effort CPU package energy via Linux RAPL; unavailable on most hosts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnergyCounter:
    path: Path
    max_energy_uj: int

    @classmethod
    def discover(cls) -> "EnergyCounter | None":
        root = Path("/sys/class/powercap")
        if not root.is_dir():
            return None
        for candidate in sorted(root.glob("intel-rapl:*/energy_uj")):
            try:
                maximum = int((candidate.parent / "max_energy_range_uj").read_text().strip())
                int(candidate.read_text().strip())
                if maximum > 0:
                    return cls(candidate, maximum)
            except (OSError, ValueError):
                continue
        return None

    def read(self) -> int | None:
        try:
            return int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None

    def delta_joules(self, before: int | None, after: int | None) -> float | None:
        if before is None or after is None:
            return None
        delta = (after - before) % self.max_energy_uj
        return delta / 1_000_000
