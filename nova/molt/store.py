"""JSON-only skill persistence, with validation checks during loading."""

from dataclasses import asdict
import json
from pathlib import Path

from .compiler import Skill
from .language import Example, Program


SCHEMA_VERSION = 1


def save_skill(skill: Skill, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "name": skill.name,
        "program": skill.program.to_dict(),
        "train_count": skill.train_count,
        "validation_count": skill.validation_count,
        "validation_accuracy": skill.validation_accuracy,
        "validation_probes": [asdict(ex) for ex in skill.validation_probes],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_skill(path: Path) -> Skill:
    path = Path(path)
    if path.stat().st_size > 500_000:
        raise ValueError("Skill file too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "name", "program", "train_count", "validation_count", "validation_accuracy", "validation_probes"}
    if not isinstance(data, dict) or set(data) != required or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unknown skill schema")
    if not isinstance(data["validation_probes"], list) or not 1 <= len(data["validation_probes"]) <= 4096:
        raise ValueError("Invalid validation probes")
    probes = []
    for ex in data["validation_probes"]:
        if not isinstance(ex, dict) or set(ex) != {"left", "right", "output"}:
            raise ValueError("Invalid validation probe")
        if any(type(ex[key]) is not int for key in ("left", "right", "output")):
            raise ValueError("Validation probe must contain integers")
        probes.append(Example(**ex))
    if type(data["train_count"]) is not int or not 1 <= data["train_count"] <= 4096:
        raise ValueError("Invalid train count")
    if type(data["validation_count"]) is not int or data["validation_count"] != len(probes):
        raise ValueError("Invalid validation count")
    if type(data["validation_accuracy"]) not in (float, int) or data["validation_accuracy"] != 1.0:
        raise ValueError("Invalid validation accuracy")
    if not isinstance(data["name"], str) or not 1 <= len(data["name"]) <= 80:
        raise ValueError("Invalid skill name")
    skill = Skill(
        data["name"], Program.from_dict(data["program"]),
        data["train_count"], data["validation_count"],
        float(data["validation_accuracy"]), tuple(probes),
    )
    # Revalidate before returning. This detects corrupted local artifacts but
    # does not authenticate the original dataset or prove universal accuracy.
    if any(skill.predict(ex.left, ex.right) != ex.output for ex in probes):
        raise ValueError("Loaded skill failed its validation probes")
    return skill
