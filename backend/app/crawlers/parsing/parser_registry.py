from pathlib import Path
from typing import Any, Dict, List

import yaml

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
REQUIRED_KEYS = ("name", "parser")


def load_registry(templates_dir: Path | None = None) -> List[Dict[str, Any]]:
    resolved_dir = templates_dir or TEMPLATES_DIR
    registry: List[Dict[str, Any]] = []

    if not resolved_dir.exists():
        return registry

    for template_path in sorted(resolved_dir.glob("*.yaml")):
        data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        entries = data.get("registry", [])
        for entry in entries:
            missing = [key for key in REQUIRED_KEYS if key not in entry]
            if missing:
                raise ValueError(
                    f"Invalid parser registry template {template_path.name}: missing keys {missing}"
                )
            registry.append(
                {
                    "name": entry["name"],
                    "url_patterns": tuple(entry.get("url_patterns", [])),
                    "categories": tuple(entry.get("categories", [])),
                    "parser": entry["parser"],
                    "priority": int(entry.get("priority", 100)),
                    "options": dict(entry.get("options", {})),
                }
            )

    registry.sort(key=lambda entry: (entry["priority"], entry["name"]))
    return registry


REGISTRY = load_registry()
