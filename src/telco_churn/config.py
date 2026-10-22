"""Load project configuration and resolve paths relative to the repository."""

from pathlib import Path
from typing import Any

import yaml


# Resolve paths from the package location so commands work from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read a YAML configuration file and require a mapping at its root."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping.")

    return config


def project_path(path: str | Path) -> Path:
    """Resolve a configured relative path against the repository root."""
    resolved_path = Path(path)
    return resolved_path if resolved_path.is_absolute() else PROJECT_ROOT / resolved_path
