from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunConfig:
    """Minimal provenance record for new experiment runs."""

    experiment_tag: str
    domain: str
    model_id: str
    llm_endpoint: str
    seed: int | None = None
    target_samples: int = 500
    generations: int = 15
    initial_particles: int = 150000
    metric: str = "summary_euclidean"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["python"] = platform.python_version()
        data["numpy"] = np.__version__
        return data


def write_run_config(config: RunConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    return path
