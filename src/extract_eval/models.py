"""The list of models `run_eval --models` can run, read from models.json."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .vocab import ROOT

MODELS_PATH = ROOT / "models.json"


@dataclass(frozen=True)
class ModelEntry:
    name: str  # LiteLLM model string
    env_key: str | None = None  # env var that must be set for this model to run
    enabled: bool = True


def load_models(path: str | Path = MODELS_PATH) -> list[ModelEntry]:
    raw = json.loads(Path(path).read_text())
    return [ModelEntry(m["name"], m.get("env_key"), m.get("enabled", True)) for m in raw["models"]]


def resolve_models(
    spec: str, entries: list[ModelEntry], env: Mapping[str, str] | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    """Turn a `--models` value into (models to run, [(model, reason skipped)]).

    `all` = every enabled entry whose API key is set. Anything else is a comma-separated list of
    model names, which are run as given (a registered name still gets its key checked).
    """
    env = os.environ if env is None else env
    by_name = {e.name: e for e in entries}
    if spec.strip().lower() == "all":
        candidates = [e.name for e in entries if e.enabled]
        explicit = False
    else:
        candidates = [s.strip() for s in spec.split(",") if s.strip()]
        explicit = True
    run, skipped = [], []
    for name in dict.fromkeys(candidates):  # de-dupe, keep order
        entry = by_name.get(name)
        if entry and entry.env_key and not env.get(entry.env_key):
            skipped.append((name, f"{entry.env_key} is not set"))
        else:
            run.append(name)
    if not explicit and not entries:
        skipped.append(("all", "models.json lists no models"))
    return run, skipped
