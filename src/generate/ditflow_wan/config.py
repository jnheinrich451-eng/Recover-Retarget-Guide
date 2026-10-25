"""Config loader for the Wan DiTFlow runner.

Plain yaml to match `src/preprocess/config.py` -- the upstream DiTFlow code
used OmegaConf, which would be a new dependency for one file. Supports both
`cfg.key` and `cfg["key"]` so the ported call sites read unchanged.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class Config(dict):
    """dict with attribute access; nested dicts are wrapped on read."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        self[name] = value


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(data)
