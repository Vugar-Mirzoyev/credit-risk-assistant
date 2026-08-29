"""Typed access to `config/config.yaml`.

Every number that moves a reported figure lives in the YAML file, not in code.
This module turns that file into frozen dataclasses so a typo becomes an
AttributeError at import time rather than a silent `None` three modules later.

Relative paths in the YAML are resolved against the repository root, so a
script behaves the same whether it is run from the root or from `scripts/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    interim_path: Path
    status_column: str
    date_column: str
    chunksize: int
    max_unparsable_rows: int


@dataclass(frozen=True)
class TargetConfig:
    bad_statuses: tuple[str, ...]
    good_statuses: tuple[str, ...]
    include_policy_failures: bool


@dataclass(frozen=True)
class SplitConfig:
    # None until the yearly volume table in reports/01_profile.md has been
    # inspected. D3 forbids guessing it.
    cutoff_date: date | None
    min_train_months: int


@dataclass(frozen=True)
class WoeConfig:
    max_bins: int
    min_bin_fraction: float
    min_iv: float


@dataclass(frozen=True)
class ModelConfig:
    random_state: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    target: TargetConfig
    split: SplitConfig
    woe: WoeConfig
    model: ModelConfig

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No configuration at {config_path}. Every tunable number lives "
                "in config/config.yaml; the repository cannot run without it."
            )
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(
            data=DataConfig(
                raw_path=_resolve(raw["data"]["raw_path"]),
                interim_path=_resolve(raw["data"]["interim_path"]),
                status_column=raw["data"]["status_column"],
                date_column=raw["data"]["date_column"],
                chunksize=int(raw["data"]["chunksize"]),
                max_unparsable_rows=int(raw["data"]["max_unparsable_rows"]),
            ),
            target=TargetConfig(
                bad_statuses=tuple(raw["target"]["bad_statuses"]),
                good_statuses=tuple(raw["target"]["good_statuses"]),
                include_policy_failures=bool(raw["target"]["include_policy_failures"]),
            ),
            split=SplitConfig(
                cutoff_date=raw["split"]["cutoff_date"],
                min_train_months=int(raw["split"]["min_train_months"]),
            ),
            woe=WoeConfig(
                max_bins=int(raw["woe"]["max_bins"]),
                min_bin_fraction=float(raw["woe"]["min_bin_fraction"]),
                min_iv=float(raw["woe"]["min_iv"]),
            ),
            model=ModelConfig(random_state=int(raw["model"]["random_state"])),
        )


def _resolve(value: str) -> Path:
    """Make a configured path absolute relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
