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
    status_column: str
    date_column: str
    chunksize: int
    max_unparsable_rows: int


@dataclass(frozen=True)
class TargetConfig:
    bad_statuses: tuple[str, ...]
    good_statuses: tuple[str, ...]
    unresolved_statuses: tuple[str, ...]
    include_policy_failures: bool


@dataclass(frozen=True)
class ProfileConfig:
    max_distinct_tracked: int
    report_path: Path
    column_csv_path: Path
    figure_path: Path


@dataclass(frozen=True)
class LeakageConfig:
    audit_path: Path


@dataclass(frozen=True)
class ChallengerConfig:
    validation_months: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_lambda: float
    n_estimators: int
    early_stopping_rounds: int
    deterministic: bool


@dataclass(frozen=True)
class ExplainConfig:
    sample_rows: int
    top_features: int


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str
    target_words: int
    overlap_words: int
    min_words: int
    prepend_heading: bool


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    bm25_k1: float
    bm25_b: float
    embedding_model: str
    batch_size: int
    rrf_k: int
    cache_path: Path
    report_path: Path
    questions_path: Path


@dataclass(frozen=True)
class AnswerConfig:
    base_url: str
    model: str
    api_key_env: str
    temperature: float
    max_tokens: int
    max_context_words: int


@dataclass(frozen=True)
class RagConfig:
    corpus_path: Path
    chunk: ChunkConfig
    retrieval: RetrievalConfig
    answer: AnswerConfig


@dataclass(frozen=True)
class SplitConfig:
    # None until the yearly volume table in reports/01_profile.md has been
    # inspected. D3 forbids guessing it.
    cutoff_date: date | None
    min_train_months: int


@dataclass(frozen=True)
class WoeConfig:
    pre_bins: int
    max_bins: int
    min_bin_fraction: float
    min_category_fraction: float
    enforce_monotonic: bool
    min_iv: float
    suspicious_iv: float


@dataclass(frozen=True)
class ScorecardConfig:
    base_score: float
    base_odds: float
    pdo: float


@dataclass(frozen=True)
class ModelConfig:
    random_state: int
    max_correlation: float
    max_iter: int
    scorecard: ScorecardConfig


@dataclass(frozen=True)
class Config:
    data: DataConfig
    target: TargetConfig
    profile: ProfileConfig
    leakage: LeakageConfig
    challenger: ChallengerConfig
    explain: ExplainConfig
    rag: RagConfig
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
                status_column=raw["data"]["status_column"],
                date_column=raw["data"]["date_column"],
                chunksize=int(raw["data"]["chunksize"]),
                max_unparsable_rows=int(raw["data"]["max_unparsable_rows"]),
            ),
            target=TargetConfig(
                bad_statuses=tuple(raw["target"]["bad_statuses"]),
                good_statuses=tuple(raw["target"]["good_statuses"]),
                unresolved_statuses=tuple(raw["target"]["unresolved_statuses"]),
                include_policy_failures=bool(raw["target"]["include_policy_failures"]),
            ),
            profile=ProfileConfig(
                max_distinct_tracked=int(raw["profile"]["max_distinct_tracked"]),
                report_path=_resolve(raw["profile"]["report_path"]),
                column_csv_path=_resolve(raw["profile"]["column_csv_path"]),
                figure_path=_resolve(raw["profile"]["figure_path"]),
            ),
            leakage=LeakageConfig(
                audit_path=_resolve(raw["leakage"]["audit_path"]),
            ),
            challenger=ChallengerConfig(
                validation_months=int(raw["challenger"]["validation_months"]),
                learning_rate=float(raw["challenger"]["learning_rate"]),
                num_leaves=int(raw["challenger"]["num_leaves"]),
                min_child_samples=int(raw["challenger"]["min_child_samples"]),
                subsample=float(raw["challenger"]["subsample"]),
                subsample_freq=int(raw["challenger"]["subsample_freq"]),
                colsample_bytree=float(raw["challenger"]["colsample_bytree"]),
                reg_lambda=float(raw["challenger"]["reg_lambda"]),
                n_estimators=int(raw["challenger"]["n_estimators"]),
                early_stopping_rounds=int(raw["challenger"]["early_stopping_rounds"]),
                deterministic=bool(raw["challenger"]["deterministic"]),
            ),
            explain=ExplainConfig(
                sample_rows=int(raw["explain"]["sample_rows"]),
                top_features=int(raw["explain"]["top_features"]),
            ),
            rag=RagConfig(
                corpus_path=_resolve(raw["rag"]["corpus_path"]),
                chunk=ChunkConfig(
                    strategy=raw["rag"]["chunk"]["strategy"],
                    target_words=int(raw["rag"]["chunk"]["target_words"]),
                    overlap_words=int(raw["rag"]["chunk"]["overlap_words"]),
                    min_words=int(raw["rag"]["chunk"]["min_words"]),
                    prepend_heading=bool(raw["rag"]["chunk"]["prepend_heading"]),
                ),
                retrieval=RetrievalConfig(
                    top_k=int(raw["rag"]["retrieval"]["top_k"]),
                    bm25_k1=float(raw["rag"]["retrieval"]["bm25_k1"]),
                    bm25_b=float(raw["rag"]["retrieval"]["bm25_b"]),
                    embedding_model=raw["rag"]["retrieval"]["embedding_model"],
                    batch_size=int(raw["rag"]["retrieval"]["batch_size"]),
                    rrf_k=int(raw["rag"]["retrieval"]["rrf_k"]),
                    cache_path=_resolve(raw["rag"]["retrieval"]["cache_path"]),
                    report_path=_resolve(raw["rag"]["retrieval"]["report_path"]),
                    questions_path=_resolve(raw["rag"]["retrieval"]["questions_path"]),
                ),
                answer=AnswerConfig(
                    base_url=raw["rag"]["answer"]["base_url"],
                    model=raw["rag"]["answer"]["model"],
                    api_key_env=raw["rag"]["answer"]["api_key_env"],
                    temperature=float(raw["rag"]["answer"]["temperature"]),
                    max_tokens=int(raw["rag"]["answer"]["max_tokens"]),
                    max_context_words=int(raw["rag"]["answer"]["max_context_words"]),
                ),
            ),
            split=SplitConfig(
                cutoff_date=raw["split"]["cutoff_date"],
                min_train_months=int(raw["split"]["min_train_months"]),
            ),
            woe=WoeConfig(
                pre_bins=int(raw["woe"]["pre_bins"]),
                max_bins=int(raw["woe"]["max_bins"]),
                min_bin_fraction=float(raw["woe"]["min_bin_fraction"]),
                min_category_fraction=float(raw["woe"]["min_category_fraction"]),
                enforce_monotonic=bool(raw["woe"]["enforce_monotonic"]),
                min_iv=float(raw["woe"]["min_iv"]),
                suspicious_iv=float(raw["woe"]["suspicious_iv"]),
            ),
            model=ModelConfig(
                random_state=int(raw["model"]["random_state"]),
                max_correlation=float(raw["model"]["max_correlation"]),
                max_iter=int(raw["model"]["max_iter"]),
                scorecard=ScorecardConfig(
                    base_score=float(raw["model"]["scorecard"]["base_score"]),
                    base_odds=float(raw["model"]["scorecard"]["base_odds"]),
                    pdo=float(raw["model"]["scorecard"]["pdo"]),
                ),
            ),
        )


def _resolve(value: str) -> Path:
    """Make a configured path absolute relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
