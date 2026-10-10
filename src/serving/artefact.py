"""D12: the champion as a file the service reads, not a fit it reruns.

`scripts/train_models.py` writes the fitted champion to `models/champion.json`
and the scoring service loads that file and nothing else. Three properties
follow from the format, and each one is the reason for choosing it.

*JSON, not a pickle.* A pickle executes code when it is loaded and cannot be
read by a person. This file is a list of bins, their WOE and their points: a
reviewer can diff two versions of it, and the service can load it without
trusting it. It also means the service never imports scikit-learn, because a
logistic regression over WOE is one dot product once its coefficients are
written down.

*The binning is stored, not re-implemented.* Edges, category maps and the
missing bin are written exactly as `WoeBinner` fitted them, and the service
rebuilds the same `FeatureBinning` objects and calls the same method training
used. Training-serving skew - two implementations of one transform drifting
apart - cannot happen when there is only one implementation.

*The version is a hash of the model, not a date.* It covers everything that
moves a score - bins, WOE, coefficients, intercept, scaling - and nothing that
does not. Refitting on the same data with the same configuration gives the
same version; changing one bin edge gives a different one. A score can always
be traced to exactly the model that produced it.

The reference points for D13's reason selection are computed here too, once,
from the training counts the bins already carry.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from src.features.woe import Bin, FeatureBinning, WoeBinner

if TYPE_CHECKING:  # the service never imports scikit-learn
    from src.models.scorecard import Scorecard

#: Bumped when the layout of the file changes, so an old service refuses a
#: new file instead of misreading it.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScoredFeature:
    """One scorecard feature: its bins, its coefficient, its points per bin."""

    binning: FeatureBinning
    coefficient: float
    points: tuple[float, ...]
    #: D13. The average points this feature gave the training population,
    #: weighted by how many applicants fell in each bin.
    reference_points: float

    @property
    def name(self) -> str:
        return self.binning.column


@dataclass(frozen=True)
class Champion:
    """The fitted champion, as the service sees it."""

    name: str
    version: str
    intercept: float
    base_points: float
    scaling: Mapping[str, float]
    features: tuple[ScoredFeature, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def feature(self, name: str) -> ScoredFeature:
        for item in self.features:
            if item.name == name:
                return item
        raise KeyError(f"{name!r} is not a feature of champion {self.version}.")


def export_champion(
    scorecard: "Scorecard",
    binner: WoeBinner,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The champion as a JSON-serialisable dictionary, versioned by content."""
    features = []
    for name in scorecard.features:
        binning = binner.binning(name)
        coefficient = scorecard.coefficients[name]
        features.append(
            {
                "name": name,
                "kind": binning.kind,
                "coefficient": coefficient,
                "iv": binning.iv,
                "edges": (
                    [_encode_edge(e) for e in binning.edges]
                    if binning.edges is not None
                    else None
                ),
                "category_map": dict(binning.category_map),
                "missing_index": binning.missing_index,
                "bins": [
                    {
                        "label": item.label,
                        "count": item.count,
                        "good": item.good,
                        "bad": item.bad,
                        "woe": item.woe,
                        "iv": item.iv,
                        "includes_missing": item.includes_missing,
                        # The same expression the scorecard uses, unrounded:
                        # the points table rounds for display, the service
                        # must not.
                        "points": -scorecard.scaling.factor * coefficient * item.woe,
                    }
                    for item in binning.bins
                ],
            }
        )

    model = {
        "name": scorecard.name,
        "intercept": scorecard.intercept,
        "base_points": scorecard.base_points,
        "scaling": {
            "base_score": scorecard.scaling.base_score,
            "base_odds": scorecard.scaling.base_odds,
            "pdo": scorecard.scaling.pdo,
        },
        "features": features,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "version": model_version(model),
        "model": model,
        "metadata": dict(metadata or {}),
    }


def write_champion(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_champion(path: Path) -> Champion:
    if not path.is_file():
        raise FileNotFoundError(
            f"No champion at {path}. It is written by "
            "`python scripts/train_models.py`; in a clone of the repository it "
            "is committed, so its absence means the checkout is incomplete."
        )
    return champion_from_dict(json.loads(path.read_text(encoding="utf-8")))


def champion_from_dict(payload: Mapping[str, Any]) -> Champion:
    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Champion file has schema version {schema!r}; this code reads "
            f"{SCHEMA_VERSION}. Retrain, or run the code the file was written by."
        )
    model = payload["model"]
    version = model_version(model)
    if version != payload["version"]:
        raise ValueError(
            f"Champion file claims version {payload['version']} but its content "
            f"hashes to {version}. It has been edited by hand or truncated, and "
            "a score from it could not be traced to a fitted model."
        )

    features = []
    for item in model["features"]:
        bins = tuple(
            Bin(
                label=b["label"],
                count=int(b["count"]),
                good=int(b["good"]),
                bad=int(b["bad"]),
                woe=float(b["woe"]),
                iv=float(b["iv"]),
                includes_missing=bool(b["includes_missing"]),
            )
            for b in item["bins"]
        )
        binning = FeatureBinning(
            column=item["name"],
            kind=item["kind"],
            bins=bins,
            iv=float(item["iv"]),
            edges=(
                tuple(_decode_edge(e) for e in item["edges"])
                if item["edges"] is not None
                else None
            ),
            category_map={str(k): int(v) for k, v in item["category_map"].items()},
            missing_index=item["missing_index"],
        )
        points = tuple(float(b["points"]) for b in item["bins"])
        features.append(
            ScoredFeature(
                binning=binning,
                coefficient=float(item["coefficient"]),
                points=points,
                reference_points=_reference_points(bins, points),
            )
        )

    return Champion(
        name=model["name"],
        version=version,
        intercept=float(model["intercept"]),
        base_points=float(model["base_points"]),
        scaling=dict(model["scaling"]),
        features=tuple(features),
        metadata=dict(payload.get("metadata", {})),
    )


def model_version(model: Mapping[str, Any]) -> str:
    """First twelve hex digits of a SHA-256 over the canonical model content."""
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _reference_points(bins: tuple[Bin, ...], points: tuple[float, ...]) -> float:
    """D13: the average points of all applicants, from the training counts.

    Regulation B commentary 9(b)(2)-5 names this as one acceptable basis for
    selecting reasons: "the average score for each of those factors achieved
    by all applicants", which "could be calculated during the development ...
    of the system". The bins already carry how many training applicants fell
    in each, so nothing has to be recomputed from the data.
    """
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("A scorecard feature whose bins hold no training rows.")
    return sum(b.count * p for b, p in zip(bins, points)) / total


def _encode_edge(value: float) -> float | str:
    # JSON has no infinity, and Python's `Infinity` extension is not JSON.
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _decode_edge(value: float | str) -> float:
    return float(value)
