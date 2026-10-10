"""Tests for the exported champion (D12).

The file is the only thing the service knows about the model, so what is
protected here is that nothing is lost or altered on the way through it, and
that a file which has been altered is refused rather than scored with.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from src.serving.artefact import (
    SCHEMA_VERSION,
    champion_from_dict,
    load_champion,
    model_version,
    write_champion,
)


def test_a_written_champion_loads_back_unchanged(fitted, tmp_path):
    path = tmp_path / "champion.json"
    write_champion(fitted.payload, path)

    loaded = load_champion(path)

    assert loaded.version == fitted.champion.version
    assert loaded.intercept == fitted.champion.intercept
    for before, after in zip(fitted.champion.features, loaded.features):
        assert before.name == after.name
        assert before.points == after.points
        assert before.binning.edges == after.binning.edges
        assert before.binning.category_map == after.binning.category_map


def test_the_file_is_strict_json(fitted, tmp_path):
    # Python writes infinity as `Infinity`, which is not JSON and which other
    # tools reject. The open-ended edges must survive as something portable.
    path = tmp_path / "champion.json"
    write_champion(fitted.payload, path)
    text = path.read_text(encoding="utf-8")

    assert "Infinity" not in text and "NaN" not in text
    numeric = [f for f in load_champion(path).features if f.binning.kind == "numeric"]
    assert numeric and all(
        f.binning.edges[0] == -math.inf and f.binning.edges[-1] == math.inf
        for f in numeric
    )


def test_the_version_ignores_metadata_and_follows_the_model(fitted):
    relabelled = copy.deepcopy(fitted.payload)
    relabelled["metadata"]["note"] = "anything"
    assert champion_from_dict(relabelled).version == fitted.champion.version

    changed = copy.deepcopy(fitted.payload["model"])
    changed["features"][0]["coefficient"] *= 1.01
    assert model_version(changed) != fitted.champion.version


def test_a_file_edited_by_hand_is_refused(fitted):
    tampered = copy.deepcopy(fitted.payload)
    tampered["model"]["features"][0]["bins"][0]["points"] += 5.0

    with pytest.raises(ValueError, match="hashes to"):
        champion_from_dict(tampered)


def test_an_unknown_schema_is_refused(fitted):
    future = copy.deepcopy(fitted.payload)
    future["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema version"):
        champion_from_dict(future)


def test_a_missing_file_says_where_it_comes_from(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_models.py"):
        load_champion(tmp_path / "absent.json")


def test_reference_points_are_the_training_average(fitted):
    # D13: "the average score for each of those factors achieved by all
    # applicants", weighted by how many fell in each bin.
    for feature in fitted.champion.features:
        counts = [b.count for b in feature.binning.bins]
        expected = sum(c * p for c, p in zip(counts, feature.points)) / sum(counts)
        assert feature.reference_points == pytest.approx(expected, abs=1e-12)


def test_points_are_stored_unrounded(fitted):
    # The points table rounds to two places for a reader. The service adds
    # the stored values, so rounding them would move every score.
    table = fitted.scorecard.points_table()
    stored = [p for f in fitted.champion.features for p in f.points]
    assert any(round(p, 2) != p for p in stored)
    assert table["points"].tolist() == [round(p, 2) for p in stored]


def test_the_payload_round_trips_through_json(fitted):
    text = json.dumps(fitted.payload, allow_nan=False)
    assert champion_from_dict(json.loads(text)).version == fitted.champion.version
