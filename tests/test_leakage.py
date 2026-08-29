"""Tests for D2.

The whole point of a whitelist is that it fails loudly. Most of these tests
therefore assert that something raises: a silent pass is the bug.
"""

from __future__ import annotations

import csv

import pytest

from src.data import leakage

ROWS = [
    ("id", "app", "identifier", "row identity"),
    ("loan_amnt", "app", "feature", "requested at application"),
    ("annual_inc", "app", "feature", "stated at application"),
    ("emp_title", "app", "text", "free text, needs grouping"),
    ("zip_code", "app", "excluded", "neighbourhood proxy, redlining exposure"),
    ("issue_d", "app", "split_key", "the out-of-time axis"),
    ("loan_status", "post", "target", "the outcome itself"),
    ("total_pymnt", "post", "excluded", "accumulates after origination"),
    ("recoveries", "post", "excluded", "only exists after default"),
]


def _write(tmp_path, rows=ROWS, header=leakage.REQUIRED_FIELDS):
    path = tmp_path / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _replace(column, **changes):
    """Return ROWS with one column's fields altered."""
    out = []
    for row in ROWS:
        if row[0] != column:
            out.append(row)
            continue
        fields = dict(zip(leakage.REQUIRED_FIELDS, row))
        fields.update(changes)
        out.append(tuple(fields[name] for name in leakage.REQUIRED_FIELDS))
    return out


def test_the_whitelist_is_only_the_feature_rows(tmp_path):
    audit = leakage.load_audit(_write(tmp_path))

    assert leakage.feature_columns(audit) == ("loan_amnt", "annual_inc")


def test_available_but_excluded_columns_stay_out(tmp_path):
    # zip_code is available at decision time and still must not be offered.
    # Availability alone does not settle whether a column may be used.
    audit = leakage.load_audit(_write(tmp_path))

    assert "zip_code" not in leakage.feature_columns(audit)
    assert "zip_code" in leakage.build_report(audit).excluded_available


def test_identifiers_text_target_and_split_key_are_not_features(tmp_path):
    audit = leakage.load_audit(_write(tmp_path))
    features = leakage.feature_columns(audit)

    assert leakage.identifier_columns(audit) == ("id",)
    assert leakage.text_columns(audit) == ("emp_title",)
    assert leakage.target_column(audit) == "loan_status"
    assert leakage.split_key(audit) == "issue_d"
    for column in ("id", "emp_title", "loan_status", "issue_d"):
        assert column not in features


def test_a_post_origination_feature_is_refused(tmp_path):
    # The single invariant that matters. If this passes silently, every AUC in
    # the project is meaningless.
    rows = _replace("total_pymnt", role="feature")

    with pytest.raises(ValueError, match="post-origination and also a"):
        leakage.load_audit(_write(tmp_path, rows))


def test_an_unfinished_label_is_refused(tmp_path):
    rows = _replace("annual_inc", availability="?")

    with pytest.raises(ValueError, match="audit is unfinished"):
        leakage.load_audit(_write(tmp_path, rows))


def test_an_unknown_role_is_refused(tmp_path):
    rows = _replace("annual_inc", role="maybe")

    with pytest.raises(ValueError, match="has role"):
        leakage.load_audit(_write(tmp_path, rows))


def test_a_label_without_a_reason_is_refused(tmp_path):
    rows = _replace("annual_inc", reason="")

    with pytest.raises(ValueError, match="carries no reason"):
        leakage.load_audit(_write(tmp_path, rows))


def test_a_duplicated_column_is_refused(tmp_path):
    rows = ROWS + [("loan_amnt", "app", "excluded", "second opinion")]

    with pytest.raises(ValueError, match="more than once"):
        leakage.load_audit(_write(tmp_path, rows))


def test_the_audit_must_name_exactly_one_target(tmp_path):
    rows = _replace("recoveries", role="target")

    with pytest.raises(ValueError, match="role 'target'"):
        leakage.load_audit(_write(tmp_path, rows))


def test_the_audit_must_name_exactly_one_split_key(tmp_path):
    rows = [row for row in ROWS if row[0] != "issue_d"]

    with pytest.raises(ValueError, match="role 'split_key'"):
        leakage.load_audit(_write(tmp_path, rows))


def test_a_missing_field_is_refused(tmp_path):
    header = ("column", "availability", "role")
    rows = [row[:3] for row in ROWS]

    with pytest.raises(ValueError, match="missing required fields"):
        leakage.load_audit(_write(tmp_path, rows, header=header))


def test_a_missing_audit_file_points_at_the_profiling_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="profile_data.py"):
        leakage.load_audit(tmp_path / "absent.csv")


def test_a_new_unaudited_column_raises(tmp_path):
    # A column appearing in the extract that nobody has reviewed must never
    # slip through as a feature.
    audit = leakage.load_audit(_write(tmp_path))
    observed = [row[0] for row in ROWS] + ["brand_new_bureau_field"]

    with pytest.raises(ValueError, match="not audited"):
        leakage.assert_covers(audit, observed)


def test_an_audited_column_missing_from_the_data_raises(tmp_path):
    audit = leakage.load_audit(_write(tmp_path))
    observed = [row[0] for row in ROWS if row[0] != "recoveries"]

    with pytest.raises(ValueError, match="absent from the data"):
        leakage.assert_covers(audit, observed)


def test_covers_passes_when_the_sets_match(tmp_path):
    audit = leakage.load_audit(_write(tmp_path))

    leakage.assert_covers(audit, [row[0] for row in ROWS])


def test_the_leaky_set_adds_post_columns_and_nothing_else(tmp_path):
    # The leaky-vs-clean gap has to measure leakage, so the leaky set differs
    # from the clean one by post-origination columns only - not by zip_code,
    # which is held out on unrelated grounds.
    audit = leakage.load_audit(_write(tmp_path))

    clean = set(leakage.feature_columns(audit))
    leaky = set(leakage.leaky_columns(audit))

    assert leaky - clean == {"total_pymnt", "recoveries"}
    assert "zip_code" not in leaky
    assert "loan_status" not in leaky


def test_report_counts_reconcile(tmp_path):
    audit = leakage.load_audit(_write(tmp_path))
    report = leakage.build_report(audit)

    assert report.total == len(ROWS)
    assert sum(report.by_role.values()) == len(ROWS)
    assert report.by_availability == {"app": 6, "post": 3}


def test_the_committed_audit_is_valid_and_matches_the_extract():
    # The real file, not a fixture. It is the artefact D2 requires to be
    # reviewable, so it has to survive the same checks.
    from src.config import Config

    config = Config.load()
    audit = leakage.load_audit(config.leakage.audit_path)
    report = leakage.build_report(audit)

    assert len(audit) == 151
    assert report.target == "loan_status"
    assert report.split_key == "issue_d"
    assert len(report.features) > 0
    assert not any(a.is_post_origination and a.is_feature for a in audit)
