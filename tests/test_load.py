"""Tests for the loader.

DESIGN.md requires tests for target, woe and split, the three places where a
silent bug corrupts everything downstream. The loader earns tests for the same
reason: dropping the trailer rows and the float32 ceiling are both failures
that would never announce themselves, they would just move the numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import load

HEADER = "id,loan_amnt,annual_inc,grade,issue_d,loan_status\n"
ROWS = (
    "1,10000.0,60000.0,B,Dec-2015,Fully Paid\n"
    "2,5000.0,45000.0,C,Jan-2016,Charged Off\n"
    "3,20000.0,90000.0,A,Feb-2016,Current\n"
)
TRAILER = (
    "Total amount funded in policy code 1: 1465324575,,,,,\n"
    "Total amount funded in policy code 2: 521953170,,,,,\n"
)


def _write(tmp_path, body: str):
    path = tmp_path / "extract.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_trailer_rows_are_dropped_and_counted(tmp_path):
    path = _write(tmp_path, HEADER + ROWS + TRAILER)

    frame, report = load.load_frame(
        path,
        columns=["id", "loan_amnt", "issue_d", "loan_status"],
        chunksize=2,
        date_column="issue_d",
    )

    assert len(frame) == 3
    assert report.unparsable_rows_dropped == 2
    assert report.rows_kept == 3
    # The trailer must not survive as a row with a null status, because that is
    # what would quietly inflate every null rate in the column profile.
    assert frame["loan_status"].notna().all()


def test_a_file_without_a_trailer_reports_zero_dropped(tmp_path):
    path = _write(tmp_path, HEADER + ROWS)

    _, report = load.load_frame(
        path, columns=["id", "loan_amnt"], chunksize=10, date_column="issue_d"
    )

    assert report.unparsable_rows_dropped == 0
    assert report.rows_kept == 3


def test_too_many_unparsable_rows_raises_instead_of_shrinking_the_population(
    tmp_path,
):
    junk = "".join(f"broken-{i},1.0,1.0,A,Dec-2015,Fully Paid\n" for i in range(20))
    path = _write(tmp_path, HEADER + ROWS + junk)

    with pytest.raises(ValueError, match="non-numeric"):
        load.load_frame(
            path,
            columns=["id", "loan_amnt"],
            chunksize=5,
            date_column="issue_d",
            max_unparsable_rows=3,
        )


def test_date_column_is_parsed_to_datetime(tmp_path):
    path = _write(tmp_path, HEADER + ROWS + TRAILER)

    frame, _ = load.load_frame(
        path, columns=["id", "issue_d"], chunksize=2, date_column="issue_d"
    )

    assert pd.api.types.is_datetime64_any_dtype(frame["issue_d"])
    assert frame["issue_d"].min() == pd.Timestamp("2015-12-01")
    assert frame["issue_d"].max() == pd.Timestamp("2016-02-01")


def test_declared_dtypes_are_applied(tmp_path):
    path = _write(tmp_path, HEADER + ROWS)

    frame, _ = load.load_frame(
        path,
        columns=["id", "loan_amnt", "grade"],
        chunksize=10,
        date_column="issue_d",
    )

    assert frame["loan_amnt"].dtype == load.FLOAT_DTYPE
    assert isinstance(frame["grade"].dtype, pd.CategoricalDtype)


def test_the_dtype_map_is_the_contract_everything_else_rests_on():
    # Every downstream module trusts these declarations: WOE routes on them,
    # LightGBM refuses anything but a real categorical, and an unclassified
    # column becoming float32 is what makes a new string column fail loudly
    # rather than quietly arrive as text.
    plan = load.dtype_map(
        ["id", "desc", "grade", "issue_d", "annual_inc", "loan_amnt", "brand_new"]
    )

    assert plan["id"] == "str"  # identifier
    assert plan["desc"] == "str"  # free text
    assert plan["grade"] == "category"
    assert plan["issue_d"] == "category"  # parsed to datetime after reading
    assert plan["annual_inc"] == "float64"  # measured past the float32 ceiling
    assert plan["loan_amnt"] == load.FLOAT_DTYPE
    # The important one: anything nobody classified is numeric by default, so
    # an unexpected text column stops the run instead of slipping through.
    assert plan["brand_new"] == load.FLOAT_DTYPE


def test_categorical_columns_are_read_as_text_and_converted_afterwards():
    # Declaring `category` to the C parser fails on this file: it infers the
    # category set per internal block and an all-null block gets float64
    # categories, which cannot be concatenated with the others.
    read_plan = load._read_dtypes(["grade", "loan_amnt", "id"])

    assert read_plan["grade"] == "str"
    assert read_plan["loan_amnt"] == load.FLOAT_DTYPE
    assert read_plan["id"] == "str"


def test_a_bounded_sample_applies_the_same_declarations(tmp_path):
    path = _write(tmp_path, HEADER + ROWS + TRAILER)

    frame = load.sample_frame(path, nrows=2, date_column="issue_d")

    assert len(frame) == 2
    assert frame["loan_amnt"].dtype == load.FLOAT_DTYPE
    assert isinstance(frame["grade"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(frame["issue_d"])


def test_the_header_is_read_without_touching_any_data_row(tmp_path):
    path = _write(tmp_path, HEADER + ROWS)

    assert load.read_header(path) == (
        "id",
        "loan_amnt",
        "annual_inc",
        "grade",
        "issue_d",
        "loan_status",
    )


def test_a_category_absent_from_the_first_chunk_survives_concatenation(tmp_path):
    # pd.concat of categoricals whose category sets differ falls back to plain
    # strings, silently. The column would then arrive as str, costing the
    # memory the declaration was for, and LightGBM refuses it outright.
    body = HEADER + (
        "1,10000.0,60000.0,A,Dec-2015,Fully Paid\n"
        "2,10000.0,60000.0,A,Dec-2015,Fully Paid\n"
        "3,10000.0,60000.0,G,Dec-2015,Fully Paid\n"
    )
    path = _write(tmp_path, body)

    frame, _ = load.load_frame(
        path, columns=["id", "grade"], chunksize=2, date_column="issue_d"
    )

    assert isinstance(frame["grade"].dtype, pd.CategoricalDtype)
    assert set(frame["grade"].cat.categories) == {"A", "G"}


def test_value_above_the_float32_integer_limit_raises(tmp_path):
    # 2**24 + 1 is the first integer float32 cannot represent exactly. A column
    # left on float32 that reaches it must stop the run, not round silently.
    over = load.FLOAT32_EXACT_INTEGER_LIMIT + 1
    body = HEADER + f"4,{over}.0,60000.0,B,Dec-2015,Fully Paid\n"
    path = _write(tmp_path, body)

    with pytest.raises(ValueError, match="float32 exact-integer limit"):
        load.load_frame(
            path, columns=["id", "loan_amnt"], chunksize=10, date_column="issue_d"
        )


def test_value_below_the_float32_integer_limit_is_accepted(tmp_path):
    under = load.FLOAT32_EXACT_INTEGER_LIMIT - 1
    body = HEADER + f"4,{under}.0,60000.0,B,Dec-2015,Fully Paid\n"
    path = _write(tmp_path, body)

    frame, _ = load.load_frame(
        path, columns=["id", "loan_amnt"], chunksize=10, date_column="issue_d"
    )

    assert frame["loan_amnt"].iloc[0] == under


def test_the_measured_large_column_is_promoted_to_float64(tmp_path):
    # annual_inc reaches 110,000,000 in the real extract, well past what
    # float32 holds exactly. It is read as float64, which both keeps the value
    # exact and takes it out of scope of the float32 ceiling.
    observed_maximum = 110_000_000
    body = HEADER + f"4,10000.0,{observed_maximum}.0,B,Dec-2015,Fully Paid\n"
    path = _write(tmp_path, body)

    frame, _ = load.load_frame(
        path, columns=["id", "annual_inc"], chunksize=10, date_column="issue_d"
    )

    assert frame["annual_inc"].dtype == "float64"
    assert frame["annual_inc"].iloc[0] == observed_maximum


def test_requesting_an_absent_column_raises(tmp_path):
    path = _write(tmp_path, HEADER + ROWS)

    with pytest.raises(KeyError, match="fico_score_we_wish_we_had"):
        load.load_frame(
            path,
            columns=["id", "fico_score_we_wish_we_had"],
            chunksize=10,
            date_column="issue_d",
        )


def test_id_is_not_returned_unless_it_was_asked_for(tmp_path):
    # id is read regardless because it is the trailer detector, but a caller
    # who did not ask for it should not silently receive it.
    path = _write(tmp_path, HEADER + ROWS + TRAILER)

    frame, report = load.load_frame(
        path, columns=["loan_amnt"], chunksize=2, date_column="issue_d"
    )

    assert list(frame.columns) == ["loan_amnt"]
    assert report.unparsable_rows_dropped == 2


def test_chunk_stream_reads_every_column_without_a_column_list(tmp_path):
    path = _write(tmp_path, HEADER + ROWS + TRAILER)

    stream = load.ChunkStream(path, chunksize=2, date_column="issue_d")
    chunks = list(stream)

    assert sum(len(chunk) for chunk in chunks) == 3
    assert stream.report.columns_read == (
        "id",
        "loan_amnt",
        "annual_inc",
        "grade",
        "issue_d",
        "loan_status",
    )
    assert stream.report.chunks_read == len(chunks)


def test_missing_file_names_the_expected_location(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load.ChunkStream(tmp_path / "absent.csv", chunksize=10, date_column="issue_d")
