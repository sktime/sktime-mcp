"""Regression tests for time-index coercion in the data adapters (#556 item 11).

Audit findings F-13, F-14, F-15, F-35 and F-37: integer time columns were fed to
``pd.to_datetime`` (epoch nanoseconds), the SQL adapter crashed on <3 rows and
created missing sqlite files, an explicit ``frequency`` that missed the
timestamps wiped the data to NaN, mixed-timezone timestamps left an object
index, and a bare-string ``exog_columns`` was iterated per character.
Every test here fails on origin/main (f6fb797) before the fix.
"""

import pandas as pd
import pytest

from sktime_mcp.tools.data_tools import load_data_source_tool, release_data_handle_tool

try:
    import sqlalchemy
except ImportError:  # pragma: no cover - SQL tests are skipped, the rest still run
    sqlalchemy = None

needs_sqlalchemy = pytest.mark.skipif(
    sqlalchemy is None, reason="SQL adapter tests need sqlalchemy"
)


def _load(config):
    result = load_data_source_tool(config)
    if result.get("data_handle"):
        release_data_handle_tool(result["data_handle"])
    return result


YEARS = {"year": [2018, 2019, 2020, 2021, 2022], "value": [1, 2, 3, 4, 5]}
MID_MONTH = {
    "date": ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"],
    "value": [1, 2, 3, 4],
}


@pytest.fixture
def sqlite_url(tmp_path):
    if sqlalchemy is None:
        pytest.skip("SQL adapter tests need sqlalchemy")
    """A sqlite file holding table ``t`` (2 rows, string dates) and ``years`` (int years)."""
    db = tmp_path / "t.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]}).to_sql(
        "t", engine, index=False
    )
    pd.DataFrame(YEARS).to_sql("years", engine, index=False)
    engine.dispose()
    return f"sqlite:///{db}"


# --- F-13: integer time column -> integer index, not epoch nanoseconds ---------


def test_pandas_integer_time_column_gives_integer_index():
    result = _load(
        {"type": "pandas", "data": YEARS, "time_column": "year", "target_column": "value"}
    )

    assert result["success"] is True
    assert result["metadata"]["frequency"] == "Integer"
    assert result["metadata"]["start_date"] == "2018"
    assert result["metadata"]["end_date"] == "2022"


def test_file_integer_time_column_gives_integer_index(tmp_path):
    path = tmp_path / "years.csv"
    path.write_text("year,value\n2018,1\n2019,2\n2020,3\n2021,4\n2022,5\n")

    result = _load(
        {"type": "file", "path": str(path), "time_column": "year", "target_column": "value"}
    )

    assert result["success"] is True
    assert result["metadata"]["frequency"] == "Integer"
    assert result["metadata"]["start_date"] == "2018"


def test_sql_integer_time_column_gives_integer_index(sqlite_url):
    result = _load(
        {
            "type": "sql",
            "connection_string": sqlite_url,
            "table": "years",
            "time_column": "year",
            "target_column": "value",
        }
    )

    assert result["success"] is True
    assert result["metadata"]["frequency"] == "Integer"
    assert result["metadata"]["start_date"] == "2018"


def test_coerce_time_index_rejects_ints_mixed_with_strings():
    from sktime_mcp.data.base import coerce_time_index

    df = pd.DataFrame({"t": [2018, "2019-01-01", 2020], "value": [1, 2, 3]})

    with pytest.raises(ValueError, match="mixes numbers with non-numeric values"):
        coerce_time_index(df, "t")


def test_coerce_time_index_still_parses_date_strings():
    from sktime_mcp.data.base import coerce_time_index

    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]})

    out, warnings = coerce_time_index(df, "date")

    assert isinstance(out.index, pd.DatetimeIndex)
    assert warnings == []


# --- F-14: SQL adapter without time_column, short tables, missing sqlite file --


def test_sql_without_time_column_keeps_range_index_and_loads_two_rows(sqlite_url):
    result = _load(
        {
            "type": "sql",
            "connection_string": sqlite_url,
            "table": "t",
            "target_column": "value",
        }
    )

    assert result["success"] is True, result.get("error")
    assert result["metadata"]["frequency"] == "Integer"
    assert result["metadata"]["rows"] == 2


@needs_sqlalchemy
def test_sql_missing_sqlite_file_errors_without_creating_it(tmp_path):
    missing = tmp_path / "nope.db"

    result = _load({"type": "sql", "connection_string": f"sqlite:///{missing}", "table": "t"})

    assert result["success"] is False
    assert result["error_type"] == "FileNotFoundError"
    assert "nope.db" in result["error"]
    assert not missing.exists()


# --- F-15: explicit frequency that does not align with the timestamps ----------


def test_pandas_misaligned_frequency_is_an_error_not_nan():
    result = _load(
        {
            "type": "pandas",
            "data": MID_MONTH,
            "time_column": "date",
            "target_column": "value",
            "frequency": "ME",
        }
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "'ME'" in result["error"]
    assert "2020-01-15" in result["error"]


def test_file_misaligned_frequency_is_an_error_not_nan(tmp_path):
    path = tmp_path / "mid.csv"
    path.write_text("date,value\n2020-01-15,1\n2020-02-15,2\n2020-03-15,3\n2020-04-15,4\n")

    result = _load(
        {
            "type": "file",
            "path": str(path),
            "time_column": "date",
            "target_column": "value",
            "frequency": "ME",
        }
    )

    assert result["success"] is False
    assert "'ME'" in result["error"]
    assert "2020-01-15" in result["error"]


def test_sql_misaligned_frequency_is_an_error_not_nan(sqlite_url):
    result = _load(
        {
            "type": "sql",
            "connection_string": sqlite_url,
            "table": "t",
            "time_column": "date",
            "target_column": "value",
            "frequency": "W-MON",
        }
    )

    assert result["success"] is False
    assert "'W-MON'" in result["error"]


def test_aligned_frequency_with_gaps_still_applies():
    """Gaps are fine (asfreq fills them); only off-grid timestamps are rejected."""
    result = _load(
        {
            "type": "pandas",
            "data": {"date": ["2020-01-01", "2020-01-03"], "value": [1, 3]},
            "time_column": "date",
            "target_column": "value",
            "frequency": "D",
        }
    )

    assert result["success"] is True, result.get("error")
    assert result["metadata"]["rows"] == 3


def test_check_frequency_alignment_ignores_invalid_strings_and_integer_index():
    """Invalid freq strings are PR #451's concern; non-datetime indexes are skipped."""
    from sktime_mcp.data.base import check_frequency_alignment

    check_frequency_alignment(pd.DatetimeIndex(["2020-01-01"]), "not_a_freq")
    check_frequency_alignment(pd.Index([1, 2, 3]), "D")


# --- F-35: mixed-timezone timestamps -> UTC with a warning ---------------------


def test_mixed_timezone_timestamps_are_normalised_to_utc_with_warning():
    result = _load(
        {
            "type": "pandas",
            "data": {
                "date": [
                    "2020-01-01T00:00:00+00:00",
                    "2020-01-02T01:00:00+01:00",
                    "2020-01-03T02:00:00+02:00",
                ],
                "value": [1, 2, 3],
            },
            "time_column": "date",
            "target_column": "value",
        }
    )

    assert result["success"] is True, result.get("error")
    assert result["metadata"]["frequency"] != "Integer"
    assert any("normalised to UTC" in w for w in result["validation"]["warnings"])


# --- F-37: exog_columns given as a bare string --------------------------------


def test_exog_columns_string_is_treated_as_one_column():
    result = _load(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
                "promo": [0, 1, 0],
            },
            "time_column": "date",
            "target_column": "value",
            "exog_columns": "promo",
        }
    )

    assert result["success"] is True, result.get("error")
    assert result["metadata"]["exog_columns"] == ["promo"]
