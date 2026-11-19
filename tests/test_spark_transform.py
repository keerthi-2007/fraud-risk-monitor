"""Tests for the Spark ingestion/transformation stage (needs pyspark + Java)."""

import math

import pytest

pyspark = pytest.importorskip("pyspark")

from scripts import spark_transform  # noqa: E402

FEATURES = spark_transform.FEATURE_COLUMNS


@pytest.fixture(scope="module")
def spark():
    try:
        session = spark_transform.build_spark("fraud-risk-spark-transform-test")
    except Exception as exc:  # e.g. no compatible Java runtime
        pytest.skip(f"Spark unavailable: {exc}")
    yield session
    session.stop()


def write_csv(path, rows, with_time=False):
    header = (["Time"] if with_time else []) + FEATURES + ["Class"]
    lines = [",".join(header)]
    for amount, label, base in rows:
        values = ([0.0] if with_time else []) + [base] * 28 + [amount, label]
        lines.append(",".join(str(v) for v in values))
    path.write_text("\n".join(lines) + "\n")


def test_transform_writes_enriched_partitioned_parquet(spark, tmp_path):
    csv = tmp_path / "data" / "raw" / "creditcard.csv"
    csv.parent.mkdir(parents=True)
    write_csv(csv, [(0.0, 0, 1.0), (10.0, 0, 2.0), (10.0, 0, 2.0), (99.0, 1, 3.0)], with_time=True)
    out = tmp_path / "features"

    report = spark_transform.run(spark, csv, out)

    assert report["rows"] == 4
    assert report["fraud_rows"] == 1
    assert report["duplicated_rows"] == 2
    assert report["rows_per_second"] > 0
    assert (out / "Class=1").is_dir() and (out / "Class=0").is_dir()

    rows = {r["Amount"]: r for r in spark.read.parquet(str(out)).collect()}
    assert "Time" not in spark.read.parquet(str(out)).columns
    assert rows[0.0]["is_zero_amount"] == 1
    assert rows[99.0]["log_amount"] == pytest.approx(math.log1p(99.0))
    assert rows[10.0]["duplicate_count"] == 2
    assert len(rows[99.0]["feature_hash"]) == 64


def test_negative_amount_fails_validation(spark, tmp_path):
    csv = tmp_path / "data" / "raw" / "creditcard.csv"
    csv.parent.mkdir(parents=True)
    write_csv(csv, [(-5.0, 0, 1.0), (1.0, 1, 1.0)])

    with pytest.raises(ValueError, match="negative_amount"):
        spark_transform.run(spark, csv, tmp_path / "out")
