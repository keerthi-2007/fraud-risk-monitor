"""Spark data-engineering stage: raw CSV -> typed, enriched Parquet.

This is the distributed ingestion/transformation step of the pipeline
(DVC stage ``spark_transform``). It:

1. ingests ``data/raw/creditcard.csv`` with an explicit schema (FAILFAST),
2. validates it (no nulls/NaNs/infinities, no negative amounts, labels 0/1),
3. engineers features:
   - ``log_amount``      log1p(Amount), tames the heavy right tail,
   - ``is_zero_amount``  flag for zero-value transactions,
   - ``amount_zscore``   Amount standardised with dataset mean/std,
   - ``feature_hash``    SHA-256 of the 29 model inputs (duplicate key),
   - ``duplicate_count`` rows sharing the same feature vector (window),
4. writes Snappy Parquet partitioned by ``Class`` to
   ``data/processed/spark_features/``,
5. records throughput (rows/second) in ``metrics/spark_transform.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "spark_features"
REPORT_PATH = ROOT / "metrics" / "spark_transform.json"

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

logger = logging.getLogger("spark_transform")


def build_spark(app_name: str = "fraud-risk-spark-transform") -> SparkSession:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_raw(spark: SparkSession, csv_path: Path) -> DataFrame:
    """Ingest the raw CSV, tolerating an optional leading ``Time`` column."""
    header = csv_path.open(encoding="utf-8").readline().strip().split(",")
    header = [name.strip('"') for name in header]
    schema = StructType(
        [
            StructField(name, IntegerType() if name == "Class" else DoubleType(), False)
            for name in header
        ]
    )
    missing = sorted(set(FEATURE_COLUMNS + ["Class"]) - set(header))
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {missing}")

    return (
        spark.read.option("header", "true")
        .option("mode", "FAILFAST")
        .schema(schema)
        .csv(str(csv_path))
        .select(*FEATURE_COLUMNS, "Class")
    )


def validate(df: DataFrame) -> None:
    """Fail the stage on any data-quality violation."""
    bad_value = F.lit(False)
    for name in FEATURE_COLUMNS:
        col = F.col(name)
        bad_value = bad_value | col.isNull() | F.isnan(col) | col.isin(
            float("inf"), float("-inf")
        )
    checks = df.agg(
        F.sum(F.when(bad_value, 1).otherwise(0)).alias("bad_rows"),
        F.sum(F.when(F.col("Amount") < 0, 1).otherwise(0)).alias("negative_amount"),
        F.sum(F.when(~F.col("Class").isin(0, 1), 1).otherwise(0)).alias("bad_labels"),
    ).first()
    problems = {key: int(value or 0) for key, value in checks.asDict().items()}
    if any(problems.values()):
        raise ValueError(f"Spark validation failed: {problems}")


def transform(df: DataFrame) -> DataFrame:
    """Add engineered columns; model inputs are left unchanged."""
    stats = df.agg(
        F.mean("Amount").alias("mean"), F.stddev("Amount").alias("std")
    ).first()
    std = stats["std"] or 1.0

    feature_hash = F.sha2(
        F.concat_ws("|", *[F.col(name).cast("string") for name in FEATURE_COLUMNS]),
        256,
    )
    by_vector = Window.partitionBy("feature_hash")

    return (
        df.withColumn("log_amount", F.log1p("Amount"))
        .withColumn("is_zero_amount", (F.col("Amount") == 0).cast("int"))
        .withColumn("amount_zscore", (F.col("Amount") - F.lit(stats["mean"])) / F.lit(std))
        .withColumn("feature_hash", feature_hash)
        .withColumn("duplicate_count", F.count(F.lit(1)).over(by_vector))
    )


def run(spark: SparkSession, csv_path: Path, output_dir: Path) -> dict:
    """Execute ingest -> validate -> transform -> write; return a report."""
    started = perf_counter()
    raw = read_raw(spark, csv_path)
    validate(raw)
    enriched = transform(raw)

    enriched.write.mode("overwrite").partitionBy("Class").parquet(str(output_dir))
    duration = perf_counter() - started

    written = spark.read.parquet(str(output_dir))
    rows = written.count()
    class_counts = {
        str(row["Class"]): int(row["count"])
        for row in written.groupBy("Class").count().collect()
    }
    return {
        "source": "data/raw/creditcard.csv",
        "output": "data/processed/spark_features (Parquet, partitioned by Class)",
        "spark_version": spark.version,
        "rows": rows,
        "fraud_rows": class_counts.get("1", 0),
        "legitimate_rows": class_counts.get("0", 0),
        "engineered_columns": [
            "log_amount", "is_zero_amount", "amount_zscore",
            "feature_hash", "duplicate_count",
        ],
        "duplicated_rows": written.filter(F.col("duplicate_count") > 1).count(),
        "duration_seconds": round(duration, 3),
        "rows_per_second": round(rows / duration, 1) if duration else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Raw CSV not found: {CSV_PATH}")

    spark = build_spark()
    try:
        report = run(spark, CSV_PATH, OUTPUT_DIR)
    finally:
        spark.stop()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Spark transform: %s rows in %.1fs (%.0f rows/s) -> %s",
        f"{report['rows']:,}", report["duration_seconds"],
        report["rows_per_second"] or 0, OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
