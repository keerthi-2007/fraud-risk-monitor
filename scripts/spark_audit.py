import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "spark_audit.json"

EXPECTED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Raw CSV not found: {CSV_PATH}")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("fraud-risk-spark-audit")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "FAILFAST")
            .csv(str(CSV_PATH))
        )

        expected = EXPECTED_COLUMNS + ["Class"]
        if set(df.columns) != set(expected):
            raise ValueError(
                f"Unexpected columns. Found {df.columns}; "
                f"expected {expected}"
            )

        numeric_columns = df.columns
        aggregate_expressions = [F.count(F.lit(1)).alias("row_count")]

        for name in numeric_columns:
            value = F.col(name).cast("double")
            aggregate_expressions.extend(
                [
                    F.sum(
                        F.when(
                            value.isNull() | F.isnan(value), 1
                        ).otherwise(0)
                    ).alias(f"missing_{name}"),
                    F.sum(
                        F.when(
                            value.isin(float("inf"), float("-inf")), 1
                        ).otherwise(0)
                    ).alias(f"non_finite_{name}"),
                ]
            )

        aggregate_expressions.extend(
            [
                F.sum(
                    F.when(F.col("Amount") < 0, 1).otherwise(0)
                ).alias("negative_amount_rows"),
                F.sum(
                    F.when(F.col("Amount") == 0, 1).otherwise(0)
                ).alias("zero_amount_rows"),
            ]
        )

        overall = df.agg(*aggregate_expressions).first().asDict()
        row_count = int(overall["row_count"])

        missing_cells = sum(
            int(overall[f"missing_{name}"] or 0)
            for name in numeric_columns
        )
        non_finite_values = sum(
            int(overall[f"non_finite_{name}"] or 0)
            for name in numeric_columns
        )

        class_counts = {
            str(int(row["Class"])): int(row["count"])
            for row in df.groupBy("Class").count().collect()
        }

        zero_amount_by_class = {
            str(int(row["Class"])): int(row["count"])
            for row in (
                df.filter(F.col("Amount") == 0)
                .groupBy("Class")
                .count()
                .collect()
            )
        }

        feature_columns = [name for name in df.columns if name != "Class"]
        feature_groups = df.groupBy(*feature_columns).agg(
            F.count(F.lit(1)).alias("group_rows"),
            F.countDistinct("Class").alias("label_count"),
        )

        group_stats = feature_groups.agg(
            F.count(F.lit(1)).alias("unique_feature_vectors"),
            F.sum(
                F.when(F.col("group_rows") > 1, 1).otherwise(0)
            ).alias("repeated_feature_groups"),
            F.sum(
                F.when(F.col("group_rows") > 1, F.col("group_rows"))
                .otherwise(0)
            ).alias("rows_in_repeated_groups"),
            F.sum(
                F.when(
                    F.col("group_rows") > 1,
                    F.col("group_rows") - 1,
                ).otherwise(0)
            ).alias("extra_rows"),
            F.max(
                F.when(
                    F.col("group_rows") > 1, F.col("group_rows")
                )
            ).alias("largest_repeated_group"),
            F.sum(
                F.when(F.col("label_count") > 1, 1).otherwise(0)
            ).alias("conflicting_label_groups"),
        ).first().asDict()

        report = {
            "source": "data/raw/creditcard.csv",
            "spark_version": spark.version,
            "rows": row_count,
            "legitimate_rows": class_counts.get("0", 0),
            "fraud_rows": class_counts.get("1", 0),
            "fraud_rate": (
                class_counts.get("1", 0) / row_count if row_count else 0.0
            ),
            "missing_cells": missing_cells,
            "non_finite_values": non_finite_values,
            "negative_amount_rows": int(
                overall["negative_amount_rows"] or 0
            ),
            "zero_amount_rows": int(overall["zero_amount_rows"] or 0),
            "zero_amount_by_class": {
                "0": zero_amount_by_class.get("0", 0),
                "1": zero_amount_by_class.get("1", 0),
            },
            "unique_feature_vectors": int(
                group_stats["unique_feature_vectors"] or 0
            ),
            "repeated_feature_groups": int(
                group_stats["repeated_feature_groups"] or 0
            ),
            "rows_in_repeated_groups": int(
                group_stats["rows_in_repeated_groups"] or 0
            ),
            "extra_rows_beyond_one_per_group": int(
                group_stats["extra_rows"] or 0
            ),
            "largest_repeated_group": int(
                group_stats["largest_repeated_group"] or 0
            ),
            "conflicting_label_groups": int(
                group_stats["conflicting_label_groups"] or 0
            ),
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print("# Spark data audit")
        print(f"Rows: {report['rows']:,}")
        print(
            f"Legitimate: {report['legitimate_rows']:,}; "
            f"fraud: {report['fraud_rows']:,}"
        )
        print(f"Missing cells: {report['missing_cells']:,}")
        print(
            "Repeated feature groups: "
            f"{report['repeated_feature_groups']:,}"
        )
        print(f"Conflicting label groups: {report['conflicting_label_groups']}")
        print(f"Saved Spark audit to: {OUTPUT_PATH}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
