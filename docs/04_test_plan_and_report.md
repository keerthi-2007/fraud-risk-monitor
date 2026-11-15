# AIOps Fraud Risk Monitor
## Test Plan, Test Cases and Test Report

**Version:** 1.0
**Date:** 14 November 2026
**Test framework:** pytest 9.2.0
**Environment:** Ubuntu VM, Python 3.14.4

---

## 1. Purpose

This document records the test strategy, implemented automated
test cases, executed results and manual integration evidence.

## 2. Testing Scope

The testing scope covers:

- Dataset integrity and split validation
- Prediction API behaviour and input validation
- Model loading and approved-model enforcement
- CSV scoring and drift integration
- Feedback persistence, security and metrics
- DVC pipeline status and recorded executions
- Spark throughput measurement
- Docker deployment and monitoring integration

## 3. Testing Approach

| Test level | Approach |
|---|---|
| Unit | pytest with isolated inputs and mocks |
| API | FastAPI test client and response assertions |
| Integration | Model, data and monitoring component checks |
| Operational | Docker, DVC, Spark and alerting verification |
| CI | Portable tests and Docker build jobs |

The final approved model and test evaluation are not rerun
as part of ordinary automated testing.

## 4. Entry and Exit Criteria

**Entry criteria:** Application source, dependencies,
datasets and approved model artifacts are available
for the relevant tests.

**Exit criteria:** Required automated tests pass, DVC
reports no outstanding pipeline changes, and targeted
operational checks complete successfully.

## 5. Automated Test Inventory

**Collected cases:** 111
**Test files:** 17

The following case IDs refer directly to pytest test names.

### Test Api

Source: `tests/test_api.py`

| Case ID | Automated test |
|---|---|
| AT-001 | `test_health` |
| AT-002 | `test_normal_prediction` |
| AT-003 | `test_review_prediction` |
| AT-004 | `test_missing_feature_rejected` |
| AT-005 | `test_negative_amount_rejected` |
| AT-006 | `test_extra_feature_rejected` |
| AT-007 | `test_csv_predictions` |
| AT-008 | `test_csv_missing_column` |
| AT-009 | `test_csv_invalid_extension` |
| AT-010 | `test_csv_negative_amount` |
| AT-011 | `test_csv_empty_file` |
| AT-012 | `test_decision_threshold_boundary` |
| AT-013 | `test_csv_validation_failure_metric` |
| AT-014 | `test_csv_validation_failure_rejects_missing_token` |
| AT-015 | `test_csv_validation_failure_rejects_wrong_token` |
| AT-016 | `test_csv_validation_failure_rejects_invalid_category` |
| AT-017 | `test_csv_includes_model_run_id` |

### Test Csv Drift Integration

Source: `tests/test_csv_drift_integration.py`

| Case ID | Automated test |
|---|---|
| AT-018 | `test_small_csv_scores_without_drift` |
| AT-019 | `test_large_csv_triggers_drift` |
| AT-020 | `test_successful_drift_updates_metrics` |
| AT-021 | `test_monitoring_failure_does_not_break_scoring` |

### Test Data

Source: `tests/test_data.py`

| Case ID | Automated test |
|---|---|
| AT-022 | `test_expected_columns` |
| AT-023 | `test_no_missing_values` |
| AT-024 | `test_valid_class_labels` |
| AT-025 | `test_manifest_counts` |
| AT-026 | `test_no_feature_overlap_between_splits` |

### Test Drift Detector

Source: `tests/test_drift_detector.py`

| Case ID | Automated test |
|---|---|
| AT-027 | `test_identical_distributions_have_zero_psi` |
| AT-028 | `test_shifted_distribution_detected` |
| AT-029 | `test_detect_drift_checks_all_features` |
| AT-030 | `test_detect_drift_flags_shifted_feature` |
| AT-031 | `test_missing_features_rejected` |
| AT-032 | `test_small_samples_rejected` |
| AT-033 | `test_nonfinite_values_rejected` |
| AT-034 | `test_constant_identical_feature_is_stable` |
| AT-035 | `test_constant_feature_change_detected` |

### Test Drift Monitor

Source: `tests/test_drift_monitor.py`

| Case ID | Automated test |
|---|---|
| AT-036 | `test_baseline_matches_direct_psi` |
| AT-037 | `test_missing_feature_rejected` |
| AT-038 | `test_insufficient_rows_rejected` |
| AT-039 | `test_invalid_values_rejected` |
| AT-040 | `test_valid_baseline_loads` |
| AT-041 | `test_invalid_baseline_rejected` |
| AT-042 | `test_constant_reference_change_detected` |

### Test Feedback

Source: `tests/test_feedback.py`

| Case ID | Automated test |
|---|---|
| AT-043 | `test_feedback_accepts_valid_token` |
| AT-044 | `test_feedback_rejects_missing_token` |
| AT-045 | `test_feedback_rejects_wrong_token` |
| AT-046 | `test_feedback_rejects_unconfigured_token` |

### Test Feedback Metrics

Source: `tests/test_feedback_metrics.py`

| Case ID | Automated test |
|---|---|
| AT-047 | `test_confusion_matrix_counts` |
| AT-048 | `test_model_versions_are_separated` |
| AT-049 | `test_duplicate_source_reference_excluded` |
| AT-050 | `test_precision_and_recall_from_sufficient_feedback` |
| AT-051 | `test_no_predicted_positives_suppresses_rates` |
| AT-052 | `test_invalid_predicted_decision_rejected` |
| AT-053 | `test_empty_feedback_file` |
| AT-054 | `test_feedback_file_loaded` |
| AT-055 | `test_invalid_json_rejected` |

### Test Feedback Metrics Api

Source: `tests/test_feedback_metrics_api.py`

| Case ID | Automated test |
|---|---|
| AT-056 | `test_metrics_export_feedback_counts` |
| AT-057 | `test_invalid_feedback_does_not_break_metrics` |
| AT-058 | `test_feedback_refresh_io_error_is_handled` |

### Test Feedback Monitor

Source: `tests/test_feedback_monitor.py`

| Case ID | Automated test |
|---|---|
| AT-059 | `test_feedback_metrics_exported` |
| AT-060 | `test_model_versions_and_duplicates` |
| AT-061 | `test_invalid_feedback_preserves_previous_metrics` |

### Test Integration

Source: `tests/test_integration.py`

| Case ID | Automated test |
|---|---|
| AT-062 | `test_deployed_api_matches_mlflow_model` |

### Test Model Loading

Source: `tests/test_model_loading.py`

| Case ID | Automated test |
|---|---|
| AT-063 | `test_load_model_from_run_uri` |
| AT-064 | `test_load_registered_model_version` |
| AT-065 | `test_reject_unsupported_model_uri` |
| AT-066 | `test_reject_unapproved_model_run` |
| AT-067 | `test_reject_wrong_decision_threshold` |

### Test Pipeline Client

Source: `tests/test_pipeline_client.py`

| Case ID | Automated test |
|---|---|
| AT-068 | `test_fetch_valid_snapshot` |
| AT-069 | `test_missing_token_rejected` |
| AT-070 | `test_http_error_propagated` |
| AT-071 | `test_connection_failure_propagated` |
| AT-072 | `test_invalid_snapshot_rejected[payload0]` |
| AT-073 | `test_invalid_snapshot_rejected[payload1]` |
| AT-074 | `test_invalid_snapshot_rejected[payload2]` |
| AT-075 | `test_invalid_snapshot_rejected[payload3]` |
| AT-076 | `test_invalid_snapshot_rejected[payload4]` |

### Test Pipeline History

Source: `tests/test_pipeline_history.py`

| Case ID | Automated test |
|---|---|
| AT-077 | `test_no_history_available` |
| AT-078 | `test_successful_execution_history` |
| AT-079 | `test_failed_execution_history` |
| AT-080 | `test_skipped_execution_history` |
| AT-081 | `test_only_latest_twenty_executions` |
| AT-082 | `test_invalid_stage_rejected` |
| AT-083 | `test_invalid_status_rejected` |
| AT-084 | `test_invalid_json_rejected` |

### Test Pipeline Status

Source: `tests/test_pipeline_status.py`

| Case ID | Automated test |
|---|---|
| AT-085 | `test_all_stages_up_to_date` |
| AT-086 | `test_modified_stage` |
| AT-087 | `test_missing_lock_entry` |
| AT-088 | `test_missing_spark_audit` |
| AT-089 | `test_invalid_dvc_json` |
| AT-090 | `test_dvc_command_failure` |

### Test Pipeline Status Api

Source: `tests/test_pipeline_status_api.py`

| Case ID | Automated test |
|---|---|
| AT-091 | `test_valid_snapshot` |
| AT-092 | `test_missing_token_rejected` |
| AT-093 | `test_wrong_token_rejected` |
| AT-094 | `test_missing_configuration_rejected` |
| AT-095 | `test_missing_snapshot_returns_503` |
| AT-096 | `test_invalid_json_returns_503` |
| AT-097 | `test_invalid_structure_returns_503` |

### Test Run Pipeline Stage

Source: `tests/test_run_pipeline_stage.py`

| Case ID | Automated test |
|---|---|
| AT-098 | `test_successful_execution` |
| AT-099 | `test_skipped_execution` |
| AT-100 | `test_failed_execution` |
| AT-101 | `test_missing_dvc_executable` |
| AT-102 | `test_unsupported_stage_never_executes` |
| AT-103 | `test_execution_history_appends` |

### Test Spark Throughput

Source: `tests/test_spark_throughput.py`

| Case ID | Automated test |
|---|---|
| AT-104 | `test_successful_spark_throughput` |
| AT-105 | `test_skipped_stage_has_no_throughput` |
| AT-106 | `test_failed_stage_has_no_throughput` |
| AT-107 | `test_missing_audit_file_has_no_throughput` |
| AT-108 | `test_invalid_row_count_has_no_throughput[0]` |
| AT-109 | `test_invalid_row_count_has_no_throughput[-1]` |
| AT-110 | `test_invalid_row_count_has_no_throughput[284807]` |
| AT-111 | `test_invalid_row_count_has_no_throughput[True]` |

## 6. Automated Test Execution Results

**Last verified full-suite result:**

- Tests executed: 111
- Tests passed: 111
- Tests failed: 0
- Dependency warnings: 2
- DVC status: Data and pipelines are up to date

The two warnings were MLflow-related dependency
deprecation warnings rather than test failures.

The results above were verified during the Git
pre-commit checks for the LLD documentation commit.

## 7. Portable CI Tests

The GitHub Actions workflow selects 105 portable tests.

All 105 passed locally, including a clean-checkout test
using only Git-tracked files.

Five dataset tests and one deployed-model integration
test remain part of the complete local test suite.

Both backend and frontend Docker images built locally.

A hosted GitHub Actions run has not yet been verified
because no Git remote is configured.

## 8. Manual Integration Test Cases

These cases were performed during project verification.
They are separate from the 111 automated pytest cases.

| Test ID | Component | Test procedure | Expected result | Observed result | Status |
|---|---|---|---|---|---|
| MT-01 | Docker deployment | Inspect running Compose services | Seven services running | Seven services running | PASS |
| MT-02 | API health | Request GET /health | HTTP 200 with approved model ID | HTTP 200, correct model ID | PASS |
| MT-03 | MLflow registry | Query registered FraudRiskClassifier | Version 1 available | Version 1, READY | PASS |
| MT-04 | DVC pipeline | Execute dvc status | No outstanding changes | Pipeline up to date | PASS |
| MT-05 | Spark audit | Execute allowlisted spark_audit stage | Successful execution and valid audit | Successful, 284807 rows | PASS |
| MT-06 | Spark throughput | Inspect recorded execution | Positive measured throughput | 18495.64 rows/second | PASS |
| MT-07 | Pipeline overview | Inspect Streamlit monitoring page | Actual stage history displayed | Execution and throughput displayed | PASS |
| MT-08 | Prometheus rules | Validate alert rule file with promtool | Valid rules | Five rules validated | PASS |
| MT-09 | Prometheus connectivity | Query Alertmanager discovery endpoint | Active Alertmanager found | One active, zero dropped | PASS |
| MT-10 | Alertmanager | Query readiness and configuration | Healthy and configured | HTTP 200, ready | PASS |
| MT-11 | Webhook notification | Submit controlled test alert | Alert recorded by webhook | One firing notification recorded | PASS |
| MT-12 | Alert cleanup | Inspect temporary alert after expiry | Test alert no longer active | Zero matching active alerts | PASS |
| MT-13 | Backend Docker build | Build backend image independently | Successful build | Image built successfully | PASS |
| MT-14 | Frontend Docker build | Build frontend image independently | Successful build | Image built successfully | PASS |
| MT-15 | GitHub Actions YAML | Parse and check workflow structure | Valid configured jobs | Portable tests and Docker jobs validated | PASS |
| MT-16 | Portable tests | Execute selected tests from Git-tracked checkout | Portable tests pass | 105 passed | PASS |

## 9. Spark Performance Test Report

A controlled execution was performed through the
allowlisted DVC stage runner.

| Measurement | Recorded value |
|---|---|
| Stage | spark_audit |
| Execution status | succeeded |
| Exit code | 0 |
| Execution duration | 15.398599 seconds |
| Rows processed | 284807 |
| Measured throughput | 18495.64 rows/second |
| Missing cells | 0 |
| Fraud records | 492 |

The recorded throughput measures end-to-end execution,
including DVC and Spark startup and shutdown overhead.

This is one observed run, not a statistical performance
benchmark or an estimate of production throughput.

## 10. Model Evaluation Evidence

The approved model is a Logistic Regression pipeline.

| Metric | Verified result |
|---|---|
| Test average precision | 0.733082 |
| Test ROC-AUC | 0.976206 |
| Validation fraud recall | 85/99 (85.859%) |
| Validation review alerts | 307 |
| Approved decision threshold | 0.87292890258894895 |

Model source run:
`27da3507bddb413bab9cd0a12aeb092a`

Final evaluation run:
`008277161614417d8da7e13727013676`

The final evaluation was previously completed and recorded
in MLflow. It was not rerun for this document.

The threshold was selected using validation data.
The held-out test set was used for final evaluation.

## 11. Monitoring Test Report

Prometheus successfully validated five alert rules.

Its Alertmanager discovery API reported:

- Active Alertmanagers: 1
- Dropped Alertmanagers: 0

Alertmanager readiness returned HTTP 200.

A controlled informational alert named
`AIOpsWebhookDeliveryTest` was submitted.

The webhook log recorded one matching notification:

- Status: firing
- Severity: info
- Summary: Controlled AIOps webhook delivery verification

The temporary test alert subsequently expired.
A follow-up query reported zero matching active alerts.

This verifies the Alertmanager-to-webhook delivery path.
It does not constitute a complete test of every
Prometheus alert condition.

## 12. CI/CD Test Report

The project implements continuous integration using GitHub Actions.

The workflow is defined in `.github/workflows/ci.yml` and contains two jobs:

1. Portable automated tests.
2. Backend and frontend Docker image builds.

### Verification Results

- 111 automated tests passed in the local Ubuntu environment.
- 105 portable automated tests passed locally.
- 105 portable tests passed from Git-tracked content.
- Backend Docker image built successfully.
- Frontend Docker image built successfully.
- GitHub Actions workflow configuration validated.
- GitHub-hosted CI workflow executed successfully.

### GitHub Actions Evidence

The latest three GitHub Actions executions were verified as successful.

| Run commit | Status |
|---|---|
| `7e4a8d3` | SUCCESS |
| `20a3467` | SUCCESS |
| `9d7f927` | SUCCESS |

The workflow and execution history are available in the repository's **Actions** tab:

https://github.com/keerthi-2007/fraud-risk-monitor/actions

### Conclusion

The project demonstrates automated testing and Docker build verification through both local checks and GitHub-hosted continuous integration.

The 105 portable CI tests are a subset of the 111 local automated tests and are not counted as additional independent test cases.

## 13. Test Summary

| Category | Executed | Passed | Failed |
|---|---:|---:|---:|
| Local automated pytest cases | 111 | 111 | 0 |
| Manual integration checks | 16 | 16 | 0 |

The 105 portable tests are a subset of the 111 local
automated cases and are not counted as additional tests.

Two MLflow-related deprecation warnings appeared during
local testing. They did not cause test failures.

## 14. Conclusion

The verified implementation passed 111 automated tests
and 16 documented manual integration checks.

The testing covers model inference, input validation,
feedback, drift monitoring, DVC pipeline monitoring,
Spark execution recording, deployment and alert delivery.

The available evidence supports a functioning local
academic AIOps application. Hosted CI execution and
production deployment are outside the verified scope.
