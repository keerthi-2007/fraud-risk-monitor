"""Streamlit frontend for Fraud Risk Monitor.

Talks to the FastAPI backend only over REST (FRAUD_API_URL). Provides single-CSV
and bulk-ZIP scoring, per-row explanations, feedback capture and the pipeline
overview console.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import os
from datetime import datetime, timezone
from pipeline_client import fetch_pipeline_status, fetch_retraining_status

API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_ROWS = 100_000
MAX_CSV_FILES = 50

st.set_page_config(
    page_title="Fraud Risk Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f1f7f7 0%, #f7f8fb 360px);
    }
    .block-container {
        max-width: 1120px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #17243b;
        font-size: 15px;
        font-weight: 750;
    }
    .brand-icon {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: #087f68;
        color: white;
        font-size: 19px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        float: right;
        padding: 9px 13px;
        border: 1px solid #dce8e4;
        border-radius: 999px;
        background: white;
        color: #435268;
        font-size: 12px;
        font-weight: 650;
    }
    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #20a477;
    }
    .status-dot.offline { background: #d45252; }
    .eyebrow {
        margin: 0;
        color: #087f68;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .14em;
    }
    .hero-icon {
        display: grid;
        place-items: center;
        width: 112px;
        height: 112px;
        margin: 0 auto;
        border: 1px solid #d8e9e4;
        border-radius: 32px;
        background: linear-gradient(145deg, #ffffff, #e2f2ed);
        box-shadow: 0 16px 40px rgba(8, 127, 104, .12);
        font-size: 58px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dfe8ed;
        border-radius: 16px;
        background: rgba(255, 255, 255, .92);
    }
    div[data-testid="stMetric"] {
        padding: 16px;
        border: 1px solid #e0e7ef;
        border-radius: 14px;
        background: white;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #9fb9b2;
        border-radius: 12px;
        background: #fbfdfd;
    }
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button {
        min-height: 44px;
        border-radius: 10px;
        font-weight: 700;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero-icon { width: 76px; height: 76px; border-radius: 24px; font-size: 40px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def validation_error_reason(error):
    """Map user-facing validation errors to monitoring categories."""
    if "Missing columns:" in error:
        return "missing_columns"
    if "Unexpected columns:" in error:
        return "unexpected_columns"
    if "non-numeric, or infinite" in error:
        return "invalid_values"
    if "Amount cannot be negative" in error:
        return "negative_amount"
    if "no transaction rows" in error:
        return "empty_file"
    if "limit" in error or "at most" in error or "more than" in error:
        return "file_limit"
    if "ZIP archive" in error or "encrypted" in error:
        return "invalid_archive"
    if "couldn't read" in error.lower():
        return "unreadable_file"
    return "other"


def report_validation_failure(upload_type, error):
    """Report once for the current rejected upload selection."""
    state_key = f"reported_invalid_{upload_type}"
    if st.session_state.get(state_key):
        return

    token = os.getenv("MONITORING_TOKEN")
    if not token:
        return

    try:
        response = requests.post(
            f"{API_URL}/monitoring/validation-failure",
            json={
                "upload_type": upload_type,
                "reason": validation_error_reason(error),
            },
            headers={"X-Monitoring-Token": token},
            timeout=5,
        )
        if response.status_code == 204:
            st.session_state[state_key] = True
    except requests.RequestException:
        pass


def reset_validation_failure(upload_type):
    """Allow a newly selected upload to generate a new event."""
    st.session_state.pop(f"reported_invalid_{upload_type}", None)


def validate_frame(raw_df):
    """Validate one CSV's columns and transaction values."""
    if raw_df.empty:
        return None, "The CSV has a header but no transaction rows."

    missing = [name for name in FEATURE_COLUMNS if name not in raw_df.columns]
    unexpected = [name for name in raw_df.columns if name not in FEATURE_COLUMNS]

    if missing or unexpected:
        problems = []
        if missing:
            problems.append("Missing columns: " + ", ".join(missing))
        if unexpected:
            problems.append("Unexpected columns: " + ", ".join(unexpected))
        problems.append(
            "Use only V1 through V28 and Amount. Remove Class labels and "
            "columns from previously scored prediction files."
        )
        return None, " ".join(problems)

    numeric_df = raw_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype="float64")
    bad_numeric = ~np.isfinite(values).all(axis=1)

    if bad_numeric.any():
        row_numbers = (np.flatnonzero(bad_numeric)[:8] + 2).tolist()
        return None, (
            "Some feature cells are blank, non-numeric, or infinite. "
            f"Check CSV row(s): {row_numbers}."
        )

    negative_amount = numeric_df["Amount"] < 0
    if negative_amount.any():
        row_numbers = (np.flatnonzero(negative_amount.to_numpy())[:8] + 2).tolist()
        return None, f"Amount cannot be negative. Check CSV row(s): {row_numbers}."

    return numeric_df.reset_index(drop=True), None

def read_single_csv(uploaded_file):
    """Read and validate one uploaded CSV; return (DataFrame, None) or (None, error message)."""
    if uploaded_file.size > MAX_FILE_BYTES:
        return None, "This file is larger than the 50 MB upload limit."

    try:
        raw_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        return None, f"We couldn't read this CSV file: {exc}"

    if len(raw_df) > MAX_ROWS:
        return None, f"A single CSV can contain at most {MAX_ROWS:,} transaction rows."

    return validate_frame(raw_df)


def read_bulk_zip(uploaded_file):
    """Read and validate every CSV inside an uploaded ZIP; return (files, None) or (None, error)."""
    if uploaded_file.size > MAX_FILE_BYTES:
        return None, "The ZIP file is larger than the 50 MB upload limit."

    try:
        archive = zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()))
    except (zipfile.BadZipFile, OSError):
        return None, "This file is not a readable ZIP archive."

    csv_members = [
        item
        for item in archive.infolist()
        if not item.is_dir()
        and item.filename.lower().endswith(".csv")
        and not item.filename.startswith("__MACOSX/")
        and not item.filename.rsplit("/", 1)[-1].startswith("._")
        and not item.filename.rsplit("/", 1)[-1].startswith(".")
    ]

    if not csv_members:
        return None, "The ZIP must contain at least one CSV file."

    if len(csv_members) > MAX_CSV_FILES:
        return None, f"The ZIP can contain at most {MAX_CSV_FILES} CSV files."

    if sum(item.file_size for item in csv_members) > MAX_FILE_BYTES:
        return None, "The CSV files inside the ZIP exceed the 50 MB combined limit."

    validated_files = []
    total_rows = 0

    for member in csv_members:
        if member.flag_bits & 0x1:
            return None, f"'{member.filename}' is encrypted. Upload an unencrypted ZIP."

        try:
            raw_df = pd.read_csv(io.BytesIO(archive.read(member)))
        except Exception as exc:
            return None, f"Couldn't read '{member.filename}': {exc}"

        numeric_df, error = validate_frame(raw_df)
        if error:
            return None, f"File '{member.filename}': {error}"

        total_rows += len(numeric_df)
        if total_rows > MAX_ROWS:
            return None, f"The ZIP contains more than {MAX_ROWS:,} transaction rows."

        # Keep each input CSV as a separate item; don't concatenate them.
        validated_files.append((member.filename, numeric_df))

    return validated_files, None


def api_error(response):
    """Extract a readable error message from a failed backend response."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return str(detail)

def fetch_explanation(features):
    """Ask the backend /explain endpoint why a transaction scored as it did.

    Returns (body, error_message); exactly one of them is None.
    """
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json={"features": features},
            timeout=30,
        )
    except requests.RequestException as exc:
        return None, f"Could not reach the prediction API: {exc}"

    if not response.ok:
        return None, f"Explanation failed ({response.status_code}): {api_error(response)}"
    return response.json(), None


def render_explanation_panel(saved, scored_df):
    """Plain-language view of the features that drove one row's score."""
    with st.expander("🔍 Why was a row scored this way?", expanded=False):
        st.caption(
            "Pick a row to see which values pushed its risk score up or down, "
            "compared with a typical transaction from the training data."
        )
        row_number = st.number_input(
            "Transaction row to explain",
            min_value=1,
            max_value=len(scored_df),
            value=1,
            step=1,
            key=f"explain_row_{saved['filename']}_{len(scored_df)}",
        )
        input_record = saved["input_records"][int(row_number) - 1]
        features = {name: float(input_record[name]) for name in FEATURE_COLUMNS}
        cache = st.session_state.setdefault("explanations", {})
        cache_key = repr(tuple(features.values()))

        if st.button("Explain this row", key="explain_button"):
            body, error = fetch_explanation(features)
            if error:
                st.error(error)
            else:
                cache[cache_key] = body

        body = cache.get(cache_key)
        if body is None:
            return

        top = body["contributions"][:8]
        decision_text = (
            "flagged for review" if body["decision"] == "review" else "normal"
        )
        st.markdown(
            f"Row {int(row_number)} scored **{body['risk_score']:.3f}** "
            f"(threshold {body['decision_threshold']:.3f}) and is "
            f"**{decision_text}**. A typical transaction scores "
            f"{body['reference_score']:.3f}."
        )
        chart = pd.DataFrame(
            {
                "Feature": [item["feature"] for item in top],
                "Effect on risk score": [item["contribution"] for item in top],
            }
        ).set_index("Feature")
        st.bar_chart(chart, horizontal=True)

        raisers = [item["feature"] for item in top if item["contribution"] > 0][:3]
        if raisers:
            st.info(
                "Biggest factors raising the score: " + ", ".join(raisers) + ". "
                "Bars to the right raised the risk; bars to the left lowered it."
            )
        else:
            st.info("No single value raised this transaction's risk score.")
        st.caption(
            "Method: each value is swapped for its typical (median) value one at "
            "a time and the score change is measured. This explains the model, "
            "not whether the transaction is actually fraudulent."
        )


def render_bulk_feedback_form(features, scored_row, source_file, row_number, file_index):
    """Render the feedback form for one scored row in bulk mode."""
    feature_values = {
        name: float(features[name])
        for name in FEATURE_COLUMNS
    }
    score = float(scored_row["risk_score"])
    threshold = float(scored_row["decision_threshold"])
    decision = str(scored_row["decision"]).lower()

    run_value = scored_row.get("model_run_id")
    model_run_id = (
        None if run_value is None or pd.isna(run_value) else str(run_value)
    )

    context = (
        source_file,
        int(row_number),
        score,
        threshold,
        decision,
        tuple(feature_values[name] for name in FEATURE_COLUMNS),
    )
    feedback_key = repr(context)
    saved_feedback = st.session_state.setdefault("bulk_feedback_records", {})

    if feedback_key in saved_feedback:
        if saved_feedback[feedback_key]:
            st.success("Feedback saved. The prediction matched the reported outcome.")
        else:
            st.info("Feedback saved. The prediction did not match the reported outcome.")
        return

    st.caption(
        "Report an outcome only when you have reliable confirmation. "
        "The checkbox records your acknowledgement; the app cannot verify it."
    )

    form_key = f"bulk_feedback_{file_index}_{row_number}"
    with st.form(key=f"{form_key}_form"):
        label = st.selectbox(
            "Confirmed transaction outcome",
            ["— Select an outcome —", "legitimate", "fraud"],
            key=f"{form_key}_label",
        )
        confirmed = st.checkbox(
            "I have reliable confirmation of this outcome.",
            key=f"{form_key}_confirmed",
        )
        submitted = st.form_submit_button("Save feedback")

    if not submitted:
        return
    if label == "— Select an outcome —":
        st.error("Choose the confirmed outcome first.")
        return
    if not confirmed:
        st.error("Confirm that you have reliable outcome information.")
        return

    payload = {
        "features": feature_values,
        "risk_score": score,
        "decision_threshold": threshold,
        "predicted_decision": decision,
        "corrected_label": label,
        "model_run_id": model_run_id,
        "source_file": source_file,
        "row_number": int(row_number),
    }

    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json=payload,
            headers={"X-Feedback-Token": os.environ["FEEDBACK_TOKEN"]},
            timeout=15,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the feedback API: {exc}")
        return

    if response.status_code != 201:
        st.error(
            f"Feedback was not saved ({response.status_code}): "
            f"{api_error(response)}"
        )
        return

    was_correct = bool(response.json()["prediction_was_correct"])
    saved_feedback[feedback_key] = was_correct
    if was_correct:
        st.success("Feedback saved. The prediction matched the reported outcome.")
    else:
        st.info("Feedback saved. The prediction did not match the reported outcome.")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    api_online = health_response.ok
except requests.RequestException:
    api_online = False

header_left, header_right = st.columns([0.72, 0.28], vertical_alignment="center")

with header_left:
    st.markdown(
        '<div class="brand"><span class="brand-icon">🛡️</span>'
        "<span>Fraud Risk Monitor</span></div>",
        unsafe_allow_html=True,
    )

with header_right:
    status_class = "" if api_online else "offline"
    status_text = "API connected" if api_online else "API unavailable"
    st.markdown(
        f'<div class="status-pill"><span class="status-dot {status_class}"></span>'
        f"{status_text}</div>",
        unsafe_allow_html=True,
    )

st.write("")
hero_text, hero_art = st.columns([3.2, 1], vertical_alignment="center")

with hero_text:
    st.markdown('<p class="eyebrow">TRANSACTION RISK ANALYSIS</p>', unsafe_allow_html=True)
    st.title("Know where to look.")
    st.markdown(
        "Score one CSV or a ZIP of seperate CSV files, review the results, "
        "and download the predictions."
    )

with hero_art:
    st.markdown('<div class="hero-icon">🔐</div>', unsafe_allow_html=True)

if not api_online:
    st.warning("The prediction API is offline. Keep FastAPI running on port 8000.")

single_tab, bulk_tab, pipeline_tab = st.tabs(["🔎  Single CSV", "📦  Bulk ZIP upload", "🔄  Pipeline Overview"])

with single_tab:
    with st.container(border=True):
        st.subheader("Score one CSV")
        st.caption(
            f"Upload one CSV containing up to {MAX_ROWS:,} transaction rows."
        )

        single_file = st.file_uploader(
            "Choose a transaction CSV",
            type=["csv"],
            key="single_upload",
            on_change=reset_validation_failure,
            args=("single_csv",),
        )

        single_df, single_error = (
            read_single_csv(single_file) if single_file else (None, None)
        )

        if single_file is not None:
            if single_error:
                st.error(f"⚠️ {single_error}")
                report_validation_failure("single_csv", single_error)
            else:
                st.success(
                    f"✅ CSV looks good: {len(single_df):,} valid transaction rows."
                )

        single_clicked = st.button(
            "✨ Score this CSV",
            type="primary",
            key="single_score",
            disabled=(
                not api_online
                or single_df is None
                or single_error is not None
            ),
        )

        if single_clicked and single_df is not None:
            csv_bytes = single_df.to_csv(index=False).encode("utf-8")

            try:
                with st.spinner(f"Scoring {len(single_df):,} transaction rows…"):
                    response = requests.post(
                        f"{API_URL}/predict/csv",
                        files={
                            "file": (
                                single_file.name,
                                csv_bytes,
                                "text/csv",
                            )
                        },
                        timeout=180,
                    )

                if not response.ok:
                    st.error(
                        f"Scoring failed ({response.status_code}): "
                        f"{api_error(response)}"
                    )
                else:
                    scored_df = pd.read_csv(io.BytesIO(response.content))
                    required_output_columns = {
                        "risk_score",
                        "decision_threshold",
                        "decision",
                    }
                    missing_output_columns = (
                        required_output_columns - set(scored_df.columns)
                    )

                    if missing_output_columns:
                        st.error(
                            "The API response is missing expected result columns: "
                            + ", ".join(sorted(missing_output_columns))
                        )
                    elif len(scored_df) != len(single_df):
                        st.error(
                            f"The API returned {len(scored_df):,} rows for "
                            f"{len(single_df):,} input transactions."
                        )
                    else:
                        st.session_state["single_result"] = {
                            "filename": single_file.name,
                            "input_records": single_df.to_dict(orient="records"),
                            "scored_df": scored_df,
                        }

            except requests.RequestException as exc:
                st.error(f"Could not reach the prediction API: {exc}")
            except Exception as exc:
                st.error(f"Could not read the API results: {exc}")

    if "single_result" in st.session_state:
        saved = st.session_state["single_result"]
        scored_df = saved["scored_df"]
        decisions = scored_df["decision"].astype(str).str.lower()

        review_count = int((decisions == "review").sum())
        normal_count = int((decisions == "normal").sum())

        st.write("")
        st.subheader("Prediction summary")

        rows_col, review_col, normal_col = st.columns(3)
        rows_col.metric("Rows processed", f"{len(scored_df):,}")
        review_col.metric("Flagged for review", f"{review_count:,}")
        normal_col.metric("Normal", f"{normal_count:,}")

        st.subheader("Prediction preview")
        st.dataframe(
            scored_df.head(20),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Showing up to 20 rows. Download the CSV to view all results.")

        output_csv = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download scored CSV",
            data=output_csv,
            file_name="fraud_risk_single_csv_predictions.csv",
            mime="text/csv",
        )

        render_explanation_panel(saved, scored_df)

        with st.expander("📝 Report a confirmed outcome", expanded=False):
            st.caption(
                "Choose a row and report its actual outcome only when you have "
                "reliable confirmation. The checkbox is your acknowledgement; "
                "the app cannot independently verify the outcome. Feedback is "
                "stored for later review and does not retrain the model now."
            )

            row_number = st.number_input(
                "Transaction row to report",
                min_value=1,
                max_value=len(scored_df),
                value=1,
                step=1,
                key=f"single_feedback_row_{saved['filename']}_{len(scored_df)}",
            )
            row_index = int(row_number) - 1
            input_record = saved["input_records"][row_index]
            scored_row = scored_df.iloc[row_index]

            score = float(scored_row["risk_score"])
            threshold = float(scored_row["decision_threshold"])
            predicted_decision = str(scored_row["decision"]).lower()
            model_run_value = scored_row.get("model_run_id")
            model_run_id = (
                None
                if model_run_value is None or pd.isna(model_run_value)
                else str(model_run_value)
            )

            feedback_context = (
                saved["filename"],
                int(row_number),
                score,
                threshold,
                predicted_decision,
                tuple(float(input_record[name]) for name in FEATURE_COLUMNS),
            )
            feedback_key = repr(feedback_context)
            feedback_records = st.session_state.setdefault(
                "single_feedback_records", {}
            )

            if feedback_key in feedback_records:
                if feedback_records[feedback_key]:
                    st.success("Feedback saved. The prediction matched the reported outcome.")
                else:
                    st.info("Feedback saved. The prediction did not match the reported outcome.")
            else:
                with st.form("single_feedback_form"):
                    corrected_label = st.selectbox(
                        "Confirmed transaction outcome",
                        ["— Select an outcome —", "legitimate", "fraud"],
                    )
                    confirmed = st.checkbox(
                        "I have reliable confirmation of this outcome."
                    )
                    submit_feedback = st.form_submit_button("Save feedback")

                if submit_feedback:
                    if corrected_label == "— Select an outcome —":
                        st.error("Choose the confirmed outcome first.")
                    elif not confirmed:
                        st.error("Confirm that you have reliable outcome information.")
                    else:
                        feedback_payload = {
                            "features": {
                                name: float(input_record[name])
                                for name in FEATURE_COLUMNS
                            },
                            "risk_score": score,
                            "decision_threshold": threshold,
                            "predicted_decision": predicted_decision,
                            "corrected_label": corrected_label,
                            "model_run_id": model_run_id,
                            "source_file": saved["filename"],
                            "row_number": int(row_number),
                        }

                        try:
                            feedback_response = requests.post(
                                f"{API_URL}/feedback",
                                json=feedback_payload,
                                headers={"X-Feedback-Token": os.environ["FEEDBACK_TOKEN"]},
                                timeout=15,
                            )

                            if feedback_response.status_code == 201:
                                feedback_body = feedback_response.json()
                                was_correct = bool(
                                    feedback_body["prediction_was_correct"]
                                )
                                feedback_records[feedback_key] = was_correct

                                if was_correct:
                                    st.success(
                                        "Feedback saved. The prediction matched "
                                        "the reported outcome."
                                    )
                                else:
                                    st.info(
                                        "Feedback saved. The prediction did not "
                                        "match the reported outcome."
                                    )
                            else:
                                st.error(
                                    f"Feedback was not saved "
                                    f"({feedback_response.status_code}): "
                                    f"{api_error(feedback_response)}"
                                )
                        except requests.RequestException as exc:
                            st.error(f"Could not reach the feedback API: {exc}")

with bulk_tab:
    with st.container(border=True):
        st.subheader("Score multiple files")
        st.caption(
            f"Upload one ZIP containing up to {MAX_CSV_FILES} CSV files. "
            "Each CSV may contain one or more transactions."
        )

        bulk_file = st.file_uploader(
            "Choose a ZIP archive",
            type=["zip"],
            key="bulk_zip_upload",
            on_change=reset_validation_failure,
            args=("bulk_zip",),
        )

        if bulk_file is not None:
            bulk_files, bulk_error = read_bulk_zip(bulk_file)
        else:
            bulk_files, bulk_error = None, None

        if bulk_file is not None:
            if bulk_error:
                st.error(f"⚠️ {bulk_error}")
                report_validation_failure("bulk_zip", bulk_error)
            else:
                total_rows = sum(len(frame) for _, frame in bulk_files)
                st.success(
                    f"✅ ZIP looks good: {len(bulk_files)} CSV files and "
                    f"{total_rows:,} valid transaction rows."
                )

        bulk_clicked = st.button(
            "🚀 Score each file separately",
            type="primary",
            key="bulk_score",
            disabled=not api_online or bulk_files is None or bulk_error is not None,
        )

        if bulk_clicked and bulk_files is not None:
            outputs = []
            errors = []
            progress = st.progress(0, text="Starting file scoring…")

            for index, (source_name, source_df) in enumerate(bulk_files, start=1):
                progress.progress(
                    (index - 1) / len(bulk_files),
                    text=f"Scoring {source_name} ({index}/{len(bulk_files)})…",
                )

                csv_bytes = source_df.to_csv(index=False).encode("utf-8")

                try:
                    response = requests.post(
                        f"{API_URL}/predict/csv",
                        files={
                            "file": (
                                "transactions.csv",
                                csv_bytes,
                                "text/csv",
                            )
                        },
                        timeout=180,
                    )
                except requests.RequestException as exc:
                    errors.append(f"{source_name}: couldn't reach the API: {exc}")
                    continue

                if not response.ok:
                    errors.append(
                        f"{source_name}: API returned "
                        f"{response.status_code}: {api_error(response)}"
                    )
                    continue

                try:
                    scored_df = pd.read_csv(io.BytesIO(response.content))
                except Exception as exc:
                    errors.append(f"{source_name}: couldn't read the API result: {exc}")
                    continue

                if len(scored_df) != len(source_df):
                    errors.append(
                        f"{source_name}: API returned {len(scored_df)} rows "
                        f"for {len(source_df)} input rows."
                    )
                    continue

                decisions = scored_df.get(
                    "decision", pd.Series(dtype="str")
                ).astype(str).str.lower()

                review_count = int(
                    response.headers.get(
                        "x-review-count",
                        int((decisions == "review").sum()),
                    )
                )
                normal_count = int(
                    response.headers.get(
                        "x-normal-count",
                        int((decisions == "normal").sum()),
                    )
                )

                original_name = source_name.rsplit("/", 1)[-1]
                stem = original_name.rsplit(".", 1)[0]
                safe_stem = "".join(
                    char if char.isalnum() or char in "-_." else "_"
                    for char in stem
                ).strip("._") or "transactions"

                output_name = f"{index:02d}_{safe_stem}_predictions.csv"
                output_csv = scored_df.to_csv(index=False).encode("utf-8")

                outputs.append(
                    {
                        "source_name": source_name,
                        "output_name": output_name,
                        "csv": output_csv,
                        "dataframe": scored_df,
                        "rows": int(
                            response.headers.get("x-processed-rows", len(scored_df))
                        ),
                        "review": review_count,
                        "normal": normal_count,
                        "input_dataframe": source_df,
                    }
                )

            progress.progress(1.0, text="File scoring finished.")
            progress.empty()

            st.session_state["bulk_result"] = {
                "zip_name": bulk_file.name,
                "outputs": outputs,
                "errors": errors,
            }

    if "bulk_result" in st.session_state:
        result = st.session_state["bulk_result"]
        outputs = result["outputs"]

        st.write("")
        st.subheader("Per-file results")
        st.caption(f"Results for {result['zip_name']}")

        if result["errors"]:
            st.warning("Some files could not be scored:")
            for error in result["errors"]:
                st.write(f"• {error}")

        if outputs:
            total_rows = sum(item["rows"] for item in outputs)
            total_review = sum(item["review"] for item in outputs)
            total_normal = sum(item["normal"] for item in outputs)

            col1, col2, col3 = st.columns(3)
            col1.metric("Files scored", len(outputs))
            col2.metric("Transactions processed", f"{total_rows:,}")
            col3.metric("Flagged for review", f"{total_review:,}")

            summary = pd.DataFrame(
                [
                    {
                        "Input file": item["source_name"],
                        "Rows": item["rows"],
                        "Review": item["review"],
                        "Normal": item["normal"],
                    }
                    for item in outputs
                ]
            )
            st.dataframe(summary, hide_index=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as result_zip:
                for item in outputs:
                    result_zip.writestr(item["output_name"], item["csv"])

            st.download_button(
                "⬇️ Download all per-file results as ZIP",
                data=zip_buffer.getvalue(),
                file_name="fraud_risk_results_by_file.zip",
                mime="application/zip",
                type="primary",
            )

            st.subheader("Review each file")
            for index, item in enumerate(outputs, start=1):
                with st.expander(
                    f"📄 {item['source_name']} · {item['rows']:,} rows"
                ):
                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Rows", f"{item['rows']:,}")
                    metric2.metric("Review", f"{item['review']:,}")
                    metric3.metric("Normal", f"{item['normal']:,}")

                    st.download_button(
                        "Download this file's predictions",
                        data=item["csv"],
                        file_name=item["output_name"],
                        mime="text/csv",
                        key=f"download_file_{index}",
                    )

                    scored_df = item["dataframe"]
                    threshold_name = next(
                        (
                            name
                            for name in ("decision_threshold", "threshold")
                            if name in scored_df.columns
                        ),
                        None,
                    )
                    preview_columns = [
                        name
                        for name in (
                            "Amount",
                            "risk_score",
                            threshold_name,
                            "decision",
                        )
                        if name and name in scored_df.columns
                    ]

                    if preview_columns:
                        st.dataframe(
                            scored_df[preview_columns].head(20),
                            hide_index=True,
                        )
                    else:
                        st.dataframe(scored_df.head(20), hide_index=True)

                    feedback_row_number = st.number_input(
                        "Transaction row to report",
                        min_value=1,
                        max_value=len(scored_df),
                        value=1,
                        step=1,
                        key=f"bulk_feedback_row_{index}",
                    )
                    feedback_row_index = int(feedback_row_number) - 1

                    render_bulk_feedback_form(
                        features=item["input_dataframe"]
                        .iloc[feedback_row_index]
                        .to_dict(),
                        scored_row=scored_df.iloc[feedback_row_index],
                        source_file=item["source_name"],
                        row_number=int(feedback_row_number),
                        file_index=index,
                    )

            st.caption(
                f"Across the ZIP: {total_normal:,} normal and "
                f"{total_review:,} flagged for review. Each CSV stays separate "
                "in the downloaded results ZIP."
            )


with pipeline_tab:
    st.subheader("Pipeline Monitor")
    st.caption("Recorded DVC pipeline state and Spark audit results.")

    try:
        snapshot = fetch_pipeline_status(
            API_URL,
            os.getenv("MONITORING_TOKEN"),
        )
    except (requests.RequestException, ValueError) as exc:
        st.warning(
            "Pipeline snapshot unavailable. "
            "Prediction services are unaffected."
        )
        st.caption(f"Status retrieval: {type(exc).__name__}")
    else:
        generated_at = snapshot["generated_at"]
        st.caption(f"Snapshot generated (UTC): {generated_at}")

        try:
            generated_time = datetime.fromisoformat(
                generated_at.replace("Z", "+00:00")
            )
            if generated_time.tzinfo is None:
                raise ValueError("Missing snapshot timezone")

            age_seconds = (
                datetime.now(timezone.utc) - generated_time
            ).total_seconds()

            if age_seconds > 3600:
                st.warning(
                    "This pipeline snapshot is over one hour old. "
                    "Its status may no longer reflect the current workspace."
                )
            elif age_seconds < -60:
                st.warning("Snapshot timestamp is in the future.")
        except ValueError:
            st.warning("Snapshot timestamp could not be validated.")

        stages = snapshot["stages"]
        up_to_date = snapshot["overall_up_to_date"]

        if up_to_date:
            st.success(
                "DVC snapshot: all recorded stages are up to date."
            )
        else:
            st.warning(
                "DVC snapshot: one or more stages require attention."
            )

        spark = snapshot.get("spark_audit") or {}

        col1, col2, col3 = st.columns(3)
        col1.metric("DVC stages", len(stages))
        col2.metric(
            "Stages up to date",
            sum(bool(s.get("up_to_date")) for s in stages),
        )
        col3.metric(
            "Spark rows audited",
            f"{spark['rows']:,}" if isinstance(
                spark.get("rows"), int
            ) else "Unavailable",
        )

        st.markdown("#### DVC pipeline stages")

        for stage in stages:
            name = stage.get("name", "Unnamed stage")
            current = stage.get("up_to_date") is True
            recorded = stage.get("recorded_in_lock") is True

            with st.container(border=True):
                st.markdown(f"**{name}**")

                if current:
                    st.success("Up to date in recorded DVC snapshot")
                elif not recorded:
                    st.error("Stage missing from DVC lockfile")
                else:
                    st.warning("Stage has outstanding DVC changes")

                dependencies = stage.get("dependencies", [])
                st.caption(
                    f"Recorded dependencies: {len(dependencies)}"
                )

        st.markdown("#### Spark data audit")
        if spark:
            st.write(
                f"Spark version: {spark.get('spark_version', 'Unknown')}"
            )
            st.write(
                f"Fraud records: {spark.get('fraud_rows', 'Unknown')}"
            )
            st.write(
                f"Missing cells: {spark.get('missing_cells', 'Unknown')}"
            )
        else:
            st.info("No Spark audit results available.")

        st.markdown("#### Recorded pipeline executions")

        executions = snapshot.get("recent_executions", [])

        if executions:
            latest = executions[-1]
            stage_name = latest.get("stage", "Unknown")
            stage_status = latest.get("status", "unknown")

            st.markdown(f"**Latest recorded stage:** `{stage_name}`")

            if stage_status == "succeeded":
                st.success("Execution succeeded")
            elif stage_status == "failed":
                st.error("Execution failed")
            elif stage_status == "skipped":
                st.info("Execution skipped — stage was up to date")
            else:
                st.warning("Execution status unavailable")

            duration = latest.get("duration_seconds")
            rows = latest.get("rows_processed")
            throughput = latest.get("throughput_rows_per_second")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Duration",
                f"{duration:.2f} s"
                if isinstance(duration, (int, float))
                else "Unavailable",
            )
            metric2.metric(
                "Rows processed",
                f"{rows:,}"
                if isinstance(rows, int)
                else "Unavailable",
            )
            metric3.metric(
                "Throughput",
                f"{throughput:,.2f} rows/s"
                if isinstance(throughput, (int, float))
                else "Unavailable",
            )

            st.caption(
                "Measured throughput includes DVC execution and "
                "Spark startup/shutdown overhead."
            )

            st.markdown("#### Recent execution history")

            history_rows = [
                {
                    "Stage": item.get("stage"),
                    "Status": item.get("status"),
                    "Started (UTC)": item.get("started_at"),
                    "Duration (s)": item.get("duration_seconds"),
                    "Rows": item.get("rows_processed"),
                    "Rows/s": item.get("throughput_rows_per_second"),
                }
                for item in reversed(executions)
            ]

            st.dataframe(
                history_rows,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No recorded pipeline executions yet. "
                "The DVC snapshot only reports dependency status."
            )

        st.markdown("#### Model workflow")
        st.info(
            "Training and final evaluation are controlled MLflow "
            "operations outside the DVC DAG. "
            "The approved model is served by FastAPI."
        )

        st.caption(
            "This is a saved pipeline snapshot, not a live job monitor. "
            "Recorded execution history and measured throughput are shown "
            "when available. Only runs captured by the local stage runner "
            "appear here. Feedback-triggered candidate retraining runs separately from DVC."
        )

    st.markdown("#### Automatic Feedback Retraining")
    st.caption(
        "The background worker checks approved transaction feedback "
        "and trains candidate models separately from the DVC pipeline."
    )

    try:
        retraining = fetch_retraining_status(
            API_URL,
            os.getenv("MONITORING_TOKEN"),
        )
    except (requests.RequestException, ValueError):
        st.info(
            "Retraining monitoring is not available yet. "
            "Prediction and feedback services are unaffected."
        )
    else:
        received = retraining["feedback_received"]
        unapproved = retraining["not_approved"]
        eligible = retraining["eligible"]
        required = retraining["required_feedback"]
        fraud = retraining["eligible_fraud"]
        legitimate = retraining["eligible_legitimate"]
        per_class = retraining["required_per_class"]

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)

            col1.metric("Feedback received", received)
            col2.metric("Awaiting approval", unapproved)
            col3.metric(
                "Eligible for retraining",
                f"{eligible} / {required}",
            )

            st.progress(
                min(eligible / required, 1.0),
                text=(
                    f"Retraining eligibility: "
                    f"{eligible} of {required} records"
                ),
            )

            st.caption(
                f"Eligible fraud: {fraud} | "
                f"Eligible legitimate: {legitimate} | "
                f"Minimum required per class: {per_class}"
            )

            if retraining["threshold_met"]:
                st.success(
                    "Feedback requirements met. Candidate retraining "
                    "is eligible to run; this does not confirm "
                    "training or promotion has completed."
                )
            else:
                st.info(
                    "Waiting for enough independently approved, "
                    "eligible feedback."
                )

            checked_at = retraining["checked_at"]
            st.caption(f"Last feedback check (UTC): {checked_at}")

            try:
                checked_time = datetime.fromisoformat(
                    checked_at.replace("Z", "+00:00")
                )
                age = (
                    datetime.now(timezone.utc) - checked_time
                ).total_seconds()

                if age > 180:
                    st.warning(
                        "Retraining status is over three minutes old. "
                        "The background worker may need attention."
                    )
            except ValueError:
                st.warning(
                    "Retraining status timestamp could not be parsed."
                )

        st.caption(
            "Monitoring counts do not confirm that a new model "
            "has been trained or deployed. "
            "Automatic deployment is disabled by default."
        )

    st.markdown("#### Deployment")
    if api_online:
        st.success("Prediction API connected")
    else:
        st.error("Prediction API unavailable")
