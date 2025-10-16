import os
import time
import pandas as pd
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import exc, text
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping
import requests

# Import database models and utilities from the API service
from db_models import PredictionLog, MonitoringMetric, SessionLocal, engine, Base, get_db

# --- Configuration ---
# Reference data path can be set via environment variable; default to /data/reference_data.csv
REFERENCE_DATA_PATH = os.environ.get("REFERENCE_DATA_PATH", "/data/reference_data.csv")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")
BATCH_HOURS = int(os.environ.get("BATCH_HOURS", 24))  # Look back 24 hours by default
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", 0))

def fetch_data_from_db(db: Session, lookback_hours: int = 24) -> pd.DataFrame:
    """Fetches recent production data from the prediction_logs table."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)

    print(f"Fetching production data from {start_time} to {end_time}")

    try:
        query = text("""
            SELECT feature_1, feature_2, prediction, target, prediction_time
            FROM prediction_logs
            WHERE prediction_time BETWEEN :start_time AND :end_time
        """)
        df = pd.read_sql(query, engine, params={'start_time': start_time, 'end_time': end_time})
        print(f"Fetched {len(df)} records for monitoring.")
        return df, start_time, end_time
    except exc.OperationalError as e:
        print(f"Database Operational Error: {e}")
        return pd.DataFrame(), start_time, end_time
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame(), start_time, end_time

def run_evidently_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Generates an Evidently Data Drift Report between reference and current data."""
    # Ensure both datasets share the same essential feature columns
    required_features = ['feature_1', 'feature_2']
    for column_name in required_features:
        if column_name not in reference_data.columns:
            reference_data[column_name] = pd.NA
        if column_name not in current_data.columns:
            current_data[column_name] = pd.NA

    # Exclude prediction column from drift analysis to avoid mixed/empty columns
    for df in (reference_data, current_data):
        if 'prediction' in df.columns:
            df.drop(columns=['prediction'], inplace=True)

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = ['feature_1', 'feature_2']
    column_mapping.prediction = None
    column_mapping.target = None

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    print("Generating Evidently Report...")
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    print("Evidently Report generated.")
    return report.as_dict()

def process_and_log_metrics(db: Session, report_dict: dict, start_time: datetime, end_time: datetime):
    """Extracts key metrics from the Evidently report and logs them to the DB."""
    drift_metric = report_dict['metrics'][0]['result']
    # Evidently returns dataset_drift as a boolean; convert to float for DB
    dataset_drift_flag = bool(drift_metric.get('dataset_drift', False))
    data_drift_score = 1.0 if dataset_drift_flag else 0.0
    num_drifted_features = int(drift_metric.get('number_of_drifted_features', 0))

    quality_metric_section = report_dict['metrics'][1]['result']
    rows_count = int(quality_metric_section.get('rows_count', 0))

    print(f"Drift Summary: Dataset Drift={data_drift_score}, Drifted Features={num_drifted_features}")

    drift_log = MonitoringMetric(
        timestamp=datetime.now(),
        data_drift_score=data_drift_score,
        num_drifted_features=num_drifted_features,
        metric_name="data_drift_summary",
        metric_value=float(data_drift_score),
        report_summary=json.dumps({"rows_checked": rows_count, "drifted_features": num_drifted_features}),
        model_version=MODEL_VERSION,
        batch_start_time=start_time,
        batch_end_time=end_time
    )

    count_log = MonitoringMetric(
        timestamp=datetime.now(),
        data_drift_score=0.0,
        num_drifted_features=0,
        metric_name="prediction_count",
        metric_value=float(rows_count),
        report_summary=None,
        model_version=MODEL_VERSION,
        batch_start_time=start_time,
        batch_end_time=end_time
    )

    try:
        db.add_all([drift_log, count_log])
        db.commit()
        print("Successfully logged 2 metrics to 'monitoring_metrics'.")
    except Exception as e:
        db.rollback()
        print(f"Failed to commit metrics: {e}")

    # After logging, check and potentially send alert
    check_for_drift_and_alert(
        drift_score=data_drift_score,
        num_drifted_features=num_drifted_features,
        start_time=start_time,
        end_time=end_time,
    )

def send_slack_alert(text: str) -> None:
    if not SLACK_WEBHOOK_URL:
        print("SLACK_WEBHOOK_URL not configured; skipping Slack alert.")
        return
    payload = {
        "username": "Vigil Drift Bot",
        "icon_emoji": ":warning:",
        "text": text,
    }
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if response.ok:
            print("Slack alert sent.")
        else:
            print(f"Slack alert failed: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error sending Slack alert: {e}")

def check_for_drift_and_alert(drift_score: float, num_drifted_features: int, start_time: datetime, end_time: datetime) -> None:
    if num_drifted_features > DRIFT_THRESHOLD:
        message = (
            f":rotating_light: Data Drift Detected!\n"
            f"Drifted Features: {num_drifted_features}\n"
            f"Drift Score: {drift_score:.4f}\n"
            f"Window: {start_time:%Y-%m-%d %H:%M:%S} -> {end_time:%Y-%m-%d %H:%M:%S}"
        )
        send_slack_alert(message)

def main_monitoring_job():
    """Main execution flow for the batch monitoring job."""
    print("--- Starting Evidently Batch Monitoring Job ---")

    # 1. Load Reference Data
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"Reference data not found at {REFERENCE_DATA_PATH}. Please run model_prep.py first.")
        return

    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    print(f"Reference data loaded: {len(reference_data)} rows.")

    # 2. Setup DB session and fetch current data
    db = SessionLocal()
    try:
        current_data, start_time, end_time = fetch_data_from_db(db, lookback_hours=BATCH_HOURS)
        if current_data.empty:
            print("No new data found in the batch window. Skipping report generation.")
            return

        # 3. Run Evidently Report
        report_data = run_evidently_report(reference_data, current_data)

        # 4. Process and Log Metrics
        process_and_log_metrics(db, report_data, start_time, end_time)
    finally:
        db.close()
        print("--- Monitoring Job Finished ---")

if __name__ == '__main__':
    print("Waiting 15 seconds to ensure DB is fully initialized...")
    time.sleep(15)
    main_monitoring_job()
