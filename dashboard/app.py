import os
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text

# --- 1. Configuration and Database Setup (Replicated from API/Data Feeder) ---

# Get DB credentials from standardized DB_* environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password")
DB_NAME = os.getenv("DB_NAME", "vigil_db")
# 'db' is the service name for PostgreSQL in the Docker network
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")

# API host configuration
API_HOST = os.getenv("API_HOST", "http://api:8000")  # Accept full URL or host
if not API_HOST.startswith("http://") and not API_HOST.startswith("https://"):
    API_URL = f"http://{API_HOST}:8000"
else:
    API_URL = API_HOST

# PostgreSQL connection URL
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy setup
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- 2. Database Models (Needed to query the data) ---

class PredictionLog(Base):
    """Reflects the prediction_logs table."""
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    feature_1 = Column(Float)
    feature_2 = Column(Float)
    prediction = Column(Integer)
    target = Column(Integer, nullable=True) 
    prediction_time = Column(DateTime)
    model_version = Column(String(50))

class MonitoringMetric(Base):
    """Reflects the monitoring_metrics table."""
    __tablename__ = "monitoring_metrics"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    data_drift_score = Column(Float)
    num_drifted_features = Column(Integer)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    report_summary = Column(Text)
    model_version = Column(String(50))
    batch_start_time = Column(DateTime)
    batch_end_time = Column(DateTime)


# --- 3. Data Interaction Functions ---

@st.cache_data(ttl=5) # Cache data for 5 seconds for fast reloads
def fetch_prediction_history(limit=100):
    """Fetches the latest prediction logs."""
    try:
        df = pd.read_sql(
            text(f"SELECT * FROM prediction_logs ORDER BY prediction_time DESC LIMIT :limit_val"), 
            engine,
            params={'limit_val': limit}
        )
        return df
    except Exception as e:
        st.error(f"Could not fetch prediction history: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60) # Cache monitoring metrics for 60 seconds
def fetch_monitoring_metrics():
    """Fetches the latest monitoring metrics."""
    try:
        df = pd.read_sql(
            text("SELECT * FROM monitoring_metrics ORDER BY timestamp DESC"), 
            engine
        )
        return df
    except Exception as e:
        st.error(f"Could not fetch monitoring metrics: {e}")
        return pd.DataFrame()

def get_prediction(feature_1: float, feature_2: float):
    """Sends a request to the FastAPI /predict endpoint."""
    payload = {
        "feature_1": feature_1,
        "feature_2": feature_2
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not connect to {API_URL}. Is the 'api' service running? Details: {e}")
        return None

# --- 4. Streamlit UI Layout ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="MLOps Monitoring Dashboard")
    st.title("🛡️ MLOps Prediction & Monitoring Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    # --- Real-Time Prediction Section (col1) ---
    with col1:
        st.header("Real-Time Prediction")
        st.subheader("Input Features")
        
        with st.form("prediction_form"):
            # Set default values close to the reference data range
            f1 = st.slider("Feature 1 (Continuous)", min_value=0.0, max_value=20.0, value=7.5, step=0.1)
            f2 = st.slider("Feature 2 (Continuous)", min_value=0.0, max_value=40.0, value=15.0, step=0.1)
            submitted = st.form_submit_button("Get Prediction & Log")

            if submitted:
                with st.spinner('Requesting prediction...'):
                    result = get_prediction(f1, f2)
                    if result:
                        st.success(f"Prediction Complete (Model V{result['model_version']})")
                        if result['prediction'] == 1:
                            st.balloons()
                            st.metric(label="Predicted Class", value="Positive (1)")
                        else:
                            st.metric(label="Predicted Class", value="Negative (0)")
                        st.text(f"Raw Input: F1={f1}, F2={f2}")
                        st.cache_data.clear() # Clear cache to refresh history
    
    # --- Monitoring Dashboard Section (col2 - Top Right) ---
    with col2:
        st.header("Model Health and Data Drift")
        metrics_df = fetch_monitoring_metrics()
        
        if not metrics_df.empty:
            latest_drift = metrics_df[metrics_df['metric_name'] == 'data_drift_summary'].iloc[0]
            latest_count = metrics_df[metrics_df['metric_name'] == 'prediction_count'].iloc[0]
            
            # Display Key Metrics
            col2_a, col2_b, col2_c = st.columns(3)

            col2_a.metric(
                label="Latest Prediction Count (Batch)",
                value=f"{int(latest_count['metric_value']):,}",
                delta=f"Batch from {latest_count['batch_start_time'].strftime('%m/%d %H:%M')}"
            )
            
            col2_b.metric(
                label="Data Drift Score",
                value=f"{latest_drift['data_drift_score']:.4f}",
                delta=f"{latest_drift['num_drifted_features']} Features Drifted",
                delta_color="inverse" if latest_drift['num_drifted_features'] > 0 else "normal"
            )
            
            col2_c.metric(
                label="Last Monitor Run",
                value=latest_drift['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            )
            
            st.markdown("---")
            
            # Display Drift History Chart
            st.subheader("Data Drift Score History")
            drift_history = metrics_df[metrics_df['metric_name'] == 'data_drift_summary'].sort_values('timestamp', ascending=True)
            drift_history = drift_history.rename(columns={'timestamp': 'Run Time', 'data_drift_score': 'Drift Score'})
            st.line_chart(drift_history, x='Run Time', y='Drift Score')

            # Detailed metrics table
            with st.expander("View All Monitoring Metrics"):
                st.dataframe(metrics_df[['timestamp', 'metric_name', 'metric_value', 'data_drift_score', 'num_drifted_features', 'model_version']], use_container_width=True)
        else:
            st.info("No monitoring metrics available yet. Run the `monitoring_job.py` script once the API has logs.")
            st.text(f"API Health Check: {API_URL}/")
            try:
                if requests.get(f"{API_URL}/", timeout=2).status_code == 200:
                    st.success("API service is reachable.")
                else:
                    st.warning("API service reachable, but health check failed.")
            except:
                st.error("API service is unreachable.")


    # --- Prediction History Section (Below the main columns) ---
    st.markdown("---")
    st.header("Prediction Log History")
    
    history_df = fetch_prediction_history(limit=50)
    if not history_df.empty:
        # Format the time column for better display
        history_df['prediction_time'] = history_df['prediction_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(history_df.drop(columns=['id', 'target']), use_container_width=True)
    else:
        st.info("No prediction logs found in the database. Use the Real-Time Prediction form to generate some data.")


if __name__ == "__main__":
    main()
