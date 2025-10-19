# 🛡️ Vigil: An End-to-End MLOps Monitoring Dashboard

### Overview
Vigil is a robust, production-ready system designed to combat silent model decay by providing **automated, proactive monitoring** for deployed machine learning models. It combines industry-standard tools for model serving, persistent logging, sophisticated drift analysis, and real-time visualization with automated alerting.

---

## Project Overview

### Problem
Machine learning models can degrade silently over time due to:

- **Data Drift:** Changes in the input feature distribution.  
- **Concept Drift:** Changes in the relationship between features and target variables.  

These drifts can lead to **inaccurate predictions** and **business losses** if left undetected.

### Solution
Vigil implements a **closed-loop monitoring system** with four key steps:

1. **Capture:** Log every prediction request with input features and outputs.  
2. **Analyze:** Compute statistical metrics and detect drift using Evidently AI.  
3. **Alert:** Trigger automated Slack notifications when drift or anomalies occur.  
4. **Visualize:** Provide real-time monitoring via an interactive Streamlit dashboard.

### Target Audience
- **Data Scientists** – Ensure deployed models are performing as expected.  
- **MLOps Engineers** – Monitor models, detect drift, and automate alerting.  
- **Product Managers** – Gain insights into model health and system reliability.
  ## Quick Start

> This will get Vigil running on your local machine within minutes.

# Clone the repository
git clone https://github.com/gaganakumarm/Vigil-MLOps-Monitoring.git
cd Vigil-MLOps-Monitoring

# Build Docker images for all services
docker compose build

# Initialize database schema and train reference model
docker compose run --rm data-feeder python create_db.py
docker compose run --rm data-feeder python model_prep.py

# Start full stack (API, DB, Dashboard)
docker compose up -d

# Simulate production traffic
docker compose run --rm data-feeder python feeder.py

# Run monitoring job (calculates drift & updates dashboard)
docker compose exec api python monitoring_job.py

Access the Dashboard: http://localhost:8501
