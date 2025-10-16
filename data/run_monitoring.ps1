# Navigate to project folder
cd "C:\Users\gagan\OneDrive\Desktop\automated-mlops-monitoring"

# Ensure Docker containers are running
docker compose up -d

# Run the monitoring job inside the API container
docker compose exec api python monitoring_job.py
