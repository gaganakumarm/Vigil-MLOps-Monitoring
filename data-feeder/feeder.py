import requests
import time
import random
import os

# --- Configuration ---
# 'api' is the service name of the FastAPI service in the Docker network
API_URL = os.getenv("API_HOST", "http://api:8000") 
PREDICT_ENDPOINT = f"{API_URL}/predict"

# Simulation parameters
NUM_RECORDS_TO_SEND = 500
BATCH_SIZE = 10
DELAY_PER_BATCH = 1 # seconds

def generate_production_data(num_records):
    """
    Generates mock production data.
    """
    mock_data = []
    for _ in range(num_records):
        # Generate features slightly different from the training data for realism
        feature_1 = round(random.uniform(5.0, 15.0), 3)
        feature_2 = round(random.uniform(10.0, 30.0), 3)
        mock_data.append({
            "feature_1": feature_1,
            "feature_2": feature_2
        })
    return mock_data

def send_batch_to_api(batch):
    """Sends a batch of records to the FastAPI /predict endpoint."""
    success_count = 0
    failure_count = 0
    
    for record in batch:
        try:
            # We don't need exponential backoff here since the run will be manually triggered
            # after all services are up, but we use a reasonable timeout.
            response = requests.post(PREDICT_ENDPOINT, json=record, timeout=5)
            
            if response.status_code == 200:
                success_count += 1
            else:
                failure_count += 1
                print(f"  [ERROR] API responded with status {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            failure_count += 1
            print(f"  [ERROR] Connection failed. Is the API service running at {API_URL}?")
        except requests.exceptions.Timeout:
            failure_count += 1
            print(f"  [ERROR] Request timed out.")
        except Exception as e:
            failure_count += 1
            print(f"  [ERROR] An unexpected error occurred: {e}")

    return success_count, failure_count


def simulate_production_traffic():
    """Simulates sending a large volume of data in batches."""
    print(f"--- Starting Production Data Simulation ---")
    print(f"Target API Endpoint: {PREDICT_ENDPOINT}")
    print(f"Total Records to Send: {NUM_RECORDS_TO_SEND}")
    
    all_data = generate_production_data(NUM_RECORDS_TO_SEND)
    total_successful = 0
    total_failed = 0
    
    for i in range(0, NUM_RECORDS_TO_SEND, BATCH_SIZE):
        batch = all_data[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"\nSending Batch {batch_num} ({len(batch)} records)...")
        
        success, failure = send_batch_to_api(batch)
        total_successful += success
        total_failed += failure
        
        if i + BATCH_SIZE < NUM_RECORDS_TO_SEND:
            print(f"Waiting {DELAY_PER_BATCH} second(s) before next batch...")
            time.sleep(DELAY_PER_BATCH)

    print("\n--- Simulation Complete ---")
    print(f"Total Successful Logs: {total_successful}")
    print(f"Total Failed Logs: {total_failed}")
    
    if total_successful > 0:
        print("Data logging successful. Check the PostgreSQL 'prediction_logs' table.")
    else:
        print("Data logging failed. Check service logs and network configuration.")


if __name__ == '__main__':
    # Wait for the API to be ready after all services start
    print("Waiting 10 seconds for API service to stabilize before feeding data...")
    time.sleep(10) 
    
    simulate_production_traffic()
