import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define file paths relative to the container's working directory (/app)
# The 'data' folder is mounted from the host's root './data' to the container's '/app/data'
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'model.joblib')
REFERENCE_DATA_PATH = os.path.join(DATA_DIR, 'reference_data.csv')

# Ensure the data directory exists (this now reliably creates the mapped folder)
os.makedirs(DATA_DIR, exist_ok=True)

def generate_dummy_data(n_samples=5000):
    """Generates a dummy dataset for a binary classification problem."""
    # Create two features with some correlation
    np.random.seed(42)
    feature_1 = np.random.rand(n_samples) * 10
    feature_2 = 2 * feature_1 + np.random.randn(n_samples) * 5
    
    # Create a target variable based on a simple rule with noise
    target = (feature_1 * 0.5 + feature_2 * 0.1 + feature_2 * 0.1 + np.random.randn(n_samples) * 2 > 4.5).astype(int)
    
    data = pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2,
        'target': target
    })
    return data

def train_and_save_model(data):
    """Trains a simple Logistic Regression model and saves it."""
    print("Training dummy model...")
    
    # Use all data for the 'training' set to ensure model is functional
    X = data[['feature_1', 'feature_2']]
    y = data['target']
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Dummy model saved to: {MODEL_PATH}")

def save_reference_data(data):
    """Saves a portion of the data to serve as the reference set for drift detection."""
    # Use the first 1000 samples as reference data
    reference_data = data.head(1000)
    
    # Ensure 'target' column is present for performance reporting later
    reference_data.to_csv(REFERENCE_DATA_PATH, index=False)
    print(f"Reference data saved to: {REFERENCE_DATA_PATH}")

if __name__ == '__main__':
    full_data = generate_dummy_data()
    
    # 1. Train and save the model for the API service to load
    train_and_save_model(full_data)
    
    # 2. Save the reference data for the monitoring service
    save_reference_data(full_data)
    
    print("\nModel preparation complete. The 'data' folder now contains 'model.joblib' and 'reference_data.csv'.")
