import time
from sqlalchemy import exc
from db_models import engine, Base

MAX_RETRIES = 10
RETRY_DELAY = 5 # seconds

def create_database_tables():
    """
    Attempts to connect to the database and create all defined tables.
    Uses exponential backoff logic to wait for the 'db' service to be ready.
    """
    print("--- Starting Database Initialization ---")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempting to connect to the database... (Attempt {attempt + 1}/{MAX_RETRIES})")
            
            # This action attempts a connection immediately
            # Base.metadata.create_all attempts to connect and create tables if they don't exist
            Base.metadata.create_all(bind=engine)
            
            print("\n Database connection successful and tables created or already exist.")
            print("Tables created: prediction_logs, monitoring_metrics")
            return
            
        except exc.OperationalError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Database not ready. Waiting {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Failed to connect to the database after multiple retries.")
                print(f"Error details: {e}")
                # Re-raise the exception on the last attempt
                raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

if __name__ == '__main__':
    create_database_tables()
