import os
import sys # Import sys to access command line arguments
import pandas as pd
from supabase import create_client, Client

# --- Environment Variable Setup ---
# These variables will be provided by GitHub Actions secrets
# and injected into the environment where this script runs.
# We're using .get() with None as default to handle cases where they might not be set
# (though in GHA, they should be).
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Use service_role key for backend operations
SUPABASE_TABLE_NAME = os.environ.get("SUPABASE_TABLE_NAME", "weather") # Default to 'weather'

# --- Supabase Client Initialization ---
# Basic validation for essential Supabase credentials
if not SUPABASE_URL or not SUPABASE_KEY:
    # Print to stderr and exit if essential variables are missing
    print("Error: Supabase URL or Key not found in environment variables.", file=sys.stderr)
    print("Please ensure SUPABASE_URL and SUPABASE_KEY are set as GitHub Secrets.", file=sys.stderr)
    sys.exit(1) # Exit with an error code

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}", file=sys.stderr)
    sys.exit(1) # Exit if client cannot be created

def save_data_to_supabase(file_path: str):
    """
    Reads data directly from a CSV file and inserts/upserts it into the
    Supabase 'weather' table, skipping rows that are duplicates based on
    the 'Date' column. No further preprocessing is applied within this function.

    Args:
        file_path (str): The path to the CSV file to read.
    """
    if not os.path.exists(file_path):
        print(f"Error: CSV file not found at: {file_path}", file=sys.stderr)
        sys.exit(1) # Exit if the CSV file is not found

    try:
        # 1. Read data directly from CSV into a DataFrame
        print(f"Reading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Initial DataFrame head:\n{df.tail(10)}")

        # 2. Convert DataFrame to a list of dictionaries for Supabase insertion.
        #    IMPORTANT: Ensure your CSV column names exactly match your Supabase table column names
        #    (including case sensitivity).
        data_to_insert = df.to_dict(orient='records')

        if not data_to_insert:
            print("No valid data rows to insert from CSV after conversion.")
            return

        # 3. Save data to Supabase using upsert with duplicate skipping
        print(f"Attempting to insert {len(data_to_insert)} records into Supabase table '{SUPABASE_TABLE_NAME}'...")

        # IMPORTANT: 'on_conflict' must match the column(s) with your UNIQUE constraint
        # in the Supabase 'weather' table. This is crucial for duplicate skipping.
        # Assuming 'Date' is the unique key. Adjust if your column name is 'date' (lowercase)
        # or if you have a composite key (e.g., "Date,Location").
        on_conflict_cols = "Date,Location" # <--- VERIFY THIS MATCHES YOUR SUPABASE UNIQUE CONSTRAINT COLUMN NAME

        response = supabase.table(SUPABASE_TABLE_NAME).upsert(
            data_to_insert,
            on_conflict=on_conflict_cols,
            ignore_duplicates=True
        ).execute()

        if response.data:
            print(f"Successfully inserted/skipped duplicates. Inserted {len(response.data)} new records.")
        else:
            print("No new records were inserted (all existing or empty input from CSV).")

    except Exception as e:
        print(f"Error during data processing or saving to Supabase: {e}", file=sys.stderr)
        sys.exit(1) # Exit with an error code if an unhandled exception occurs

# --- Main execution block ---
if __name__ == "__main__":
    # The script expects the CSV file path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python data_ingestion_script.py <path_to_csv_file>", file=sys.stderr)
        sys.exit(1) # Exit if no file path is provided

    csv_file_path = sys.argv[1] # Get the file path from the first command-line argument
    print(f"Starting data ingestion process from CSV: {csv_file_path}")
    save_data_to_supabase(csv_file_path)
    print("Data ingestion script finished.")