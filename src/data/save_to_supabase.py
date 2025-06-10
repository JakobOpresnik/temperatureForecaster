import os
import sys
import pandas as pd
import numpy as np
import yaml
from supabase import create_client, Client



SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE_NAME = os.environ.get("SUPABASE_TABLE_NAME", "weather") # default to 'weather'

SUPABASE_URL="https://yscjmekilrlflvefhyog.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlzY2ptZWtpbHJsZmx2ZWZoeW9nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MDM1OTAsImV4cCI6MjA2NTA3OTU5MH0.kXu0CcfbJbBm93qpArxBYzc9PcQlipPE4nwKIgL301w"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Supabase URL or Key not found in environment variables.", file=sys.stderr)
    print("Please ensure SUPABASE_URL and SUPABASE_KEY are set as GitHub Secrets.", file=sys.stderr)
    sys.exit(1)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}", file=sys.stderr)
    sys.exit(1) # exit if client cannot be created


def save_data_to_supabase():
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    stations = yaml.safe_load(open("params.yaml"))["stations"]

    output_file_path_template = params["output_file_path_template"]

    for station in stations:
        file_path = output_file_path_template.format(station=station)

        if not os.path.exists(file_path):
            print(f"Error: CSV file not found at: {file_path}", file=sys.stderr)
            sys.exit(1) # exit if csv file doesn't exist
        try:
            print(f"Reading data from {file_path}...")
            df = pd.read_csv(file_path)
            print(f"Initial DataFrame head:\n{df.tail(10)}")

            # replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # print("Replaced infinite values with NaN.")

            # replace NaN values in numeric columns with None (for SQL NULL)
            for col in df.select_dtypes(include=['number']).columns:
                df[col] = df[col].replace(np.nan, None)
            # print("Replaced NaN values in numeric columns with None (for SQL NULL).")

            # convert df to dict & ensure csv column names match exactly with Supabase table column names
            data_to_insert = df.to_dict(orient='records')

            if not data_to_insert:
                print("No valid data rows to insert from CSV after conversion.")
                return

            # must match Supabase unique constraint
            on_conflict_cols = "Date,Location"

            # save data to Supabase using upsert with duplicate skipping
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
            sys.exit(1)


if __name__ == "__main__":
    save_data_to_supabase()