import os
import pandas as pd
from supabase import create_client, Client

# handled by railway service variables
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)


def save_data_to_supabase(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: CSV file not found at: {file_path}")
        return # Exit the function if file doesn't exist
    
    try:
        # 1. Read data directly from CSV into a DataFrame
        print(f"Reading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Initial DataFrame head:\n{df.head()}")

        # 2. Convert DataFrame to a list of dictionaries for Supabase insertion.
        #    Ensure your CSV column names exactly match your Supabase table column names.
        data_to_insert = df.to_dict(orient='records')

        if not data_to_insert:
            print("No valid data rows to insert from CSV.")
            return

        # 3. Save data to Supabase using upsert with duplicate skipping
        print(f"Attempting to insert {len(data_to_insert)} records into Supabase table 'weather'...")

        # IMPORTANT: 'on_conflict' must match the column(s) with your UNIQUE constraint
        # in the Supabase 'weather' table. Assuming 'Date' is the unique key.
        on_conflict_cols = "Date" # Adjust if your unique constraint is 'date' or 'Date,location' etc.

        response = supabase.table("weather").upsert(
            data_to_insert,
            on_conflict=on_conflict_cols,
            ignore_duplicates=True
        ).execute()

        if response.data:
            print(f"Successfully inserted/skipped duplicates. Inserted {len(response.data)} new records.")
        else:
            print("No new records were inserted (all existing or empty input).")

    except Exception as e:
        print(f"Error during data processing or saving to Supabase: {e}")
        raise # Re-raise the exception to indicate failure if used in a cron job



""" if not data:
        print("No new data to insert...")
        return
    
    try:
        response = supabase.table("weather").upsert(
            data,
            on_conflict="Date", # Or "Date,location" if your unique constraint is on both
            ignore_duplicates=True # This tells Supabase to skip rows that conflict
        ).execute()

        # The response.data will contain the rows that were actually inserted/updated.
        # If ignore_duplicates=True, it will only show newly inserted rows.
        if response.data:
            print(f"Successfully inserted/skipped duplicates. Inserted {len(response.data)} new records.")
        else:
            print("No new records inserted (all were duplicates or empty input).")

    except Exception as e:
        print(f"Error saving data to Supabase: {e}") """


def fetch_weather_data_for_station(station: str, limit: int = 240):
    response = (
        supabase
        .from_("weather")
        .select("*")
        .eq("Location", station)
        .order("Date", desc=True)
        .limit(limit)
        .execute()
    )

    # print("response data: ", response.data)

    return response.data[::-1]  # oldest rows first