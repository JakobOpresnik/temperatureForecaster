import os
import requests
import numpy as np
import pandas as pd
import yaml
from lxml import etree as ET
from supabase import create_client, Client


def process_and_upload_temperature_data():
    # Load parameters from params.yaml
    params = yaml.safe_load(open("params.yaml"))

    # Supabase credentials (assuming these are in your params.yaml or environment variables)
    supabase_url = os.environ.get("SUPABASE_URL") #or params["supabase"]["url"]
    supabase_key = os.environ.get("SUPABASE_KEY") #or params["supabase"]["key"]
    # supabase_table_name = params["supabase"]["table_name"]

    print("SUPABASE-URL: ", supabase_url)

    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)

    # Fetch and Preprocess parameters
    fetch_params = params["fetch"]
    preprocess_params = params["preprocess"]
    stations = params["stations"]

    base_url = fetch_params["base_url"]
    station_url_suffix = fetch_params["station_url_suffix"]
    xml_data_tag = preprocess_params["xml_data_tag"]
    filter_half_hour_stations = preprocess_params["filter_half_hour_stations"]
    columns = preprocess_params["data_columns"]

    for station in stations:
        filename = station_url_suffix.format(station=station)
        url = base_url + filename

        try:
            # 1. Fetch the XML data directly (no file saving)
            print(f"Fetching data for station: {station} from {url}")
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the XML data from the response content
            root = ET.fromstring(response.content)

            # Extract and print source information
            print(f"Source: {root.find('credit').text}")
            print(f"Suggested Capture: {root.find('suggested_pickup').text}")
            print(f"Suggested Capture Period: {root.find('suggested_pickup_period').text}")

            # Extract list of temperature records for this station
            records = root.xpath(f'//{xml_data_tag}')

            # Reverse records (as per your original script)
            records = records[::-1]

            # Initialize an empty list to store processed rows
            processed_rows = []

            # Convert the XML data to a list of dictionaries
            for record in records:
                location = record.find("domain_longTitle").text
                date = record.find('tsValid_issued').text.replace(" CEST", "")
                
                # Extract all data points, handling potential missing tags with np.nan
                row_data = {
                    "location": location,
                    "Date": date,
                    "temperature": record.find('t').text if record.find('t') is not None else np.nan,
                    "temperature_dew_point": record.find('td').text if record.find('td') is not None else np.nan,
                    "temperature_avg": record.find('tavg').text if record.find('tavg') is not None else np.nan,
                    "temperature_max": record.find("tx").text if record.find('tx') is not None else np.nan,
                    "temperature_min": record.find("tn").text if record.find('tn') is not None else np.nan,
                    "humidity_relative": record.find('rh').text if record.find('rh') is not None else np.nan,
                    "humidity_relative_avg": record.find('rhavg').text if record.find('rhavg') is not None else np.nan,
                    "wind_direction": record.find('dd_val').text if record.find('dd_val') is not None else np.nan,
                    "wind_direction_avg": record.find('ddavg_val').text if record.find('ddavg_val') is not None else np.nan,
                    "wind_direction_max": record.find('ddmax_val').text if record.find('ddmax_val') is not None else np.nan,
                    "wind_speed": record.find('ff_val_kmh').text if record.find('ff_val_kmh') is not None else np.nan,
                    "wind_speed_avg": record.find('ffavg_val_kmh').text if record.find('ffavg_val_kmh') is not None else np.nan,
                    "wind_speed_max": record.find('ffmax_val_kmh').text if record.find('ffmax_val_kmh') is not None else np.nan,
                    "air_pressure": record.find('p').text if record.find('p') is not None else np.nan,
                    "air_pressure_avg": record.find('pavg').text if record.find('pavg') is not None else np.nan,
                    "precipitation_total": record.find('rr_val').text if record.find('rr_val') is not None else np.nan,
                    "solar_radiation_total": record.find('gSunRad').text if record.find('gSunRad') is not None else np.nan,
                    "solar_radiation_avg": record.find('gSunRadavg').text if record.find('gSunRadavg') is not None else np.nan
                }
                processed_rows.append(row_data)

            # Create DataFrame from processed rows
            df = pd.DataFrame(processed_rows, columns=columns)

            # Convert 'Date' column to datetime objects
            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')

            station_name = records[0].find("domain_title").text.strip().upper()

            # Filter out data which is more frequent than 30 minutes
            if station_name in filter_half_hour_stations:
                print(f"Filtering {station_name} data for 30 minutes intervals...")
                df = df[df['Date'].dt.minute.isin([0, 30])]

            # Format 'Date' back to string for Supabase (Supabase handles ISO 8601 timestamps well)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Filter unique "Date" values
            df = df.drop_duplicates(subset=["Date"])

            print(df)
            print("Preprocessing successful.")

            # Convert DataFrame to a list of dictionaries for Supabase insertion
            data_to_insert = df.to_dict(orient='records')

            # 3. Save preprocessed data to Supabase
            print(f"Uploading pre-processed data for {station_name} to Supabase table: weather")
            response = supabase.table("weather").insert(data_to_insert).execute()

            if response.data:
                print(f"Successfully uploaded {len(response.data)} records to Supabase for {station_name}.")
            else:
                print(f"No data uploaded to Supabase for {station_name}. Response: {response.data}")


        except requests.RequestException as e:
            print(f"Error fetching data for station {station}: {e}")
        except ET.XMLSyntaxError as e:
            print(f"Error parsing XML for station {station}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for station {station}: {e}")


if __name__ == "__main__":
    process_and_upload_temperature_data()