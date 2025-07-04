import os
import requests
import numpy as np
import pandas as pd
import yaml
from lxml import etree as ET


def preprocess_temperature_data():
    params = yaml.safe_load(open("params.yaml"))
    preprocess = params["preprocess"]
    stations = params["stations"]

    xml_data_tag = preprocess["xml_data_tag"]
    input_file_path_template = preprocess["input_file_path_template"]
    output_file_path_template = preprocess["output_file_path_template"]
    filter_half_hour_stations = preprocess["filter_half_hour_stations"]
    columns = preprocess["data_columns"]

    for station in stations:
        input_file_path = input_file_path_template.format(station=station)
        output_file_path = output_file_path_template.format(station=station)

        try:
            # open XML file
            with open(input_file_path, "rb") as file:
                tree = ET.parse(file)
                root = tree.getroot()

            # extract and print data
            print(f"Source: {root.find('credit').text}")
            print(f"Suggested Capture: {root.find('suggested_pickup').text}")
            print(f"Suggested Capture Period: {root.find('suggested_pickup_period').text}")

            # extract list of temperature records for this station
            records = tree.xpath(f'//{xml_data_tag}')

            # reverse records
            records = records[::-1]
            
            # initialize an empty DataFrame
            df = pd.DataFrame(columns=columns)

            # check if csv file already exists
            if os.path.exists(output_file_path):
                df = pd.read_csv(output_file_path, parse_dates=["Date"])

            # convert the XML data to a DataFrame
            for record in records:
                location = record.find("domain_longTitle").text
                date = record.find('tsValid_issued').text.replace(" CEST", "")  # remove CEST part of the datetime string
                temperature = record.find('t').text if record.find('t') is not None else np.nan
                temperature_dew_point = record.find('td').text if record.find('td') is not None else np.nan
                temperature_avg = record.find('tavg').text if record.find('tavg') is not None else np.nan
                temperature_max = record.find("tx").text if record.find('tx') is not None else np.nan
                temperature_min = record.find("tn").text if record.find('tn') is not None else np.nan
                humidity_relative = record.find('rh').text if record.find('rh') is not None else np.nan
                humidity_relative_avg = record.find('rhavg').text if record.find('rhavg') is not None else np.nan
                wind_direction = record.find('dd_val').text if record.find('dd_val') is not None else np.nan
                wind_direction_avg = record.find('ddavg_val').text if record.find('ddavg_val') is not None else np.nan
                wind_direction_max = record.find('ddmax_val').text if record.find('ddmax_val') is not None else np.nan
                wind_speed = record.find('ff_val_kmh').text if record.find('ff_val_kmh') is not None else np.nan
                wind_speed_avg = record.find('ffavg_val_kmh').text if record.find('ffavg_val_kmh') is not None else np.nan
                wind_speed_max = record.find('ffmax_val_kmh').text if record.find('ffmax_val_kmh') is not None else np.nan
                air_pressure = record.find('p').text if record.find('p') is not None else np.nan
                air_pressure_avg = record.find('pavg').text if record.find('pavg') is not None else np.nan
                precipitation_total = record.find('rr_val').text if record.find('rr_val') is not None else np.nan
                solar_radiation_total = record.find('gSunRad').text if record.find('gSunRad') is not None else np.nan
                solar_radiation_avg = record.find('gSunRadavg').text if record.find('gSunRadavg') is not None else np.nan

                # append the data as a new row in the DataFrame
                df = pd.concat([df, pd.DataFrame([[
                    location, 
                    date, 
                    temperature, 
                    temperature_dew_point,
                    temperature_avg,
                    temperature_max,
                    temperature_min,
                    humidity_relative,
                    humidity_relative_avg,
                    wind_direction,
                    wind_direction_avg,
                    wind_direction_max,
                    wind_speed,
                    wind_speed_avg,
                    wind_speed_max,
                    air_pressure,
                    air_pressure_avg,
                    precipitation_total,
                    solar_radiation_total,
                    solar_radiation_avg
                ]], columns=columns)], ignore_index=True)

            if df['Date'].dtype == object:
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')

            station_name = records[0].find("domain_title").text.strip().upper()

            # filter out data which is more frequent than 30 minutes
            if station_name in filter_half_hour_stations:
                print(f"Filtering {station_name} data for 30 minutes intervals...")
                # filter the data to only keep records that are at the 30-minute mark or the hour mark
                df = df[df['Date'].dt.minute.isin([0, 30])]

            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # filter unique "Date" values
            df = df.drop_duplicates(subset=["Date"])

            print(df)
            print("Preprocessing successful.")
            print(f"Saving pre-processed data to: {output_file_path}")

            # save the DataFrame to a CSV file
            df.to_csv(output_file_path, index=False)
        
        except requests.RequestException as e:
            # print error message if there is a problem preprocessing the file
            print(f"Error preprocessing data: {e}")


if __name__ == "__main__":
    preprocess_temperature_data()