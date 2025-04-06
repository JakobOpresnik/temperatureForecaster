import os
import sys
import numpy as np
import pandas as pd
from lxml import etree as ET

def preprocess_temp_data(station_id):
    # open XML file
    with open(f"data/raw/temp/temp_data_{station_id}.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    # extract and print data
    print(f"Source: {root.find('credit').text}")
    print(f"Suggested Capture: {root.find('suggested_pickup').text}")
    print(f"Suggested Capture Period: {root.find('suggested_pickup_period').text}")

    # extract list of temperature records for this station
    records = tree.xpath('//metData')

    # reverse list
    records = records[::-1]
    
    # initialize an empty DataFrame
    columns = [
        "Location", 
        "Date", 
        "Temperature", 
        "Temperature dew point", 
        "Temperature average in time interval", 
        "Temperature maximum in time interval", 
        "Temperature minimum in time interval", 
        "Humidity relative", 
        "Humidity relative average in time interval", 
        "Wind direction", 
        "Wind direction average in time interval", 
        "Wind direction maximum gust in time interval", 
        "Wind speed", 
        "Wind speed average in time interval", 
        "Wind speed maximum in time interval", 
        "Air pressure", 
        "Air pressure average in time interval", 
        "Precipitation total in time interval", 
        "Solar radiation", 
        "Solar radiation average in time interval", 
    ]

    df = pd.DataFrame(columns=columns)

    # check if csv file already exists
    if os.path.exists(f"data/preprocessed/temp/{station_id}.csv"):
        df = pd.read_csv(f"data/preprocessed/temp/{station_id}.csv")

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

    # sort the DataFrame by the "date_to" column
    # df = df.sort_values(by="Date")

    # filter unique "Date" values
    df = df.drop_duplicates(subset=["Date"])

    station_name = records[0].find("domain_title").text.strip().upper()

    # filter out data which is more frequent than 30 minutes
    if station_name == "PTUJ":
        print("Filtering PTUJ data for 30 minutes intervals...")

        # convert 'Date' to datetime object for filtering
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')

        # filter the data to only keep records that are at the 30-minute mark or the hour mark
        df = df[df['Date'].dt.minute.isin([0, 30])]

        # convert 'Date' back to the format 'YYYY-MM-DD HH:MM:SS'
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # start_date = records[0].find("tsValid_issued").text.split(" ")[0].replace(".", "-")
    # end_date = records[len(records)-1].find("tsValid_issued").text.split(" ")[0].replace(".", "-")

    print(df)
    # print(f"Fetching successful. Fetched data from {start_date} to {end_date}")
    print("Fetching successful.")
    print(f"Saving pre-processed data to: data/preprocessed/temp/{station_id}.csv")

    # save the DataFrame to a CSV file
    # df.to_csv(f"data/preprocessed/temp/{station_name}_{start_date}_to_{end_date}.csv", index=False)
    df.to_csv(f"data/preprocessed/temp/{station_id}.csv", index=False)

if __name__ == "__main__":
    station_id = sys.argv[1]
    preprocess_temp_data(station_id=station_id)