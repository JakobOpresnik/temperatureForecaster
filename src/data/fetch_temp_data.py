import os
import requests
import yaml
from datetime import datetime


def fetch_temperature_data():
    params = yaml.safe_load(open("params.yaml"))["fetch"]
    stations = yaml.safe_load(open("params.yaml"))["stations"]

    # URL to fetch the XML data
    base_url = params["base_url"]
    station_url_suffix = params["station_url_suffix"]
    output_file_path_template = params["output_file_path_template"]

    for station in stations:
        filename = station_url_suffix.format(station=station)
        url = base_url + filename
        output_file_path = output_file_path_template.format(station=station)

        # ensure the directory exists before saving the file
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        try:
            # fetch the XML data
            response = requests.get(url)
            response.raise_for_status()  # raise an exception for HTTP errors

            # save the XML data to a file
            with open(output_file_path, "wb") as file:
                file.write(response.content)

            # print success message with file location and datetime
            print(f"Fetching successful. Data saved to {output_file_path} at {datetime.now()}")

        except requests.RequestException as e:
            # print error message if there is a problem fetching the file
            print(f"Error fetching data: {e}")


if __name__ == "__main__":
    fetch_temperature_data()