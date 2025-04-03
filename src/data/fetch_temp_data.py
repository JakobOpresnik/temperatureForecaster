import sys
import requests
from datetime import datetime
import xml.etree.ElementTree as ET

def fetch_air_data(station_id):
    try:
        # URL to fetch the XML data
        url = f"https://meteo.arso.gov.si/uploads/probase/www/observ/surface/text/sl/recent/observationAms_{station_id}_history.xml"

        # fetch the XML data
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP errors

        # save the XML data to a file
        file_path = f"data/raw/temp/temp_data_{station_id}.xml"
        with open(file_path, "wb") as file:
            file.write(response.content)

        # print success message with file location and datetime
        print(f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        # print error message if there is a problem fetching the file
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    station_id = sys.argv[1]
    fetch_air_data(station_id=station_id)