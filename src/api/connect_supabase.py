import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

def fetch_weather_data_for_station(station: str, limit: int = 240):
    print("station: ", station)

    response = (
        supabase
        .from_("weather")
        .select("*")
        .eq("Location", station)
        .order("Date", desc=True)
        .limit(limit)
        .execute()
    )

    """ if response.error:
        raise Exception(response.error.message) """
    
    """ for row in response.data:
        print(f"{row['Location']} {row['Date']}") """
    
    # print("response data: ", response.data)
    return response.data[::-1]  # oldest first


# fetch_weather_data_for_station("Lendava", 240)