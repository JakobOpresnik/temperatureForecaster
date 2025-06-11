import os
from supabase import create_client, Client

# handled by railway service variables
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)


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
    
    # oldest rows first
    return response.data[::-1]