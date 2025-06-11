import os
from supabase import create_client, Client

# handled by railway service variables
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)


def fetch_stations():
    response = supabase.from_('station').select('*').execute()
    return response.data


def fetch_station_by_name(station_name: str):
    response = supabase.from_('station').select('*').eq('name', station_name).single().execute()
    return response.data