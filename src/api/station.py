from supabase_client import get_supabase_client

supabase = get_supabase_client()

def fetch_stations():
    response = supabase.from_('station').select('*').execute()
    return response.data


def fetch_station_by_name(station_name: str):
    response = supabase.from_('station').select('*').eq('name', station_name).single().execute()
    return response.data