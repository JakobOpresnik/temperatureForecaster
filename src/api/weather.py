from supabase_client import get_supabase_client

supabase = get_supabase_client()

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