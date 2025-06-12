import os
from supabase import create_client, Client


def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise EnvironmentError("ERROR: Missing SUPABASE_URL or SUPABASE_KEY")

    return create_client(url, key)