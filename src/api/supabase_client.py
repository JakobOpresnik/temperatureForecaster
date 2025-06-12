import os
from supabase import create_client, Client


def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    url="https://yscjmekilrlflvefhyog.supabase.co"
    key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlzY2ptZWtpbHJsZmx2ZWZoeW9nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MDM1OTAsImV4cCI6MjA2NTA3OTU5MH0.kXu0CcfbJbBm93qpArxBYzc9PcQlipPE4nwKIgL301w"

    if not url or not key:
        raise EnvironmentError("ERROR: Missing SUPABASE_URL or SUPABASE_KEY")

    return create_client(url, key)