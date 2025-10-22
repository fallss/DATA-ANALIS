from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

class SupabaseConnector:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")  # https://yjmijtkdihindeugpwif.supabase.co
        key = os.getenv("SUPABASE_KEY")  # Anon key dari dashboard
        self.client = create_client(url, key)
    
    def execute_query(self, table, query_params=None):
        return self.client.table(table).select("*").execute()