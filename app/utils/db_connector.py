import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnector:
    def __init__(self):
        # Get database URL from environment variable
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Create engine with connection pooling settings
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )
    
    def execute_query(self, query, params=None):
        """Execute a SELECT query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                # Fetch all results and column names
                rows = result.fetchall()
                columns = result.keys()
                return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            print(f"Error executing query: {e}")
            raise
    
    def insert_job_vacancy(self, role_name, job_level, role_purpose, selected_talent_ids):
        """Insert a new job vacancy and return the generated ID"""
        query = """
        INSERT INTO talent_benchmarks (role_name, job_level, role_purpose, selected_talent_ids)
        VALUES (:role_name, :job_level, :role_purpose, :selected_talent_ids)
        RETURNING job_vacancy_id
        """
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(query), {
                    'role_name': role_name,
                    'job_level': job_level,
                    'role_purpose': role_purpose,
                    'selected_talent_ids': selected_talent_ids
                })
                return result.fetchone()[0]
        except Exception as e:
            print(f"Error inserting job vacancy: {e}")
            raise
    
    def get_match_results(self, job_vacancy_id):
        """Get matching results for a job vacancy"""
        try:
            # Check if SQL file exists
            sql_file_path = 'sql/03_final_match_calculation.sql'
            if os.path.exists(sql_file_path):
                with open(sql_file_path, 'r') as f:
                    query = f.read()
            else:
                # Fallback query if file doesn't exist
                query = """
                SELECT * FROM match_results 
                WHERE job_vacancy_id = :job_vacancy_id
                ORDER BY final_match_rate DESC
                """
            
            return self.execute_query(query, {'job_vacancy_id': job_vacancy_id})
        except Exception as e:
            print(f"Error getting match results: {e}")
            raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False