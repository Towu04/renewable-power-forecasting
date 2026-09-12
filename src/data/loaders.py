import logging
import os
import time
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("DatabaseLoader")


class PostgresLoader:
    """Uploads DataFrames to PostgreSQL with automatic retries."""

    def __init__(self):
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is missing from .env file.")
        self.engine = create_engine(db_url, pool_pre_ping=True)

    def load(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace", retries: int = 3) -> None:
        """Uploads a DataFrame to a PostgreSQL table."""
        if df.empty:
            logger.warning(f"DataFrame for '{table_name}' is empty. Skipping upload.")
            return

        for attempt in range(0, retries + 1):
            try:
                logger.info(f"Uploading {len(df)} rows to '{table_name}' (attempt {attempt}/{retries})...")
                df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=False, chunksize=1000)
                logger.info(f"Successfully loaded '{table_name}'.")
                return
            except Exception as e:
                logger.warning(f"Upload attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise
                time.sleep(2 * (attempt + 1))