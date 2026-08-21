import logging
import pandas as pd

logger = logging.getLogger(__name__)

class OpenMeteoTransformer:
    """Transforms raw Open-Meteo JSON into Pandas DataFrames."""
    
    @staticmethod
    def transform(raw_data: dict) -> pd.DataFrame:
        hourly_data = raw_data.get("hourly", {})
        if not hourly_data:
            logger.warning("No hourly data found in the API response.")
            return pd.DataFrame()

        df = pd.DataFrame(hourly_data)

        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.rename(columns={"time": "timestamp"})

        return df