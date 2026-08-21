import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("WeatherDataTransformer")

class WeatherDataTransformer:
    """Transforms raw Open-Meteo JSON into Pandas DataFrames."""
    
    @staticmethod
    def transform(raw_data: dict) -> pd.DataFrame:
        hourly_data = raw_data.get("hourly", {})
        if not hourly_data:
            logger.warning("No hourly data found in the API response.")
            return pd.DataFrame()

        df = pd.DataFrame(hourly_data)

        return WeatherDataTransformer._clean_data(df)

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame by handling missing values and duplicates."""
        if df.empty:
            logger.warning("Received an empty DataFrame for cleaning.")
            return df
        
        df = df.rename(columns={"time": "timestamp"})

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        cols_to_clean = ["shortwave_radiation", "direct_normal_irradiance", "wind_speed_10m", "wind_speed_100m"]
        df[cols_to_clean] = df[cols_to_clean].fillna(0.0)

        df = df.drop_duplicates(subset=["timestamp"])

        return df