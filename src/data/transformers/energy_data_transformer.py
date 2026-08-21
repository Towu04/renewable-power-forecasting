import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EnergyDataTransformer")

class EnergyDataTransformer:
    """Transforms raw Fraunhofer JSON into clean Pandas DataFrames."""
    
    @staticmethod
    def transform(raw_data: dict) -> pd.DataFrame:
        records = raw_data.get("data", [])
        if not records:
            logger.warning("API returned no data records.")
            return pd.DataFrame()

        rows = []
        for entry in records:
            timestamp = entry.get("timestamp")
            values = entry.get("values", {})

            rows.append({
                "timestamp": timestamp,
                "solar": values.get("solar"),
                "wind_onshore": values.get("wind_onshore"),
                "wind_offshore": values.get("wind_offshore"),
            })

        df = pd.DataFrame(rows)

    
        return EnergyDataTransformer._clean_data(df)

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame by handling missing values and duplicates."""
        if df.empty:
            logger.warning("Received an empty DataFrame for cleaning.")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        
        df = df.sort_values("timestamp").reset_index(drop=True)
            
        cols_to_clean = ["solar", "wind_onshore", "wind_offshore"]
        df[cols_to_clean] = df[cols_to_clean].fillna(0.0)
            
        df = df.drop_duplicates(subset=["timestamp"])

        return df