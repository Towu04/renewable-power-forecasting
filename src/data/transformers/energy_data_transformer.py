import logging
import pandas as pd

logger = logging.getLogger(__name__)

class EnergyDataTransformer:
    """Transforms raw Fraunhofer JSON into Pandas DataFrames."""
    
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

        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df