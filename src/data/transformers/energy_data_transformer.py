import logging
import pandas as pd

logger = logging.getLogger("EnergyDataTransformer")

class EnergyDataTransformer:
    """Transforms raw Fraunhofer JSON into a structured, faithful DataFrame."""

    @staticmethod
    def transform(raw_data: dict) -> pd.DataFrame:
        records = raw_data.get("data", [])
        if not records:
            logger.warning("API returned no data records.")
            return pd.DataFrame()

        rows = []
        for entry in records:
            row = {"timestamp": entry.get("timestamp")}
            row.update(entry.get("values", {}))
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return df.reset_index(drop=True)