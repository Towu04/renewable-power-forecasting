import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("WeatherDataTransformer")

class WeatherDataTransformer:
    """Transforms raw Open-Meteo JSON into structured, aligned DataFrames."""

    @classmethod
    def transform(
        cls,
        payloads: List[Dict],
        location_tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not payloads:
            return pd.DataFrame()

        location_dfs = [
            df for idx, payload in enumerate(payloads)
            if (df := cls._parse_location(payload, idx, len(payloads), location_tags)) is not None
        ]

        if not location_dfs:
            logger.warning("No valid hourly time-series data found in payload.")
            return pd.DataFrame()

        merged_df = location_dfs[0]
        for df in location_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")

        merged_df = merged_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return merged_df.reset_index(drop=True)

    @staticmethod
    def _parse_location(
        payload: Dict,
        index: int,
        total_locations: int,
        location_tags: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            logger.warning(f"Location at index {index} missing 'hourly.time' series.")
            return None

        df = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if total_locations > 1:
            tag = location_tags[index] if location_tags and index < len(location_tags) else f"loc_{index}"
            rename_map = {col: f"{col}_{tag}" for col in df.columns if col != "timestamp"}
            df = df.rename(columns=rename_map)

        return df