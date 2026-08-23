import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("WeatherDataTransformer")


class WeatherDataTransformer:
    """Transforms raw Open-Meteo JSON into cleaned, aligned Pandas DataFrames."""

    @classmethod
    def transform(
        self,
        payloads: List[Dict],
        location_tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Main transformation pipeline for single or multi-coordinate API payloads.

        Args:
            payloads: List of dictionaries from Open-Meteo.
            location_tags: Optional semantic labels (e.g., ['north_sea', 'flanders']).
        """
        if not payloads:
            return pd.DataFrame()

        location_dfs = [
            df
            for idx, payload in enumerate(payloads)
            if (df := self._parse_location(payload, idx, len(payloads), location_tags)) is not None
        ]

        if not location_dfs:
            logger.warning("No valid hourly time-series data found in payload.")
            return pd.DataFrame()

        merged_df = self._merge_locations(location_dfs)
        return self._clean_dataframe(merged_df)

    @staticmethod
    def _parse_location(
        payload: Dict,
        index: int,
        total_locations: int,
        location_tags: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Parses a single location payload into a standardized, tagged DataFrame."""
        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            logger.warning(f"Location at index {index} missing 'hourly.time' series.")
            return None

        df = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Append region tag if dealing with multi-location grids
        if total_locations > 1:
            tag = location_tags[index] if location_tags and index < len(location_tags) else f"loc_{index}"
            rename_map = {col: f"{col}_{tag}" for col in df.columns if col != "timestamp"}
            df = df.rename(columns=rename_map)

        return df

    @staticmethod
    def _merge_locations(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merges multiple location DataFrames side-by-side on timestamp."""
        master_df = dfs[0]
        for df in dfs[1:]:
            master_df = pd.merge(master_df, df, on="timestamp", how="outer")
        return master_df

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans, sorts, deduplicates, and fills missing feature values."""
        if df.empty:
            return df

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        feature_cols = [col for col in df.columns if col != "timestamp"]
        df[feature_cols] = df[feature_cols].fillna(0.0)

        return df