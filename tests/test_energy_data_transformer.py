import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.data.transformers import EnergyDataTransformer

class TestEnergyDataTransformer:

    def test_transforms_valid_payload_correctly(self):
        """Behavior: It should flatten the JSON and cast timestamps properly."""
        mock_raw_data = {
            "data": [
                {"timestamp": "2026-01-01T00:00:00Z", "values": {"solar": 10.5, "wind_onshore": 5.0}},
                {"timestamp": "2026-01-01T01:00:00Z", "values": {"solar": 12.0, "wind_onshore": 6.5}}
            ]
        }
        
        result_df = EnergyDataTransformer.transform(mock_raw_data)
        
        # Build the expected DataFrame manually
        expected_df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True),
            "solar": [10.5, 12.0],
            "wind_onshore": [5.0, 6.5]
        })
        
        # assert_frame_equal strictly checks values, datatypes, and index
        assert_frame_equal(result_df, expected_df)

    def test_handles_empty_data_gracefully(self):
        """Behavior: An empty data array should return an empty DataFrame, not a KeyError."""
        mock_raw_data = {"data": []}
        result_df = EnergyDataTransformer.transform(mock_raw_data)
        assert result_df.empty

    def test_removes_duplicate_timestamps(self):
        """Behavior: It should drop duplicate records based on the timestamp."""
        mock_raw_data = {
            "data": [
                {"timestamp": "2026-01-01T00:00:00Z", "values": {"solar": 10.5}},
                {"timestamp": "2026-01-01T00:00:00Z", "values": {"solar": 10.5}} # Duplicate
            ]
        }
        
        result_df = EnergyDataTransformer.transform(mock_raw_data)
        assert len(result_df) == 1