import pandas as pd
from pandas.testing import assert_frame_equal
import logging

from src.data.transformers.weather_data_transformer import WeatherDataTransformer

class TestWeatherDataTransformer:

    def test_transforms_single_location_correctly(self):
        """Behavior: A single location payload should parse without appending tags to columns."""
        payloads = [{
            "hourly": {
                "time": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                "temperature_2m": [5.5, 6.0],
                "wind_speed_10m": [15.2, 16.0]
            }
        }]
        
        result_df = WeatherDataTransformer.transform(payloads)
        
        expected_df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True),
            "temperature_2m": [5.5, 6.0],
            "wind_speed_10m": [15.2, 16.0]
        })
        
        assert_frame_equal(result_df, expected_df)

    def test_transforms_and_merges_multiple_locations_with_tags(self):
        """Behavior: Multiple payloads should outer-join on timestamp and append provided tags to feature columns."""
        payloads = [
            {
                "hourly": {
                    "time": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                    "temperature_2m": [5.5, 6.0]
                }
            },
            {
                "hourly": {
                    "time": ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"], # Note the overlapping but different times
                    "temperature_2m": [7.0, 8.0]
                }
            }
        ]
        
        result_df = WeatherDataTransformer.transform(payloads, location_tags=["coast", "inland"])
        
        expected_df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-01T00:00:00Z", 
                "2026-01-01T01:00:00Z", 
                "2026-01-01T02:00:00Z"
            ], utc=True),
            "temperature_2m_coast": [5.5, 6.0, float('nan')],
            "temperature_2m_inland": [7.0, float('nan'), 8.0]
        })
        
        assert_frame_equal(result_df, expected_df)

    def test_applies_fallback_tags_when_tags_omitted(self):
        """Behavior: If tags are omitted for multi-location data, it should fallback to 'loc_0', 'loc_1'."""
        payloads = [
            {"hourly": {"time": ["2026-01-01T00:00:00Z"], "temperature_2m": [5.5]}},
            {"hourly": {"time": ["2026-01-01T00:00:00Z"], "temperature_2m": [7.0]}}
        ]
        
        result_df = WeatherDataTransformer.transform(payloads)
        
        assert "temperature_2m_loc_0" in result_df.columns
        assert "temperature_2m_loc_1" in result_df.columns

    def test_returns_empty_dataframe_for_empty_payload(self):
        """Behavior: Supplying an empty list should return an empty DataFrame without errors."""
        result_df = WeatherDataTransformer.transform([])
        assert result_df.empty
        assert isinstance(result_df, pd.DataFrame)

    def test_handles_malformed_payload_gracefully(self, caplog):
        """Behavior: If 'hourly' or 'time' is missing, it should log a warning and skip that payload."""
        # Use caplog to capture the logger output
        with caplog.at_level(logging.WARNING):
            payloads = [
                {"hourly": {"temperature_2m": [5.5]}}, # Missing 'time'
                {"daily": {"time": ["2026-01-01T00:00:00Z"]}} # Missing 'hourly' entirely
            ]
            
            result_df = WeatherDataTransformer.transform(payloads)
            
            assert result_df.empty
            assert "missing 'hourly.time' series" in caplog.text

    def test_deduplicates_overlapping_timestamps(self):
        """Behavior: Duplicate timestamps inside a payload should be cleanly dropped."""
        payloads = [{
            "hourly": {
                "time": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], # Duplicate
                "temperature_2m": [5.5, 9.9]
            }
        }]
        
        result_df = WeatherDataTransformer.transform(payloads)
        
        # Should only keep the first occurrence
        assert len(result_df) == 1
        assert result_df.iloc[0]["temperature_2m"] == 5.5