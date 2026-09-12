import pytest
import requests
from unittest.mock import patch, MagicMock

from src.data.extractors.open_meteo_client import OpenMeteoClient, OpenMeteoValidationError

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def client():
    """Provides a fresh client instance for each test."""
    return OpenMeteoClient()

# ==========================================
# TESTS
# ==========================================

class TestContextManager:

    def test_context_manager_closes_session(self):
        """Behavior: When exiting a 'with' block, the underlying requests.Session must be closed."""
        with OpenMeteoClient() as client:
            session_mock = MagicMock()
            client.session = session_mock
            
        # Once the block exits, close() should have been called automatically
        session_mock.close.assert_called_once()


class TestCoordinateValidation:
    """Tests the fail-fast boundary validations for geospatial inputs."""

    def test_blocks_empty_coordinates(self, client):
        """Behavior: Cannot pass empty lists for coordinates."""
        with pytest.raises(OpenMeteoValidationError, match="Coordinate lists must not be empty"):
            client.fetch_forecast_weather(lats=[], lons=[])

    def test_blocks_mismatched_coordinate_lengths(self, client):
        """Behavior: Latitudes and Longitudes arrays must be strictly paired 1:1."""
        with pytest.raises(OpenMeteoValidationError, match="Coordinates count mismatch"):
            client.fetch_historical_weather(lats=[50.85, 51.21], lons=[4.35], start_date="2026-01-01", end_date="2026-01-02")

    def test_blocks_out_of_bounds_latitude(self, client):
        """Behavior: Latitude must be strictly between -90 and 90."""
        with pytest.raises(OpenMeteoValidationError, match="Invalid latitude 95.0"):
            client.fetch_forecast_weather(lats=[95.0], lons=[4.35])

    def test_blocks_out_of_bounds_longitude(self, client):
        """Behavior: Longitude must be strictly between -180 and 180."""
        with pytest.raises(OpenMeteoValidationError, match="Invalid longitude -200.0"):
            client.fetch_forecast_weather(lats=[50.85], lons=[-200.0])


class TestTemporalValidation:
    """Tests the fail-fast temporal validations for historical and forecast data."""

    def test_blocks_invalid_date_format(self, client):
        """Behavior: API requires strictly YYYY-MM-DD."""
        with pytest.raises(OpenMeteoValidationError, match="Invalid or impossible date for 'start_date': '01-01-2026'. Expected a valid 'YYYY-MM-DD' calendar date."):
            # Passing DD-MM-YYYY instead of YYYY-MM-DD
            client.fetch_historical_weather(lats=[50.85], lons=[4.35], start_date="01-01-2026", end_date="2026-01-02")

    def test_blocks_chronological_mismatch(self, client):
        """Behavior: Start date cannot be in the future of end date."""
        with pytest.raises(OpenMeteoValidationError, match="cannot be after end_date"):
            client.fetch_historical_weather(lats=[50.85], lons=[4.35], start_date="2026-02-01", end_date="2026-01-01")

    def test_blocks_invalid_forecast_days(self, client):
        """Behavior: Open-Meteo restricts forecast days between 1 and 16."""
        with pytest.raises(OpenMeteoValidationError, match="forecast_days must be between 1 and 16"):
            client.fetch_forecast_weather(lats=[50.85], lons=[4.35], forecast_days=20)


class TestDataFormatting:
    """Tests the URL parameter generation and response list normalization."""

    @patch.object(requests.Session, 'get')
    def test_comma_separated_parameter_generation(self, mock_get, client):
        """Behavior: Arrays of coordinates must be collapsed into comma-separated strings for the URL query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        client.fetch_forecast_weather(lats=[50.85, 51.21], lons=[4.35, 4.40])
        
        call_args = mock_get.call_args[1]["params"]
        assert call_args["latitude"] == "50.85,51.21"
        assert call_args["longitude"] == "4.35,4.4"

    @patch.object(requests.Session, 'get')
    def test_normalizes_single_location_dict_to_list(self, mock_get, client):
        """Behavior: Open-Meteo returns a pure Dict for 1 location. The client must wrap it in a List."""
        mock_response = MagicMock()
        # Mocking a single-dictionary return
        mock_response.json.return_value = {"latitude": 50.85, "elevation": 15}
        mock_get.return_value = mock_response

        result = client.fetch_forecast_weather(lats=[50.85], lons=[4.35])
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["latitude"] == 50.85

    @patch.object(requests.Session, 'get')
    def test_preserves_multi_location_list(self, mock_get, client):
        """Behavior: Open-Meteo returns a List for multiple locations. The client must return it as-is."""
        mock_response = MagicMock()
        # Mocking a list return
        mock_response.json.return_value = [{"latitude": 50.85}, {"latitude": 51.21}]
        mock_get.return_value = mock_response

        result = client.fetch_forecast_weather(lats=[50.85, 51.21], lons=[4.35, 4.40])
        
        assert isinstance(result, list)
        assert len(result) == 2


class TestNetworkResilience:

    @patch.object(requests.Session, 'get')
    def test_raises_http_errors_correctly(self, mock_get, client):
        """Behavior: 400s, 429s, and 500s should be raised, not swallowed."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            client.fetch_forecast_weather(lats=[50.85], lons=[4.35])