import pytest
import requests
from unittest.mock import patch, MagicMock

from src.data.extractors.energy_charts_client import EnergyChartsClient, APIValidationError

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_api_schema():
    """Simulates the /v2 catalog response for predictable testing."""
    return {
        "endpoints": [
            {
                "path": "/v2/installed_power",
                "parameters": [
                    {"name": "country", "required": True, "allowed_values": ["be", "de", "fr"]},
                    {"name": "time_step", "required": False, "allowed_values": ["yearly", "monthly"]}
                ]
            },
            {
                "path": "/v2/public_power",
                "parameters": [
                    {"name": "country", "required": True, "allowed_values": ["be", "de"]},
                    {"name": "start", "required": False}
                ]
            }
        ]
    }

@pytest.fixture
def client(mock_api_schema):
    """Provides a client instance with a pre-loaded mock schema, bypassing the init network call."""
    with patch.object(requests.Session, 'get') as mock_get:
        # Mock the initial /v2 schema fetch
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_schema
        mock_get.return_value = mock_response
        
        return EnergyChartsClient()

# ==========================================
# TESTS
# ==========================================

class TestClientInitialization:
    
    @patch.object(requests.Session, 'get')
    def test_init_survives_schema_fetch_failure(self, mock_get):
        """
        Behavior: If the API is down when the client boots, it should not crash.
        It should fail gracefully, disable validation, and allow subsequent requests to try.
        """
        # Simulate a 500 Internal Server Error on initialization
        mock_get.side_effect = requests.exceptions.HTTPError("500 Server Error")
        
        client = EnergyChartsClient()
        
        # Client should instantiate successfully with an empty schema
        assert client.api_schema == {}

class TestParameterValidation:
    
    @patch.object(requests.Session, 'get')
    def test_blocks_invalid_enum_value(self, mock_get, client):
        """Behavior: Fails fast locally if a parameter is not in the allowed list."""
        with pytest.raises(APIValidationError) as exc_info:
            client.fetch_installed_power(country="us") # "us" is not in our mock schema
            
        assert "'us' is not a valid 'country'" in str(exc_info.value)
        mock_get.assert_not_called() # Proves the request never hit the network

    @patch.object(requests.Session, 'get')
    def test_blocks_missing_required_parameter(self, mock_get, client):
        """Behavior: Fails fast locally if a strictly required parameter is absent."""
        with pytest.raises(APIValidationError) as exc_info:
            # Manually triggering the internal method to simulate missing 'country'
            client._make_api_request("/v2/installed_power", {"time_step": "yearly"})
            
        assert "Missing required parameter 'country'" in str(exc_info.value)
        mock_get.assert_not_called()

    @patch.object(requests.Session, 'get')
    def test_allows_unknown_parameters_for_forward_compatibility(self, mock_get, client, caplog):
        """
        Behavior: If the user passes a parameter the schema doesn't know, it should warn 
        but NOT block the request. The API might have updated before our schema refreshed.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_get.return_value = mock_response

        # 'future_filter' doesn't exist in our mock schema
        client._make_api_request("/v2/installed_power", {"country": "be", "future_filter": "true"})
        
        # Assert the request was still sent
        mock_get.assert_called_once()
        # Assert a warning was logged
        assert "is not recognized by the API schema" in caplog.text

class TestDataFormatting:

    @patch.object(requests.Session, 'get')
    def test_drops_empty_string_parameters(self, mock_get, client):
        """Behavior: Empty strings should be stripped so they don't corrupt the URL query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        # User provides empty start/end/subtype strings
        client.fetch_renewable_power_generation_data(country="be", start="", end="")
        
        # Verify the actual params sent to requests.get only contained 'country'
        call_args = mock_get.call_args[1]
        assert "start" not in call_args["params"]
        assert "end" not in call_args["params"]
        assert call_args["params"]["country"] == "be"

class TestNetworkResilience:

    @patch.object(requests.Session, 'get')
    def test_raises_http_errors_correctly(self, mock_get, client):
        """Behavior: 404s, 500s, and other HTTP errors should not be swallowed."""
        
        # Create a mock response that throws when raise_for_status() is called
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            client.fetch_installed_power(country="be")      