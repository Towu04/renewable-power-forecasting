import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

class EnergyChartsClient:
    """client for interacting with the public Fraunhofer energy-charts API"""

    BASE_URL = "https://api.energy-charts.info"

    def __init__(self):
        self.session = requests.Session()

    def fetch_power_generation_data(
            self, 
            country: str = "de", 
            start: str = "", 
            end: str = "", 
            subtype: str = ""
        ) -> pd.DataFrame:
        """Fetches solar and wind generation data for a given country and timeframe.

            Args:
                country: Country code (e.g., 'be', 'de', 'fr'). Defaults to 'de'
                start: Start date  (e.g., '2026-01-01')
                end: End date or timestamp.
                subtype: Optional sub-category filter (e.g., 'solarlog')

            Returns:
                A pandas DataFrame containing timestamps and generation values in MW.
            
            Raises:
                requests.exceptions.RequestException: If the HTTP request fails or 
                    the API returns a non-200 status code.
            """
        pass # TODO

    def _build_params(self, country: str, start: str, end: str, subtype: str) -> dict:
        """Cleans and filters out empty parameters"""
        pass # TODO

    def _make_api_request(self, endpoint: str, params: dict) -> dict:
        """Makes a GET request to the specified API endpoint with given parameters.

            Args:
                endpoint: The API endpoint to call (e.g., '/generation').
                params: A dictionary of query parameters.

            Returns:
                The JSON response from the API as a dictionary.

            Raises:
                requests.exceptions.RequestException: If the HTTP request fails or 
                    the API returns a non-200 status code.
        """
        pass # TODO



    def _transform_to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """Transforms the raw API response into a pandas DataFrame.

            Args:
                raw_data: The raw JSON response from the API.

            Returns:
                A pandas DataFrame with timestamps and generation values in MW.
        """
        pass # TODO
