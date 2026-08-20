import logging
import requests
from requests.adapters import HTTPAdapter
import pandas as pd
from urllib3 import Retry

logger = logging.getLogger(__name__)

class EnergyChartsClient:
    """client for interacting with the public Fraunhofer energy-charts API"""

    BASE_URL = "https://api.energy-charts.info"

    def __init__(self):
        self.session = requests.Session()

        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

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
        endpoint = f"{self.BASE_URL}/public_power"
        params = {"country": country, "start": start, "end": end, "subtype": subtype}
        raw_data = self._make_api_request(endpoint, params)
        df = self._transform_to_dataframe(raw_data)
        return df

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
        url = f"{self.BASE_URL}{endpoint}"
        logger.info(f"Requesting URL: {url} with params: {params}")

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise



    def _transform_to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """Transforms the raw API response into a pandas DataFrame.

            Args:
                raw_data: The raw JSON response from the API.

            Returns:
                A pandas DataFrame with timestamps and generation values in MW.
        """
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
