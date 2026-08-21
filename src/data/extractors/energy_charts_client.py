import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

logger = logging.getLogger(__name__)

class EnergyChartsClient:
    """Pure API Client for extracting raw Fraunhofer JSON data."""

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

    def fetch_renewable_power_generation_data(
            self, country: str = "be", start: str = "", end: str = "", subtype: str = ""
        ) -> dict:
        """Fetches solar and wind generation data and returns raw JSON."""
        endpoint = f"{self.BASE_URL}/public_power"
        
        # Clean empty params
        raw_params = {"country": country, "start": start, "end": end, "subtype": subtype}
        params = {k: v for k, v in raw_params.items() if v}
        
        # Returns the raw dictionary, stops here.
        return self._make_api_request(endpoint, params)

    def _make_api_request(self, endpoint: str, params: dict) -> dict:
        url = f"{self.BASE_URL}{endpoint}"
        logger.info(f"Requesting URL: {url} with params: {params}")
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise