import logging
from typing import List, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OpenMeteoClient")

class OpenMeteoClient:
    """Pure API Client for extracting raw Open-Meteo JSON data across multiple locations."""

    FORECAST_BASE_URL = "https://api.open-meteo.com"
    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com"

    def __init__(self):
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)

    def fetch_historical_weather(self, lats: List[float], lons: List[float], start_date: str, end_date: str) -> List[dict]:
        """Fetches historical solar and wind data. Returns a list of JSONs if multiple coordinates provided."""
        
        if len(lats) != len(lons):
            raise ValueError("Latitude and Longitude lists must be of the same length.")

        endpoint = "/v1/archive"
        variables = [
            "temperature_2m",
            "shortwave_radiation", 
            "direct_normal_irradiance", 
            "wind_speed_10m", 
            "wind_speed_100m"
        ]
        
        # Convert lists of floats into comma-separated strings for the API
        params = {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "auto"
        }

        data = self._make_api_request(self.ARCHIVE_BASE_URL, endpoint, params)

        return data if isinstance(data, list) else [data]

    def fetch_forecast_weather(self, lats: List[float], lons: List[float], forecast_days: int = 3) -> List[dict]:
        """Fetches live forecast data. Returns a list of JSONs if multiple coordinates provided."""
        
        if len(lats) != len(lons):
            raise ValueError("Latitude and Longitude lists must be of the same length.")
            
        endpoint = "/v1/forecast"
        variables = [
            "temperature_2m",
            "shortwave_radiation", 
            "direct_normal_irradiance", 
            "wind_speed_10m", 
            "wind_speed_100m"
        ]
        
        params = {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "hourly": ",".join(variables),
            "forecast_days": forecast_days,
            "timezone": "auto"
        }

        data = self._make_api_request(self.FORECAST_BASE_URL, endpoint, params)
        return data if isinstance(data, list) else [data]

    def _make_api_request(self, base_url: str, endpoint: str, params: dict) -> Union[dict, List[dict]]:
        url = f"{base_url}{endpoint}"
        logger.info(f"Requesting weather data from: {url}")
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API request failed: {e}")
            raise