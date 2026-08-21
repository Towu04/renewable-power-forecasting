import logging
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
    """Pure API Client for extracting raw Open-Meteo JSON data."""

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

    def fetch_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> dict:
        """Fetches historical solar and wind data and returns raw JSON."""
        endpoint = "/v1/archive"
        variables = [
            "shortwave_radiation", 
            "direct_normal_irradiance", 
            "wind_speed_10m", 
            "wind_speed_100m"
        ]
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "auto"
        }

        return self._make_api_request(self.ARCHIVE_BASE_URL, endpoint, params)

    def fetch_forecast_weather(self, lat: float, lon: float, forecast_days: int = 3) -> dict:
        """
        Fetches live forecast data and returns raw JSON.
        
        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: How many days into the future to predict (default 3)
        """
        endpoint = "/v1/forecast"
        
        variables = [
            "shortwave_radiation", 
            "direct_normal_irradiance", 
            "wind_speed_10m", 
            "wind_speed_100m"
        ]
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "forecast_days": forecast_days,
            "timezone": "auto"
        }

        return self._make_api_request(self.FORECAST_BASE_URL, endpoint, params)

    def _make_api_request(self, base_url: str, endpoint: str, params: dict) -> dict:
        url = f"{base_url}{endpoint}"
        logger.info(f"Requesting weather data from: {url}")
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API request failed: {e}")
            raise