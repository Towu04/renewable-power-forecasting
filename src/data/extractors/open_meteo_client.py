import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

logger = logging.getLogger(__name__)

class OpenMeteoClient:
    """Client for interacting with the Open-Meteo API"""

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

    def fetch_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical solar and wind data for training machine learning models.
        
        Args:
            lat: Latitude 
            lon: Longitude 
            start_date: Format 'YYYY-MM-DD'
            end_date: Format 'YYYY-MM-DD'
        """
        endpoint = "/v1/archive"
        
        variables = [
            "shortwave_radiation",      # Total solar energy hitting the ground
            "direct_normal_irradiance", # Direct sunlight (DNI)
            "wind_speed_10m",           # Surface wind
            "wind_speed_100m"           # Turbine-height wind
        ]

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "auto"
        }

        raw_data = self._make_api_request(self.ARCHIVE_BASE_URL, endpoint, params)
        return self._transform_to_dataframe(raw_data)

    def _make_api_request(self, base_url: str, endpoint: str, params: dict) -> dict:
        """Handles the HTTP GET request and error handling safely."""
        url = f"{base_url}{endpoint}"
        logger.info(f"Requesting weather data from: {url}")

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API request failed: {e}")
            raise

    def _transform_to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """Transforms the Open-Meteo JSON response into a pandas DataFrame."""
        # Open-Meteo packs all the time-series arrays inside an "hourly" key
        hourly_data = raw_data.get("hourly", {})
        
        if not hourly_data:
            logger.warning("No hourly data found in the API response.")
            return pd.DataFrame()

        df = pd.DataFrame(hourly_data)

        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.rename(columns={"time": "timestamp"})

        return df