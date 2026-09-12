from datetime import datetime
import logging
from typing import List, Dict, Any, Sequence, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OpenMeteoClient")

class OpenMeteoValidationError(ValueError):
    """Raised when coordinate or temporal parameters violate API boundaries."""
    pass

class OpenMeteoClient:
    """Pure API Client for extracting raw Open-Meteo weather data across multiple locations."""

    FORECAST_BASE_URL = "https://api.open-meteo.com"
    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com"
    DEFAULT_TIMEOUT = 15

    DEFAULT_HOURLY_VARIABLES: List[str] = [
        # --- Solar Features ---
        "temperature_2m",              
        "shortwave_radiation",         
        "direct_normal_irradiance",    
        "diffuse_radiation",           
        "cloud_cover",                
        
        # --- Wind Features ---
        "wind_speed_10m",             
        "wind_speed_100m",             
        "wind_direction_100m",         
        "wind_gusts_10m",              
        "surface_pressure"            
    ]

    def __init__(self):
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def __enter__(self) -> "OpenMeteoClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.session.close()

    def _validate_coordinates(self, lats: Sequence[float], lons: Sequence[float]) -> None:
        if not lats or not lons:
            raise OpenMeteoValidationError("Coordinate lists must not be empty.")
        if len(lats) != len(lons):
            raise OpenMeteoValidationError(
                f"Coordinates count mismatch: {len(lats)} latitudes vs {len(lons)} longitudes."
            )
        for lat in lats:
            if not (-90.0 <= lat <= 90.0):
                raise OpenMeteoValidationError(f"Invalid latitude {lat}. Must be between -90 and 90.")
        for lon in lons:
            if not (-180.0 <= lon <= 180.0):
                raise OpenMeteoValidationError(f"Invalid longitude {lon}. Must be between -180 and 180.")

    def _validate_date(self, date_str: str, field_name: str) -> None:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise OpenMeteoValidationError(
                f"Invalid or impossible date for '{field_name}': '{date_str}'. Expected a valid 'YYYY-MM-DD' calendar date."
            )

    def fetch_historical_weather(
        self, 
        lats: Sequence[float], 
        lons: Sequence[float], 
        start_date: str, 
        end_date: str,
        hourly_vars: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetches historical reanalysis weather data. Returns one dictionary per coordinate pair."""
        self._validate_coordinates(lats, lons)
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        
        if start_date > end_date:
            raise OpenMeteoValidationError(f"start_date '{start_date}' cannot be after end_date '{end_date}'.")

        params = {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars or self.DEFAULT_HOURLY_VARIABLES),
            "timezone": "UTC"
        }

        data = self._make_api_request(self.ARCHIVE_BASE_URL, "/v1/archive", params)
        return data if isinstance(data, list) else [data]

    def fetch_forecast_weather(
        self, 
        lats: Sequence[float], 
        lons: Sequence[float], 
        forecast_days: int = 3,
        hourly_vars: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetches NWP weather forecast. Returns one dictionary per coordinate pair."""
        self._validate_coordinates(lats, lons)
        
        if not (1 <= forecast_days <= 16):
            raise OpenMeteoValidationError(f"forecast_days must be between 1 and 16, got {forecast_days}.")

        params = {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "hourly": ",".join(hourly_vars or self.DEFAULT_HOURLY_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": "UTC"
        }

        data = self._make_api_request(self.FORECAST_BASE_URL, "/v1/forecast", params)
        return data if isinstance(data, list) else [data]

    def _make_api_request(self, base_url: str, endpoint: str, params: Dict[str, Any]) -> Any:
        url = f"{base_url}{endpoint}"
        logger.info(f"OpenMeteo: Requesting data from {url}")
        
        try:
            response = self.session.get(url, params=params, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API request failed: {e}")
            raise