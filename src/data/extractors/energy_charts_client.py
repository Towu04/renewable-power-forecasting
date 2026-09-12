import logging
from typing import Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EnergyChartsClient")

class EnergyChartsValidationError(ValueError):
    """Raised when request parameters fail client-side schema validation."""
    pass

class EnergyChartsClient:
    """Pure API Client for extracting raw Fraunhofer JSON data."""

    BASE_URL = "https://api.energy-charts.info"
    DEFAULT_TIMEOUT = 10

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

        self.api_schema = self._fetch_api_catalog()

    def __enter__(self) -> "EnergyChartsClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.session.close()

    def _fetch_api_catalog(self) -> Dict[str, Any]:
        """Fetches the /v2 directory to be used for dynamic parameter validation."""
        try:
            return self._make_api_request("/v2", {}, validate=False)
        except Exception as e:
            logger.warning(f"Could not load API schema. Pre-flight validation disabled: {e}")
            return {}

    def fetch_renewable_power_generation_data(
        self, country: str = "be", start: str = "", end: str = "", subtype: str = ""
    ) -> Dict[str, Any]:
        """Fetches solar and wind generation data and returns raw JSON."""
        endpoint = "/v2/public_power"
        
        raw_params = {"country": country, "start": start, "end": end, "subtype": subtype}
        params = {k: v for k, v in raw_params.items() if v}
        
        return self._make_api_request(endpoint, params)

    def fetch_installed_power(self, country: str = "be", time_step: str = "yearly") -> Dict[str, Any]:
        """Fetches installed power capacity data and returns raw JSON."""
        params = {"country": country, "time_step": time_step}
        return self._make_api_request("/v2/installed_power", params)

    def _validate_params(self, endpoint: str, params: Dict[str, Any]) -> None:
        """Validates parameters against the API schema before sending the request."""
        if not self.api_schema:
            return 

        endpoints_list = self.api_schema.get("endpoints", [])
        endpoint_schema = next((ep for ep in endpoints_list if ep.get("path") == endpoint), None)

        if not endpoint_schema:
            logger.debug(f"Endpoint '{endpoint}' not found in schema. Skipping validation.")
            return

        schema_params = {p["name"]: p for p in endpoint_schema.get("parameters", [])}

        for param_name, param_def in schema_params.items():
            if param_def.get("required") and param_name not in params:
                raise EnergyChartsValidationError(
                    f"Validation Error: Missing required parameter '{param_name}' "
                    f"for endpoint '{endpoint}'."
                )

        for key, value in params.items():
            param_def = schema_params.get(key)
            
            if not param_def:
                logger.warning(f"Parameter '{key}' is not recognized by the API schema for '{endpoint}'.")
                continue

            allowed_values = param_def.get("allowed_values")
            if allowed_values and value not in allowed_values:
                if str(value) not in [str(v) for v in allowed_values]:
                    raise EnergyChartsValidationError(
                        f"Validation Error: '{value}' is not a valid '{key}'. "
                        f"Allowed values: {allowed_values}"
                    )

    def _make_api_request(self, endpoint: str, params: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        if validate:
            self._validate_params(endpoint, params)
        
        url = f"{self.BASE_URL}{endpoint}"
        logger.info(f"EnergyCharts: Requesting data from {url}")
        
        try:
            response = self.session.get(url, params=params, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Energy-Charts API request failed: {e}")
            raise