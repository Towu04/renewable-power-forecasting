import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data.extractors import EnergyChartsClient, OpenMeteoClient
from src.data.loaders import PostgresLoader
from src.data.transformers import EnergyDataTransformer, WeatherDataTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "02_interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "03_processed"

ANCHOR_LATS = [51.55, 51.25, 50.85, 50.25]
ANCHOR_LONS = [2.90, 3.20, 4.35, 5.50]
LOCATION_TAGS = ["north_sea", "coast", "flanders", "wallonia"]


# ---------------------------------------------------------------------------
# Core Pipeline Steps
# ---------------------------------------------------------------------------

def extract_and_transform_historical(start_date: str, end_date: str, output_path: Path) -> None:
    """Extracts historical data from APIs, transforms it, and stages it locally."""
    logger.info(f"=== Extraction Phase: Historical Data ({start_date} to {end_date}) ===")

    # 1. Weather Data Extraction
    with OpenMeteoClient() as weather_client:
        raw_weather = weather_client.fetch_historical_weather(ANCHOR_LATS, ANCHOR_LONS, start_date, end_date)
        weather_df = WeatherDataTransformer.transform(raw_weather, LOCATION_TAGS)

    if weather_df.empty or "timestamp" not in weather_df.columns:
        raise RuntimeError("Weather data extraction failed. The resulting dataset is empty.")

    # 2. Energy Data Extraction
    chunks = []
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    current = start_dt
    
    with EnergyChartsClient() as energy_client:
        while current < end_dt:
            nxt = min(current + pd.DateOffset(months=1), end_dt)
            raw_energy = energy_client.fetch_renewable_power_generation_data(
                country="be",
                start=current.strftime("%Y-%m-%d"),
                end=nxt.strftime("%Y-%m-%d"),
            )
            df_chunk = EnergyDataTransformer.transform(raw_energy)
            if not df_chunk.empty:
                chunks.append(df_chunk)
            current = nxt

    # 3. Process and Merge
    energy_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
    energy_df = energy_df.set_index("timestamp").resample("1h").mean().reset_index()

    master_df = pd.merge(energy_df, weather_df, on="timestamp", how="inner")

    # 4. Save to Disk
    master_df.to_parquet(output_path, index=False)
    logger.info(f"Staged historical dataset ({len(master_df)} rows) to {output_path}")


def extract_and_transform_forecast(output_path: Path, forecast_days: int) -> None:
    """Extracts live forecast data and stages it locally."""
    logger.info(f"=== Extraction Phase: {forecast_days}-Day Forecast ===")

    with OpenMeteoClient() as client:
        raw_forecast = client.fetch_forecast_weather(ANCHOR_LATS, ANCHOR_LONS, forecast_days=forecast_days)
        forecast_df = WeatherDataTransformer.transform(raw_forecast, LOCATION_TAGS)

    forecast_df.to_parquet(output_path, index=False)
    logger.info(f"Staged forecast dataset ({len(forecast_df)} rows) to {output_path}")


def load_to_database(file_path: Path, table_name: str) -> None:
    """Loads a staged Parquet file into the Postgres database."""
    logger.info(f"=== Loading Phase: {file_path.name} -> {table_name} ===")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot load data: {file_path} does not exist. Run extraction first.")

    df = pd.read_parquet(file_path)
    PostgresLoader().load(df, table_name=table_name, if_exists="replace")
    logger.info("Database load complete.")


# ---------------------------------------------------------------------------
# CLI & Orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renewable Energy Data Pipeline")
    parser.add_argument(
        "--step", 
        choices=["extract", "load", "run-all"], 
        required=True, 
        help="Run just the extraction/transformation, just the DB load, or both sequentially."
    )
    parser.add_argument(
        "--mode", 
        choices=["train", "inference"], 
        required=True,
        help="Train mode pulls historical data; inference mode pulls future forecasts."
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) for train mode")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD) for train mode")
    parser.add_argument(
        "--forecast-days", 
        type=int, 
        default=3, 
        choices=range(1, 17),
        help="Number of days to forecast for inference mode (1-16). Default is 3."
    )
    return parser.parse_args()


def main() -> None:
    for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    
    if args.mode == "train":
        if args.step in ["extract", "run-all"] and (not args.start or not args.end):
            raise ValueError("Training mode extraction requires both --start and --end dates.")
        target_file = PROCESSED_DIR / "historical_training.parquet"
        target_table = "historical_training"
    else:
        target_file = INTERIM_DIR / "current_forecast.parquet"
        target_table = "daily_forecasts"

    if args.step in ["extract", "run-all"]:
        if args.mode == "train":
            extract_and_transform_historical(args.start, args.end, target_file)
        else:
            extract_and_transform_forecast(target_file, forecast_days=args.forecast_days)

    if args.step in ["load", "run-all"]:
        load_to_database(target_file, target_table)


if __name__ == "__main__":
    main()