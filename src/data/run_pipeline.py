import argparse
import logging
import time
from pathlib import Path
from typing import List
import pandas as pd

from src.config import config
from src.data.extractors import EnergyChartsClient, OpenMeteoClient
from src.data.loaders import PostgresLoader
from src.data.transformers import EnergyDataTransformer, WeatherDataTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")

# ---------------------------------------------------------------------------
# Core Pipeline Steps
# ---------------------------------------------------------------------------

def extract_and_transform_historical(start_date: str, end_date: str, output_dir: Path) -> None:
    logger.info(f"=== Extraction Phase: Historical Data ({start_date} to {end_date}) ===")

    # 1. Weather Data
    try:
        with OpenMeteoClient() as weather_client:
            weather_df = _extract_weather_data(weather_client, start_date, end_date)
            
        if weather_df.empty:
            logger.error("Weather extraction yielded empty data.")
        else:
            weather_df.to_parquet(output_dir / "weather.parquet", index=False)
            logger.info("Staged historical weather dataset.")
            
    except Exception as e:
        logger.error(f"Failed to extract weather data: {e}")

    # 2. Energy Data
    try:
        with EnergyChartsClient() as energy_client:
            energy_production_df = _extract_energy_production_data(energy_client, start_date, end_date)
            installed_energy_df = _extract_installed_energy_data(energy_client)

        if not energy_production_df.empty:
            energy_production_df.to_parquet(output_dir / "energy_production.parquet", index=False)
        if not installed_energy_df.empty:
            installed_energy_df.to_parquet(output_dir / "installed_energy.parquet", index=False)
            
        logger.info("Staged energy datasets.")
        
    except Exception as e:
        logger.error(f"Failed to extract energy data: {e}")


def extract_and_transform_forecast(output_dir: Path, forecast_days: int) -> None:
    """Extracts live forecast data and stages it locally."""
    logger.info(f"=== Extraction Phase: {forecast_days}-Day Forecast ===")

    with OpenMeteoClient() as client:
        raw_forecast = client.fetch_forecast_weather(config.anchor_lats, config.anchor_lons, forecast_days=forecast_days)
        forecast_df = WeatherDataTransformer.transform(raw_forecast, config.location_tags)

    output_file = output_dir / "forecast.parquet"
    forecast_df.to_parquet(output_file, index=False)
    logger.info(f"Staged forecast dataset ({len(forecast_df)} rows) to {output_file}")


def load_to_database(directory_path: Path) -> None:
    """Loads all Parquet files from a directory into respective Postgres tables."""
    logger.info(f"=== Loading Phase: Directory {directory_path} ===")
    
    if not directory_path.exists() or not directory_path.is_dir():
        raise FileNotFoundError(f"Cannot load data: Directory {directory_path} does not exist.")

    parquet_files = list(directory_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {directory_path}.")
        return

    for file_path in parquet_files:
        table_name = file_path.stem
        logger.info(f"Loading {file_path.name} into table '{table_name}'...")
        
        df = pd.read_parquet(file_path)
        PostgresLoader().load(df, table_name=table_name, if_exists="replace")
        
    logger.info("Database load complete.")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _extract_weather_data(client: OpenMeteoClient, start_date: str, end_date: str) -> pd.DataFrame:
    chunks = []
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    current = start_dt

    chunk_dir = config.raw_dir / "weather_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    while current < end_dt:
        nxt = min(current + pd.DateOffset(years=1), end_dt) 
        chunk_file = chunk_dir / f"weather_{current.strftime('%Y%m%d')}_{nxt.strftime('%Y%m%d')}.parquet"

        if chunk_file.exists():
            logger.info(f"Loading cached chunk: {chunk_file.name}")
            df_chunk = pd.read_parquet(chunk_file)
        else:
            raw_chunk = client.fetch_historical_weather(
                config.anchor_lats, config.anchor_lons, 
                current.strftime("%Y-%m-%d"), 
                nxt.strftime("%Y-%m-%d")
            )
            df_chunk = WeatherDataTransformer.transform(raw_chunk, config.location_tags)
            
            if not df_chunk.empty:
                df_chunk.to_parquet(chunk_file, index=False)
            
            if current < end_dt:
                time.sleep(1.5)

        if not df_chunk.empty:
            chunks.append(df_chunk)
        current = nxt

    if not chunks:
        return pd.DataFrame()

    weather_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
    weather_df = weather_df.set_index("timestamp").resample("1h").mean().reset_index()

    return weather_df

def _extract_energy_production_data(client: EnergyChartsClient, start_date: str, end_date: str) -> pd.DataFrame:
    chunks = []
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    current = start_dt

    chunk_dir = config.raw_dir / "energy_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    while current < end_dt:
        nxt = min(current + pd.DateOffset(years=1), end_dt)
        chunk_file = chunk_dir / f"energy_prod_{current.strftime('%Y%m%d')}_{nxt.strftime('%Y%m%d')}.parquet"

        if chunk_file.exists():
            logger.info(f"Loading cached chunk: {chunk_file.name}")
            df_chunk = pd.read_parquet(chunk_file)
        else:
            raw_chunk = client.fetch_renewable_power_generation_data(
                country="be",
                start=current.strftime("%Y-%m-%d"),
                end=nxt.strftime("%Y-%m-%d"),
            )
            df_chunk = EnergyDataTransformer.transform(raw_chunk)
            
            if not df_chunk.empty:
                df_chunk.to_parquet(chunk_file, index=False)
            
            if current < end_dt:
                time.sleep(1.5)

        if not df_chunk.empty:
            chunks.append(df_chunk)
            
        current = nxt

    if not chunks:
        return pd.DataFrame()

    energy_production_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
    energy_production_df = energy_production_df.set_index("timestamp").resample("1h").mean().reset_index()

    return energy_production_df

def _extract_installed_energy_data(client: EnergyChartsClient) -> pd.DataFrame:
    raw_data = client.fetch_installed_power(country="be", time_step="yearly")
    installed_energy_df = EnergyDataTransformer.transform(raw_data)
    return installed_energy_df

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
    # 1. Initialize Infrastructure
    for directory in [config.raw_dir, config.interim_dir, config.processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    
    # 2. Configure Paths & Validate
    target_dir = config.processed_dir if args.mode == "train" else config.interim_dir

    if args.mode == "train" and args.step in ["extract", "run-all"]:
        if not args.start or not args.end:
            raise ValueError("Training mode extraction requires both --start and --end dates.")

    # 3. Execute Pipeline Steps
    if args.step in ["extract", "run-all"]:
        if args.mode == "train":
            extract_and_transform_historical(args.start, args.end, target_dir)
        else:
            extract_and_transform_forecast(target_dir, forecast_days=args.forecast_days)

    if args.step in ["load", "run-all"]:
        load_to_database(target_dir)


if __name__ == "__main__":
    main()