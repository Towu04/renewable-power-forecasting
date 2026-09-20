import argparse
import logging
import time
from pathlib import Path
import pandas as pd

from src.config import config
from src.data.extractors import EnergyChartsClient, OpenMeteoClient
from src.data.loaders import PostgresLoader
from src.data.transformers import EnergyDataTransformer, WeatherDataTransformer
from src.features.build_features import FeatureBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")

class DataPipeline:
    """Orchestrates the flow of data through the Medallion architecture."""
    
    def __init__(self):
        for directory in [config.raw_dir, config.interim_dir, config.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _generate_date_chunks(self, start_date: str, end_date: str):
        """Yields start and end dates for API chunking to prevent memory bloat and rate limits."""
        current = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        while current < end_dt:
            nxt = min(current + pd.DateOffset(months=config.chunk_size_months), end_dt)
            yield current, nxt
            current = nxt

    def extract_weather(self, start_date: str, end_date: str) -> None:
        """Extracts Weather API data, transforms to Interim, and saves."""
        logger.info("Extracting Weather Data...")
        chunks = []
        
        with OpenMeteoClient() as client:
            for start, end in self._generate_date_chunks(start_date, end_date):
                chunk_file = config.raw_dir / f"weather_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
                
                if chunk_file.exists():
                    logger.info(f"Loaded from cache: {chunk_file.name}")
                    df_chunk = pd.read_parquet(chunk_file)
                else:
                    raw_data = client.fetch_historical_weather(
                        config.anchor_lats, config.anchor_lons,
                        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                    )
                    df_chunk = WeatherDataTransformer.transform(raw_data, config.location_tags)
                    
                    if not df_chunk.empty:
                        df_chunk.to_parquet(chunk_file, index=False)
                    time.sleep(config.throttle_sleep_seconds)
                
                if not df_chunk.empty:
                    chunks.append(df_chunk)

        if chunks:
            # Combine chunks and save to Silver/Interim. No resampling here!
            interim_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
            interim_df.to_parquet(config.interim_dir / "weather.parquet", index=False)
            logger.info("Weather data successfully staged to Interim.")

    def extract_energy(self, start_date: str, end_date: str) -> None:
        """Extracts Energy API data, transforms to Interim, and saves."""
        logger.info("Extracting Energy Production Data...")
        chunks = []
        
        with EnergyChartsClient() as client:
            for start, end in self._generate_date_chunks(start_date, end_date):
                chunk_file = config.raw_dir / f"energy_prod_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
                
                if chunk_file.exists():
                    logger.info(f"Loaded from cache: {chunk_file.name}")
                    df_chunk = pd.read_parquet(chunk_file)
                else:
                    raw_data = client.fetch_renewable_power_generation_data(
                        country="be", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
                    )
                    df_chunk = EnergyDataTransformer.transform(raw_data)
                    
                    if not df_chunk.empty:
                        df_chunk.to_parquet(chunk_file, index=False)
                    time.sleep(config.throttle_sleep_seconds)
                
                if not df_chunk.empty:
                    chunks.append(df_chunk)
                    
            installed_raw = client.fetch_installed_power(country="be", time_step="yearly")
            installed_df = EnergyDataTransformer.transform(installed_raw)
            if not installed_df.empty:
                installed_df.to_parquet(config.interim_dir / "installed_energy.parquet", index=False)

        if chunks:
            interim_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
            interim_df.to_parquet(config.interim_dir / "energy_production.parquet", index=False)
            logger.info("Energy data successfully staged to Interim.")

    def load_to_database(self) -> None:
        """Loads final Processed data to Postgres."""
        target_file = config.processed_dir / "historical_training.parquet"
        if not target_file.exists():
            raise FileNotFoundError("Processed data not found. Run feature builder first.")
            
        logger.info(f"Loading {target_file.name} to Database...")
        df = pd.read_parquet(target_file)
        PostgresLoader().load(df, table_name="historical_training", if_exists="replace")
        logger.info("Database load complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["extract", "build-features", "load", "run-all"], required=True)
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DataPipeline()

    if args.step in ["extract", "run-all"]:
        if not args.start or not args.end:
            raise ValueError("Extraction requires --start and --end dates.")
        pipeline.extract_weather(args.start, args.end)
        pipeline.extract_energy(args.start, args.end)

    if args.step in ["build-features", "run-all"]:
        FeatureBuilder.build_training_set()

    if args.step in ["load", "run-all"]:
        pipeline.load_to_database()

if __name__ == "__main__":
    main()