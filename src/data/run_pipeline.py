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
        current = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        while current < end_dt:
            nxt = min(current + pd.DateOffset(months=config.chunk_size_months), end_dt)
            yield current, nxt
            current = nxt

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _process_weather_chunk(self, client: OpenMeteoClient, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Handles cache checking, API fetching, and raw saving for a single weather chunk."""
        chunk_file = config.raw_dir / f"weather_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        
        if chunk_file.exists():
            logger.info(f"Loaded from cache: {chunk_file.name}")
            return pd.read_parquet(chunk_file)
            
        raw_data = client.fetch_historical_weather(
            config.anchor_lats, config.anchor_lons,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )
        df_chunk = WeatherDataTransformer.transform(raw_data, config.location_tags)
        
        if not df_chunk.empty:
            df_chunk.to_parquet(chunk_file, index=False)
            
        time.sleep(config.throttle_sleep_seconds)
        return df_chunk

    def _process_energy_chunk(self, client: EnergyChartsClient, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Handles cache checking, API fetching, and raw saving for a single energy chunk."""
        chunk_file = config.raw_dir / f"energy_prod_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        
        if chunk_file.exists():
            logger.info(f"Loaded from cache: {chunk_file.name}")
            return pd.read_parquet(chunk_file)
            
        raw_data = client.fetch_renewable_power_generation_data(
            country="be", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
        )
        df_chunk = EnergyDataTransformer.transform(raw_data)
        
        if not df_chunk.empty:
            df_chunk.to_parquet(chunk_file, index=False)
            
        time.sleep(config.throttle_sleep_seconds)
        return df_chunk

    def _save_interim(self, chunks: list, filename: str) -> None:
        """Combines chunks and saves them to the interim directory."""
        if not chunks:
            return
            
        interim_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
        interim_df.to_parquet(config.interim_dir / filename, index=False)
        logger.info(f"Successfully staged {filename} to Interim.")

    # -----------------------------------------------------------------------
    # Main Orchestration Methods
    # -----------------------------------------------------------------------

    def extract_weather(self, start_date: str, end_date: str) -> None:
        logger.info("Extracting Weather Data...")
        chunks = []
        
        try:
            with OpenMeteoClient() as client:
                for start, end in self._generate_date_chunks(start_date, end_date):
                    df = self._process_weather_chunk(client, start, end)
                    if not df.empty:
                        chunks.append(df)
        except Exception as e:
            logger.error(f"Weather extraction interrupted: {e}")

        self._save_interim(chunks, "weather.parquet")

    def extract_energy(self, start_date: str, end_date: str) -> None:
        logger.info("Extracting Energy Production Data...")
        chunks = []
        
        try:
            with EnergyChartsClient() as client:
                for start, end in self._generate_date_chunks(start_date, end_date):
                    df = self._process_energy_chunk(client, start, end)
                    if not df.empty:
                        chunks.append(df)
                        
                # Installed Power (Yearly data)
                installed_raw = client.fetch_installed_power(country="be", time_step="yearly")
                installed_df = EnergyDataTransformer.transform(installed_raw)
                if not installed_df.empty:
                    installed_df.to_parquet(config.interim_dir / "installed_energy.parquet", index=False)

        except Exception as e:
            logger.error(f"Energy extraction interrupted: {e}")

        self._save_interim(chunks, "energy_production.parquet")

    def extract_forecast(self, forecast_days: int) -> None:
        """Extracts live forecast data and stages it locally."""
        logger.info(f"=== Extraction Phase: {forecast_days}-Day Forecast ===")
        with OpenMeteoClient() as client:
            raw_forecast = client.fetch_forecast_weather(
                config.anchor_lats, config.anchor_lons, forecast_days=forecast_days
            )
            forecast_df = WeatherDataTransformer.transform(raw_forecast, config.location_tags)

        output_file = config.interim_dir / "forecast.parquet"
        forecast_df.to_parquet(output_file, index=False)
        logger.info(f"Staged forecast dataset to {output_file.name}")

    def load_to_database(self, mode: str) -> None:
        """Loads final data to Postgres depending on the mode."""
        if mode == "train":
            target_file = config.processed_dir / "historical_training.parquet"
            table_name = "historical_training"
        else:
            target_file = config.interim_dir / "forecast.parquet"
            table_name = "daily_forecasts"

        if not target_file.exists():
            raise FileNotFoundError(f"Data not found at {target_file}. Run extraction/feature builder first.")
            
        logger.info(f"Loading {target_file.name} to Database table '{table_name}'...")
        df = pd.read_parquet(target_file)
        PostgresLoader().load(df, table_name=table_name, if_exists="replace")
        logger.info("Database load complete.")


# ---------------------------------------------------------------------------
# CLI & Orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renewable Energy Data Pipeline")
    parser.add_argument("--step", choices=["extract", "build-features", "load", "run-all"], required=True)
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--forecast-days", type=int, default=3, choices=range(1, 17))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pipeline = DataPipeline()

    if args.step in ["extract", "run-all"]:
        if args.mode == "train":
            if not args.start or not args.end:
                raise ValueError("Training mode extraction requires --start and --end dates.")
            pipeline.extract_weather(args.start, args.end)
            pipeline.extract_energy(args.start, args.end)
        else:
            pipeline.extract_forecast(args.forecast_days)

    if args.step in ["build-features", "run-all"]:
        if args.mode == "train":
            FeatureBuilder.build_training_set()
        else:
            logger.info("Feature engineering for inference mode not yet implemented.")

    if args.step in ["load", "run-all"]:
        pipeline.load_to_database(mode=args.mode)

if __name__ == "__main__":
    main()