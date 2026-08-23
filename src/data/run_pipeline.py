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


def fetch_and_clean_weather(start_date: str, end_date: str, lats: list[float], lons: list[float]) -> pd.DataFrame:
    """Fetches and cleans historical weather data."""
    logger.info("1/3 Fetching weather data...")
    raw = OpenMeteoClient().fetch_historical_weather(lats, lons, start_date, end_date)
    return WeatherDataTransformer.transform(raw, LOCATION_TAGS)


def fetch_and_clean_energy(start_date: str, end_date: str, country: str = "be") -> pd.DataFrame:
    """Fetches and cleans historical energy data in monthly chunks."""
    logger.info("2/3 Fetching energy generation data...")
    client = EnergyChartsClient()
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    
    chunks = []
    current = start_dt
    while current < end_dt:
        nxt = min(current + pd.DateOffset(months=1), end_dt)
        raw = client.fetch_renewable_power_generation_data(
            country=country,
            start=current.strftime("%Y-%m-%d"),
            end=nxt.strftime("%Y-%m-%d"),
        )
        df_chunk = EnergyDataTransformer.transform(raw)
        if not df_chunk.empty:
            chunks.append(df_chunk)
        current = nxt

    # Combine and resample 15-min data to hourly averages
    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp").resample("1h").mean().reset_index()
    return df


def run_training_pipeline(start_date: str, end_date: str, lats: list[float], lons: list[float]) -> None:
    """Extracts historical data, merges it, and saves locally + to database."""
    logger.info(f"=== Starting Training Data Pipeline ({start_date} to {end_date}) ===")

    # 1. Extract & Transform
    weather_df = fetch_and_clean_weather(start_date, end_date, lats, lons)
    energy_df = fetch_and_clean_energy(start_date, end_date)

    # 2. Merge (Hourly UTC alignment)
    logger.info("3/3 Merging datasets...")
    master_df = pd.merge(energy_df, weather_df, on="timestamp", how="inner")

    # 3. Save Local Artifact (Backup)
    output_path = PROCESSED_DIR / "ml_training_dataset.csv"
    master_df.to_csv(output_path, index=False)
    logger.info(f"Saved local dataset ({len(master_df)} rows) to {output_path}")

    # 4. Load to Database
    PostgresLoader().load(master_df, table_name="historical_training", if_exists="replace")


def run_inference_pipeline(lats: List[float], lons: List[float]) -> None:
    """Fetches current forecast and saves for model prediction."""
    logger.info("=== Starting Daily Forecast Pipeline ===")

    raw_forecast = OpenMeteoClient().fetch_forecast_weather(lats, lons, forecast_days=3)
    forecast_df = WeatherDataTransformer.transform(raw_forecast)

    # Save locally and push to database
    output_path = INTERIM_DIR / "current_forecast.csv"
    forecast_df.to_csv(output_path, index=False)
    PostgresLoader().load(forecast_df, table_name="daily_forecasts", if_exists="replace")


# ---------------------------------------------------------------------------
# CLI & Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments cleanly."""
    parser = argparse.ArgumentParser(description="Renewable Energy Data Pipeline")
    parser.add_argument("--mode", choices=["train", "inference", "load_only"], required=True)
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    """Main application entry point."""
    # Ensure folders exist
    for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    args = parse_args()

    if args.mode == "train":
        if not args.start or not args.end:
            raise ValueError("Training mode requires both --start and --end dates.")
        run_training_pipeline(args.start, args.end, ANCHOR_LATS, ANCHOR_LONS)

    elif args.mode == "inference":
        run_inference_pipeline(ANCHOR_LATS, ANCHOR_LONS)

    elif args.mode == "load_only":
        csv_path = PROCESSED_DIR / "ml_training_dataset.csv"
        df = pd.read_csv(csv_path)
        PostgresLoader().load(df, table_name="historical_training", if_exists="replace")


if __name__ == "__main__":
    main()