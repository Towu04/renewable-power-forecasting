import os
import json
import logging
import argparse
from zipfile import Path
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.data.extractors import EnergyChartsClient, OpenMeteoClient
from src.data.transformers import EnergyDataTransformer, WeatherDataTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"

def setup_folders():
    """Creates the data folders if they don't exist yet."""
    for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        os.makedirs(str(directory), exist_ok=True)

def save_raw_json(data: dict, filename: str):
    """Utility to save the exact API response for backup purposes."""
    filepath = os.path.join(RAW_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved raw data to {filepath}")

def run_historical_training_pipeline(start_date: str, end_date: str, lat: float, lon: float):
    """Pipeline 1: Fetches years of data, saves it, and preps it for ML training."""
    logger.info(f"--- STARTING HISTORICAL PIPELINE ({start_date} to {end_date}) ---")
    
    energy_client = EnergyChartsClient()
    weather_client = OpenMeteoClient()

    # 1. --- PROCESS WEATHER DATA ---
    logger.info("Extracting Historical Weather Data...")
    raw_weather = weather_client.fetch_historical_weather(lat, lon, start_date, end_date)
    save_raw_json(raw_weather, f"weather_historical_{start_date}_{end_date}.json")
    
    clean_weather_df = WeatherDataTransformer.transform(raw_weather)
    weather_csv_path = os.path.join(INTERIM_DIR, "clean_historical_weather.csv")
    clean_weather_df.to_csv(weather_csv_path, index=False)
    logger.info(f"Saved clean weather data to {weather_csv_path}")

    # 2. --- PROCESS ENERGY DATA (With Chunking!) ---
    logger.info("Extracting Historical Energy Data...")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    current_start = start_dt
    
    all_energy_chunks = []
    
    while current_start < end_dt:
        current_end = min(current_start + pd.DateOffset(months=1), end_dt)
        str_start = current_start.strftime("%Y-%m-%d")
        str_end = current_end.strftime("%Y-%m-%d")
        
        # EXTRACT
        raw_energy = energy_client.fetch_renewable_power_generation_data(
            country="be", start=str_start, end=str_end
        )
        save_raw_json(raw_energy, f"energy_raw_be_{str_start}_{str_end}.json")
        
        # TRANSFORM
        chunk_df = EnergyDataTransformer.transform(raw_energy)
        if not chunk_df.empty:
            all_energy_chunks.append(chunk_df)
            
        current_start = current_end 

    if all_energy_chunks:
        clean_energy_df = pd.concat(all_energy_chunks, ignore_index=True)
        clean_energy_df = clean_energy_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        
        energy_csv_path = os.path.join(INTERIM_DIR, "clean_historical_energy.csv")
        clean_energy_df.to_csv(energy_csv_path, index=False)
        logger.info(f"Saved clean energy data to {energy_csv_path}")
        
        # The ultimate ML dataset: Joining Weather and Power together!
        # Both datasets use UTC and the column name "timestamp"
        master_df = pd.merge(clean_energy_df, clean_weather_df, on="timestamp", how="inner")
        
        master_csv_path = os.path.join(PROCESSED_DIR, "ml_training_dataset.csv")
        master_df.to_csv(master_csv_path, index=False)
        logger.info(f"SUCCESS! Master training dataset saved to {master_csv_path}")

def run_daily_inference_pipeline(lat: float, lon: float):
    """Pipeline 2: Fetches tomorrow's weather to predict tomorrow's power."""
    logger.info("--- STARTING DAILY INFERENCE PIPELINE ---")
    
    weather_client = OpenMeteoClient()
    
    logger.info("Extracting Tomorrow's Forecast...")
    raw_forecast = weather_client.fetch_forecast_weather(lat, lon, forecast_days=3)
    
    today_str = datetime.now().strftime("%Y%m%d")
    save_raw_json(raw_forecast, f"weather_forecast_{today_str}.json")
    
    clean_forecast_df = WeatherDataTransformer.transform(raw_forecast)
    forecast_csv_path = os.path.join(INTERIM_DIR, "current_forecast.csv")
    clean_forecast_df.to_csv(forecast_csv_path, index=False)
    
    logger.info(f"Forecast transformed and ready for ML Model at {forecast_csv_path}")

if __name__ == "__main__":
    setup_folders()
    
    parser = argparse.ArgumentParser(description="Renewable Energy Data Pipeline")
    
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Run the historical training or daily inference pipeline")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) for training mode")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD) for training mode")
    parser.add_argument("--lat", type=float, default=51.0, help="Latitude (Defaults to Londerzeel)")
    parser.add_argument("--lon", type=float, default=4.30, help="Longitude (Defaults to Londerzeel)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.start or not args.end:
            parser.error("Training mode requires --start and --end dates!")
        run_historical_training_pipeline(args.start, args.end, args.lat, args.lon)
        
    elif args.mode == "inference":
        run_daily_inference_pipeline(args.lat, args.lon)