import logging
import pandas as pd
from src.config import config

logger = logging.getLogger("FeatureBuilder")

class FeatureBuilder:
    """Handles heavy transformations: joins, resampling, and ML feature generation."""
    
    @staticmethod
    def build_training_set() -> None:
        logger.info("=== Building Features (Interim -> Processed) ===")
        
        weather_file = config.interim_dir / "weather.parquet"
        energy_file = config.interim_dir / "energy_production.parquet"
        
        if not weather_file.exists() or not energy_file.exists():
            raise FileNotFoundError("Interim data missing. Run `make extract-train` first.")
            
        logger.info("Loading interim datasets...")
        weather_df = pd.read_parquet(weather_file)
        energy_df = pd.read_parquet(energy_file)

        logger.info("Joining weather and energy data on timestamp...")
        master_df = pd.merge(weather_df, energy_df, on="timestamp", how="inner")

        logger.info("Interpolating missing values...")
        master_df = master_df.set_index("timestamp")
        master_df = master_df.interpolate(method="linear", limit=3) # Only bridge small gaps (up to 3 hours)
        master_df = master_df.dropna() # Drop any remaining NaNs at the very edges
        master_df = master_df.reset_index()

        # -----------------------------------------------------------
        # TODO: Advanced ML Feature Engineering
        # -----------------------------------------------------------

        output_path = config.processed_dir / "historical_training.parquet"
        master_df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed training set: {len(master_df)} rows to {output_path.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    FeatureBuilder.build_training_set()