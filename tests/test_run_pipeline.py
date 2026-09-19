import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Adjust import to match the actual name of your pipeline script
from src.data.run_pipeline import (
    parse_args, 
    extract_and_transform_historical,
    extract_and_transform_forecast,
    load_to_database,
    main
)

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_weather_df():
    """Provides a dummy transformed weather DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True),
        "temperature_2m_flanders": [5.5, 6.0]
    })

@pytest.fixture
def mock_energy_df():
    """Provides a dummy transformed energy DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True),
        "solar": [100.5, 120.0]
    })

@pytest.fixture
def temp_dirs(tmp_path):
    """Overrides the global directory variables to use a temporary pytest folder."""
    raw_dir = tmp_path / "01_raw"
    interim_dir = tmp_path / "02_interim"
    processed_dir = tmp_path / "03_processed"
    
    with patch("src.data.run_pipeline.RAW_DIR", raw_dir), \
         patch("src.data.run_pipeline.INTERIM_DIR", interim_dir), \
         patch("src.data.run_pipeline.PROCESSED_DIR", processed_dir):
        yield {
            "raw": raw_dir, 
            "interim": interim_dir, 
            "processed": processed_dir
        }


# ==========================================
# TESTS: CLI Arguments
# ==========================================

class TestCommandLineInterface:
    
    @patch('sys.argv', ['pipeline.py', '--mode', 'train', '--step', 'extract', '--start', '2026-01-01', '--end', '2026-02-01'])
    def test_parse_args_valid_train(self):
        """Behavior: Parses valid training arguments correctly."""
        args = parse_args()
        assert args.mode == "train"
        assert args.step == "extract"
        assert args.start == "2026-01-01"

    @patch('sys.argv', ['pipeline.py', '--mode', 'inference', '--step', 'run-all'])
    def test_parse_args_valid_inference(self):
        """Behavior: Parses valid inference arguments (start/end not required)."""
        args = parse_args()
        assert args.mode == "inference"
        assert args.step == "run-all"
        assert args.forecast_days == 3  # Default value


# ==========================================
# TESTS: Extraction & Transformation
# ==========================================

class TestExtractionLogic:

    @patch("src.data.run_pipeline.OpenMeteoClient")
    @patch("src.data.run_pipeline.WeatherDataTransformer.transform")
    def test_extract_forecast(self, mock_weather_transform, mock_meteo_client, tmp_path, mock_weather_df):
        """Behavior: Forecast extraction writes a valid parquet file without hitting APIs."""
        mock_instance = MagicMock()
        mock_meteo_client.return_value.__enter__.return_value = mock_instance
        
        mock_weather_transform.return_value = mock_weather_df
        
        output_dir = tmp_path 
        
        extract_and_transform_forecast(output_dir, forecast_days=5)
        
        mock_instance.fetch_forecast_weather.assert_called_once()
        
        # Verify file was written inside the directory
        saved_file = output_dir / "forecast.parquet"
        assert saved_file.exists()
        
        saved_df = pd.read_parquet(saved_file)
        assert len(saved_df) == 2
        assert "temperature_2m_flanders" in saved_df.columns

    @patch("src.data.run_pipeline.OpenMeteoClient")
    @patch("src.data.run_pipeline.EnergyChartsClient")
    @patch("src.data.run_pipeline.WeatherDataTransformer.transform")
    @patch("src.data.run_pipeline.EnergyDataTransformer.transform")
    def test_extract_historical(
        self, mock_energy_transform, mock_weather_transform, 
        mock_energy_client, mock_meteo_client, 
        tmp_path, mock_weather_df, mock_energy_df
    ):
        """Behavior: Historical extraction orchestrates both clients and saves decoupled files."""
        mock_meteo_client.return_value.__enter__.return_value = MagicMock()
        mock_energy_client.return_value.__enter__.return_value = MagicMock()
        
        mock_weather_transform.return_value = mock_weather_df
        
        # Since transform is called multiple times, we just mock it to return the dummy df every time
        mock_energy_transform.return_value = mock_energy_df 
        
        output_dir = tmp_path
        
        extract_and_transform_historical("2026-01-01", "2026-01-02", output_dir)
        
        # Verify multiple decoupled files exist instead of one merged file
        weather_file = output_dir / "weather.parquet"
        energy_prod_file = output_dir / "energy_production.parquet"
        installed_file = output_dir / "installed_energy.parquet"
        
        assert weather_file.exists()
        assert energy_prod_file.exists()
        assert installed_file.exists()
        
        # Verify contents of the decoupled files
        assert "temperature_2m_flanders" in pd.read_parquet(weather_file).columns
        assert "solar" in pd.read_parquet(energy_prod_file).columns


# ==========================================
# TESTS: Database Loading
# ==========================================

class TestLoadingLogic:

    @patch("src.data.run_pipeline.PostgresLoader")
    def test_load_to_database_success(self, mock_loader, tmp_path, mock_weather_df):
        """Behavior: Loader iterates over a directory of parquet files and loads them."""
        # Create a real parquet file in the temp directory named "weather.parquet"
        valid_file = tmp_path / "weather.parquet"
        mock_weather_df.to_parquet(valid_file, index=False)
        
        mock_instance = MagicMock()
        mock_loader.return_value = mock_instance
        
        load_to_database(tmp_path)
        
        # Verify the loader was called
        mock_instance.load.assert_called_once()
        
        # Assert the dataframe passed to `load()` matches and inferred the right table name
        df_passed = mock_instance.load.call_args[0][0]
        assert len(df_passed) == len(mock_weather_df)
        assert mock_instance.load.call_args[1]["table_name"] == "weather"

    def test_load_to_database_file_not_found(self, tmp_path):
        """Behavior: Crashes fast with FileNotFoundError if extraction hasn't run."""
        missing_dir = tmp_path / "does_not_exist"
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_to_database(missing_dir)


# ==========================================
# TESTS: Main Routing (Orchestration)
# ==========================================

class TestMainOrchestration:

    @patch('sys.argv', ['pipeline.py', '--mode', 'train', '--step', 'extract'])
    def test_main_raises_error_if_train_missing_dates(self, temp_dirs):
        """Behavior: main() enforces logic constraints beyond simple argparse types."""
        with pytest.raises(ValueError, match="requires both --start and --end dates"):
            main()

    @patch('sys.argv', ['pipeline.py', '--mode', 'inference', '--step', 'run-all'])
    @patch("src.data.run_pipeline.extract_and_transform_forecast")
    @patch("src.data.run_pipeline.load_to_database")
    def test_main_run_all_inference(self, mock_load, mock_extract, temp_dirs):
        """Behavior: 'run-all' triggers both extraction and loading functions sequentially."""
        main()
        
        mock_extract.assert_called_once()
        mock_load.assert_called_once()
        
        # We check equality instead of checking the `.parent`
        assert mock_extract.call_args[0][0] == temp_dirs["interim"]
        assert mock_load.call_args[0][0] == temp_dirs["interim"]