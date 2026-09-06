import pytest
import pandas as pd
import logging
from unittest.mock import patch, call

from src.data.loaders import PostgresLoader 

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_env(monkeypatch):
    """Sets a fake DATABASE_URL so the loader can initialize without a real .env file."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/mockdb")
    # Patch targeting src.data.loaders directly
    with patch("src.data.loaders.load_dotenv"):
        yield

@pytest.fixture
def mock_engine():
    """Mocks the SQLAlchemy engine creation to prevent real database connections."""
    # Patch targeting src.data.loaders directly
    with patch("src.data.loaders.create_engine") as mock:
        yield mock

@pytest.fixture
def loader(mock_env, mock_engine):
    """Provides a safely mocked PostgresLoader instance."""
    return PostgresLoader()

@pytest.fixture
def sample_df():
    """Provides a simple DataFrame for testing."""
    return pd.DataFrame({"timestamp": ["2026-01-01"], "value": [10.5]})

# ==========================================
# TESTS
# ==========================================

class TestPostgresLoaderInitialization:

    def test_raises_error_if_missing_db_url(self, monkeypatch):
        """Behavior: It must fail fast upon initialization if the credentials are missing."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        
        with patch("src.data.loaders.load_dotenv"):
            with pytest.raises(ValueError, match="DATABASE_URL is missing"):
                PostgresLoader()

    def test_creates_engine_with_correct_url(self, mock_env, mock_engine):
        """Behavior: It should pass the connection string to SQLAlchemy correctly."""
        PostgresLoader()
        mock_engine.assert_called_once_with("postgresql://user:pass@localhost:5432/mockdb", pool_pre_ping=True)


class TestPostgresLoaderLoadBehavior:

    @patch.object(pd.DataFrame, "to_sql")
    def test_skips_empty_dataframe(self, mock_to_sql, loader, caplog):
        """Behavior: If the DataFrame is empty, do nothing and log a warning."""
        empty_df = pd.DataFrame()
        
        loader.load(empty_df, table_name="test_table")
        
        mock_to_sql.assert_not_called()
        assert "is empty. Skipping upload" in caplog.text

    @patch.object(pd.DataFrame, "to_sql")
    def test_successful_upload_calls_to_sql_correctly(self, mock_to_sql, loader, sample_df):
        """Behavior: A normal upload should call pandas.to_sql with the correct chunking and engine parameters."""
        loader.load(sample_df, table_name="energy_data", if_exists="append")
        
        mock_to_sql.assert_called_once_with(
            name="energy_data",
            con=loader.engine,
            if_exists="append",
            index=False,
            chunksize=1000
        )


class TestPostgresLoaderRetryLogic:

    @patch("src.data.loaders.time.sleep")
    @patch.object(pd.DataFrame, "to_sql")
    def test_retries_and_succeeds_on_second_attempt(self, mock_to_sql, mock_sleep, loader, sample_df, caplog):
        """
        Behavior: If the database drops the connection on attempt 1, it should sleep, 
        retry, and succeed on attempt 2.
        """
        mock_to_sql.side_effect = [Exception("Connection lost"), None]
        
        # Tell pytest to capture INFO logs during this execution
        import logging
        with caplog.at_level(logging.INFO):
            loader.load(sample_df, table_name="test_table", retries=3)
        
        assert mock_to_sql.call_count == 2
        mock_sleep.assert_called_once_with(2) 
        assert "Successfully loaded 'test_table'" in caplog.text

    @patch("src.data.loaders.time.sleep")
    @patch.object(pd.DataFrame, "to_sql")
    def test_exhausts_retries_and_raises_exception(self, mock_to_sql, mock_sleep, loader, sample_df):
        """Behavior: If all retries fail, it should ultimately raise the exception to crash the pipeline."""
        mock_to_sql.side_effect = Exception("Database is down")
        
        with pytest.raises(Exception, match="Database is down"):
            loader.load(sample_df, table_name="test_table", retries=3)
            
        assert mock_to_sql.call_count == 3
        mock_sleep.assert_has_calls([call(2), call(4)])