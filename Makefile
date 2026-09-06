START_DATE ?= 2025-01-01
END_DATE ?= 2026-01-01
FORECAST_DAYS ?= 3
MODULE ?= src.data.run_pipeline

.PHONY: setup clean train inference test extract-train load-train extract-inference load-inference

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Setup complete!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up all cache files!"

# --- Combined Pipeline Commands ---

train:
	@echo "Starting full historical data pipeline (Extract & Load)..."
	python -m $(MODULE) --mode train --step run-all --start $(START_DATE) --end $(END_DATE)

inference:
	@echo "Fetching $(FORECAST_DAYS)-day forecast for ML inference..."
	python -m $(MODULE) --mode inference --step run-all --forecast-days $(FORECAST_DAYS)

# --- Decoupled Commands ---

extract-train:
	@echo "Extracting historical data only..."
	python -m $(MODULE) --mode train --step extract --start $(START_DATE) --end $(END_DATE)

load-train:
	@echo "Loading staged historical data to Postgres..."
	python -m $(MODULE) --mode train --step load

extract-inference:
	@echo "Extracting $(FORECAST_DAYS)-day forecast only..."
	python -m $(MODULE) --mode inference --step extract --forecast-days $(FORECAST_DAYS)

load-inference:
	@echo "Loading staged forecast data to Postgres..."
	python -m $(MODULE) --mode inference --step load

# --- Testing ---

test:
	@echo "Running tests..."
	python -m pytest