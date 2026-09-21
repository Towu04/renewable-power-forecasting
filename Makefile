START_DATE ?= 2014-01-01
END_DATE ?= 2025-01-01
FORECAST_DAYS ?= 3
MODULE ?= src.data.run_pipeline

.PHONY: setup clean train inference test extract-train build-features load-train extract-inference load-inference

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
	@echo "Starting full historical data pipeline (Extract -> Build Features -> Load)..."
	python -m $(MODULE) --mode train --step run-all --start $(START_DATE) --end $(END_DATE)

inference:
	@echo "Fetching $(FORECAST_DAYS)-day forecast for ML inference..."
	python -m $(MODULE) --mode inference --step run-all --forecast-days $(FORECAST_DAYS)

# --- Decoupled Commands (Medallion Architecture) ---

extract-train:
	@echo "Extracting historical data to Raw/Interim..."
	python -m $(MODULE) --mode train --step extract --start $(START_DATE) --end $(END_DATE)

build-features:
	@echo "Building ML features (Interim -> Processed)..."
	python -m $(MODULE) --mode train --step build-features

load-train:
	@echo "Loading Processed historical data to Postgres..."
	python -m $(MODULE) --mode train --step load

# --- Inference ---

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