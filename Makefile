.PHONY: setup clean run

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Setup complete!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up all cache files!"

train:
	@echo "Starting historical data extraction..."
	python -m src.data.run_pipeline --mode train --start 2025-01-01 --end 2026-01-01

inference:
	@echo "Fetching tomorrow's forecast for ML inference..."
	python -m src.data.run_pipeline --mode inference