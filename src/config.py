# src/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class PipelineConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_dir: Path = project_root / "data" / "01_raw"
    interim_dir: Path = project_root / "data" / "02_interim"
    processed_dir: Path = project_root / "data" / "03_processed"
    
    # API Parameters
    anchor_lats: tuple = (51.55, 51.25, 50.85, 50.25)
    anchor_lons: tuple = (2.90, 3.20, 4.35, 5.50)
    location_tags: tuple = ("north_sea", "coast", "flanders", "wallonia")
    
    chunk_size_months: int = 12
    throttle_sleep_seconds: float = 1.5

config = PipelineConfig()