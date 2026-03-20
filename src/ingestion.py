"""
ingestion.py  —  Step 1 of the ETL pipeline
Reads raw CSVs from data/raw/, validates schema, logs issues.
"""

import pandas as pd
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/ingestion.log"),
    ],
)
log = logging.getLogger(__name__)

RAW_DIR = Path("../data/raw")

SCHEMAS = {
    "carriers":  ["carrier_id","carrier_name","carrier_type","rating","country","fleet_size"],
    "ports":     ["port_id","port_name","country","region","congestion_index"],
    "shipments": ["shipment_id","origin_port_id","destination_port_id","carrier_id",
                  "departure_date","scheduled_arrival","actual_arrival","cost_usd","status"],
    "weather":   ["weather_id","port_id","weather_date","condition","wind_speed_kmh","visibility_km"],
}

def ingest(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        log.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    log.info(f"[{name}] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Schema check
    expected = set(SCHEMAS[name])
    actual   = set(df.columns)
    missing  = expected - actual
    extra    = actual - expected
    if missing:
        log.warning(f"[{name}] Missing columns: {missing}")
    if extra:
        log.info(f"[{name}] Extra columns (will be kept): {extra}")

    # Null counts
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        log.warning(f"[{name}] Nulls found:\n{nulls.to_string()}")
    else:
        log.info(f"[{name}] No nulls found")

    return df

def ingest_all() -> dict:
    log.info("=== Ingestion started ===")
    dfs = {}
    for name in SCHEMAS:
        dfs[name] = ingest(name)
    log.info("=== Ingestion complete ===")
    return dfs

if __name__ == "__main__":
    dfs = ingest_all()
    for name, df in dfs.items():
        print(f"\n{name}:")
        print(df.head(3).to_string(index=False))
