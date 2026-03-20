"""
transform.py  —  Step 2 of the ETL pipeline
Cleans, type-casts, derives KPI columns, joins tables.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/transform.log"),
    ],
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Individual table transforms
# ──────────────────────────────────────────────

def transform_carriers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating"]     = pd.to_numeric(df["rating"],     errors="coerce")
    df["fleet_size"] = pd.to_numeric(df["fleet_size"], errors="coerce").astype("Int64")
    df["carrier_name"] = df["carrier_name"].str.strip()
    log.info(f"[carriers] transformed → {len(df)} rows")
    return df


def transform_ports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["congestion_index"] = pd.to_numeric(df["congestion_index"], errors="coerce")
    df["port_name"] = df["port_name"].str.strip()

    # Label congestion severity
    bins   = [0, 0.59, 0.74, 0.84, 1.0]
    labels = ["Low", "Medium", "High", "Critical"]
    df["congestion_level"] = pd.cut(df["congestion_index"], bins=bins, labels=labels)
    log.info(f"[ports] transformed → {len(df)} rows")
    return df


def transform_shipments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse dates
    for col in ["departure_date", "scheduled_arrival", "actual_arrival"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")

    # ── KPI derived columns ──────────────────
    # Delay in days (actual - scheduled)
    df["delay_days"] = (
        (df["actual_arrival"] - df["scheduled_arrival"])
        .dt.days
        .clip(lower=0)        # negative means early → treat as 0
    )

    # On-time flag
    df["is_on_time"] = (df["delay_days"] == 0) & (df["status"] != "Cancelled")

    # Transit time (actual end-to-end)
    df["transit_days"] = (df["actual_arrival"] - df["departure_date"]).dt.days

    # Month / Year for time-series
    df["departure_month"] = df["departure_date"].dt.to_period("M").astype(str)
    df["departure_year"]  = df["departure_date"].dt.year

    # Drop cancelled rows for downstream ML
    df_clean = df[df["status"] != "Cancelled"].copy()

    log.info(f"[shipments] transformed → {len(df_clean)} rows (excl. cancelled)")
    log.info(f"[shipments] on-time rate: {df_clean['is_on_time'].mean():.1%}")
    log.info(f"[shipments] avg delay (delayed only): "
             f"{df_clean[df_clean.delay_days>0]['delay_days'].mean():.2f} days")
    return df_clean


def transform_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weather_date"]    = pd.to_datetime(df["weather_date"], errors="coerce")
    df["wind_speed_kmh"]  = pd.to_numeric(df["wind_speed_kmh"], errors="coerce")
    df["visibility_km"]   = pd.to_numeric(df["visibility_km"],  errors="coerce")

    # Severity score for ML feature (0=clear, 3=storm)
    severity_map = {
        "Clear": 0, "Cloudy": 0, "Rain": 1,
        "Fog": 1, "Thunderstorm": 2, "Storm": 3
    }
    df["weather_severity"] = df["condition"].map(severity_map).fillna(0).astype(int)
    log.info(f"[weather] transformed → {len(df)} rows")
    return df


# ──────────────────────────────────────────────
# Master analytical table (wide join)
# ──────────────────────────────────────────────

def build_master_table(shipments, carriers, ports, weather):
    """
    Join all tables into one flat fact table for analytics and ML.
    """
    df = shipments.copy()

    # Join carrier info
    df = df.merge(
        carriers[["carrier_id","carrier_name","carrier_type","rating","country"]],
        on="carrier_id", how="left",
        suffixes=("", "_carrier")
    ).rename(columns={"country": "carrier_country"})

    # Join origin port
    df = df.merge(
        ports[["port_id","port_name","region","congestion_index","congestion_level"]],
        left_on="origin_port_id", right_on="port_id", how="left"
    ).rename(columns={
        "port_name": "origin_port_name",
        "region":    "origin_region",
        "congestion_index": "origin_congestion",
        "congestion_level": "origin_congestion_level",
    }).drop(columns=["port_id"])

    # Join destination port
    df = df.merge(
        ports[["port_id","port_name","region","congestion_index","congestion_level"]],
        left_on="destination_port_id", right_on="port_id", how="left"
    ).rename(columns={
        "port_name": "dest_port_name",
        "region":    "dest_region",
        "congestion_index": "dest_congestion",
        "congestion_level": "dest_congestion_level",
    }).drop(columns=["port_id"])

    # Join worst weather at origin on departure date
    weather_agg = (
        weather.groupby(["port_id","weather_date"])
        .agg(max_severity=("weather_severity","max"),
             max_wind=("wind_speed_kmh","max"))
        .reset_index()
    )
    df = df.merge(
        weather_agg.rename(columns={
            "port_id":       "origin_port_id",
            "weather_date":  "departure_date",
            "max_severity":  "origin_weather_severity",
            "max_wind":      "origin_wind_kmh",
        }),
        on=["origin_port_id","departure_date"], how="left"
    )
    df["origin_weather_severity"] = df["origin_weather_severity"].fillna(0)
    df["origin_wind_kmh"]         = df["origin_wind_kmh"].fillna(df["origin_wind_kmh"].median())

    # Route label
    df["route"] = df["origin_port_name"] + " → " + df["dest_port_name"]

    log.info(f"[master] built → {len(df)} rows, {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────
# Save processed files
# ──────────────────────────────────────────────

def save_all(dfs: dict):
    for name, df in dfs.items():
        path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info(f"Saved → {path}  ({len(df)} rows)")


def run_transforms(raw_dfs: dict) -> dict:
    log.info("=== Transform started ===")
    carriers  = transform_carriers(raw_dfs["carriers"])
    ports     = transform_ports(raw_dfs["ports"])
    shipments = transform_shipments(raw_dfs["shipments"])
    weather   = transform_weather(raw_dfs["weather"])
    master    = build_master_table(shipments, carriers, ports, weather)

    processed = {
        "carriers":  carriers,
        "ports":     ports,
        "shipments": shipments,
        "weather":   weather,
        "master":    master,
    }
    save_all(processed)
    log.info("=== Transform complete ===")
    return processed


if __name__ == "__main__":
    from ingestion import ingest_all
    raw  = ingest_all()
    proc = run_transforms(raw)
    print("\nMaster table preview:")
    print(proc["master"].head(3).to_string())
