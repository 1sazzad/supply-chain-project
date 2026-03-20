"""
generate_data.py
Generates realistic supply chain datasets for:
  - carriers.csv
  - ports.csv
  - shipments.csv
  - weather.csv
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/generate_data.log"),
    ],
)
log = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR = "../data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. CARRIERS
# ──────────────────────────────────────────────
CARRIER_DATA = [
    ("C001", "Maersk Line",      "Ocean",  4.5, "Denmark",  700),
    ("C002", "MSC",              "Ocean",  4.3, "Switzerland", 580),
    ("C003", "CMA CGM",          "Ocean",  4.1, "France",   560),
    ("C004", "Evergreen Marine", "Ocean",  3.8, "Taiwan",   210),
    ("C005", "DHL Global",       "Air",    4.7, "Germany",  300),
    ("C006", "FedEx Freight",    "Air",    4.6, "USA",      680),
    ("C007", "DB Schenker",      "Rail",   4.2, "Germany",  120),
    ("C008", "Kuehne+Nagel",     "Multi",  4.4, "Switzerland", 450),
    ("C009", "Hapag-Lloyd",      "Ocean",  4.0, "Germany",  250),
    ("C010", "COSCO Shipping",   "Ocean",  3.6, "China",    380),
]

def build_carriers():
    df = pd.DataFrame(CARRIER_DATA,
        columns=["carrier_id","carrier_name","carrier_type","rating","country","fleet_size"])
    path = f"{RAW_DIR}/carriers.csv"
    df.to_csv(path, index=False)
    log.info(f"carriers.csv saved → {len(df)} rows")
    return df

# ──────────────────────────────────────────────
# 2. PORTS
# ──────────────────────────────────────────────
PORT_DATA = [
    ("P001", "Shanghai",        "China",       "Asia Pacific",  0.92),
    ("P002", "Singapore",       "Singapore",   "Asia Pacific",  0.71),
    ("P003", "Rotterdam",       "Netherlands", "Europe",        0.68),
    ("P004", "Los Angeles",     "USA",         "North America", 0.85),
    ("P005", "Dubai (Jebel Ali)","UAE",        "Middle East",   0.74),
    ("P006", "Hamburg",         "Germany",     "Europe",        0.62),
    ("P007", "Busan",           "South Korea", "Asia Pacific",  0.78),
    ("P008", "Antwerp",         "Belgium",     "Europe",        0.65),
    ("P009", "New York / NJ",   "USA",         "North America", 0.80),
    ("P010", "Colombo",         "Sri Lanka",   "South Asia",    0.55),
    ("P011", "Tanjung Pelepas", "Malaysia",    "Asia Pacific",  0.60),
    ("P012", "Long Beach",      "USA",         "North America", 0.83),
]

def build_ports():
    df = pd.DataFrame(PORT_DATA,
        columns=["port_id","port_name","country","region","congestion_index"])
    path = f"{RAW_DIR}/ports.csv"
    df.to_csv(path, index=False)
    log.info(f"ports.csv saved → {len(df)} rows")
    return df

# ──────────────────────────────────────────────
# 3. SHIPMENTS  (1 000 rows, 2022-2024)
# ──────────────────────────────────────────────
STATUS_MAP = ["Delivered", "In Transit", "Delayed", "Cancelled"]

# Typical transit days between region pairs (min, max)
ROUTE_DAYS = {
    ("Asia Pacific",  "North America"): (14, 22),
    ("Asia Pacific",  "Europe"):        (20, 30),
    ("Asia Pacific",  "Middle East"):   (8,  14),
    ("Asia Pacific",  "South Asia"):    (5,  10),
    ("Europe",        "North America"): (10, 16),
    ("Europe",        "Middle East"):   (8,  14),
    ("Middle East",   "Europe"):        (8,  14),
    ("Middle East",   "North America"): (18, 26),
    ("North America", "Europe"):        (10, 16),
    ("North America", "Asia Pacific"):  (14, 22),
    ("South Asia",    "Europe"):        (18, 26),
    ("South Asia",    "Asia Pacific"):  (5,  10),
}

def route_days(origin_region, dest_region):
    key = (origin_region, dest_region)
    rev = (dest_region, origin_region)
    lo, hi = ROUTE_DAYS.get(key) or ROUTE_DAYS.get(rev) or (10, 20)
    return lo, hi

def build_shipments(ports_df, carriers_df, n=1000):
    records = []
    port_ids   = ports_df["port_id"].tolist()
    region_map = dict(zip(ports_df["port_id"], ports_df["region"]))
    cong_map   = dict(zip(ports_df["port_id"], ports_df["congestion_index"]))
    carrier_ids = carriers_df["carrier_id"].tolist()
    carrier_rat = dict(zip(carriers_df["carrier_id"], carriers_df["rating"]))

    start = datetime(2022, 1, 1)
    end   = datetime(2024, 6, 30)

    for i in range(1, n + 1):
        origin = random.choice(port_ids)
        dest   = random.choice([p for p in port_ids if p != origin])
        carrier = random.choice(carrier_ids)

        # Random departure date
        dep = start + timedelta(days=random.randint(0, (end - start).days))

        # Scheduled transit
        lo, hi = route_days(region_map[origin], region_map[dest])
        sched_days = random.randint(lo, hi)
        sched_arr  = dep + timedelta(days=sched_days)

        # Delay model: congestion + inverse of carrier rating → delay days
        cong  = (cong_map[origin] + cong_map[dest]) / 2
        rat   = carrier_rat[carrier]
        delay_prob = cong * (5 - rat) / 4          # 0..1-ish
        delay_days = 0
        if random.random() < delay_prob:
            delay_days = int(np.random.exponential(scale=3.5)) + 1
            delay_days = min(delay_days, 15)       # cap at 15

        actual_arr = sched_arr + timedelta(days=delay_days)

        # Cost model: base by route + noise + carrier premium
        lo_d, hi_d = route_days(region_map[origin], region_map[dest])
        base_cost  = lo_d * random.uniform(180, 320)   # $/day * days
        cost       = round(base_cost * (1 + (rat - 3.5) * 0.08) + random.gauss(0, 500), 2)
        cost       = max(cost, 400)

        status = "Delivered"
        if delay_days > 0:
            status = "Delayed"
        if dep > datetime(2024, 4, 1):
            status = random.choice(["In Transit", "Delivered"])
        if random.random() < 0.01:
            status = "Cancelled"

        records.append({
            "shipment_id":           f"SHP{i:05d}",
            "origin_port_id":        origin,
            "destination_port_id":   dest,
            "carrier_id":            carrier,
            "departure_date":        dep.date(),
            "scheduled_arrival":     sched_arr.date(),
            "actual_arrival":        actual_arr.date() if status != "Cancelled" else None,
            "cost_usd":              cost,
            "status":                status,
        })

    df = pd.DataFrame(records)
    path = f"{RAW_DIR}/shipments.csv"
    df.to_csv(path, index=False)
    log.info(f"shipments.csv saved → {len(df)} rows | delayed: {(df.status=='Delayed').sum()}")
    return df

# ──────────────────────────────────────────────
# 4. WEATHER  (daily, per port, 2022-2024)
# ──────────────────────────────────────────────
CONDITIONS = ["Clear", "Cloudy", "Rain", "Thunderstorm", "Fog", "Storm"]
COND_WEIGHTS = [0.40, 0.25, 0.18, 0.07, 0.06, 0.04]

def build_weather(ports_df):
    records = []
    start = datetime(2022, 1, 1)
    end   = datetime(2024, 6, 30)
    day_range = (end - start).days + 1
    wid = 1

    for _, port in ports_df.iterrows():
        for d in range(0, day_range, 3):          # every 3rd day to keep size sane
            dt = (start + timedelta(days=d)).date()
            cond = random.choices(CONDITIONS, weights=COND_WEIGHTS)[0]
            if cond in ("Storm", "Thunderstorm"):
                wind  = round(random.uniform(40, 90), 1)
                vis   = random.randint(1, 5)
            elif cond == "Fog":
                wind  = round(random.uniform(5, 20), 1)
                vis   = random.randint(1, 3)
            else:
                wind  = round(random.uniform(5, 35), 1)
                vis   = random.randint(5, 20)

            records.append({
                "weather_id":   wid,
                "port_id":      port["port_id"],
                "weather_date": dt,
                "condition":    cond,
                "wind_speed_kmh": wind,
                "visibility_km":  vis,
            })
            wid += 1

    df = pd.DataFrame(records)
    path = f"{RAW_DIR}/weather.csv"
    df.to_csv(path, index=False)
    log.info(f"weather.csv saved → {len(df)} rows")
    return df

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Dataset generation started ===")
    carriers  = build_carriers()
    ports     = build_ports()
    shipments = build_shipments(ports, carriers, n=1000)
    weather   = build_weather(ports)
    log.info("=== All CSVs written to data/raw/ ===")
    print("\nSummary:")
    print(f"  carriers : {len(carriers)} rows")
    print(f"  ports    : {len(ports)} rows")
    print(f"  shipments: {len(shipments)} rows")
    print(f"  weather  : {len(weather)} rows")
