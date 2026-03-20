"""
load.py  —  Step 3 of the ETL pipeline
Loads processed DataFrames into SQLite via SQLAlchemy.
Also creates analytical SQL views (KPIs, carrier ranking, port congestion).
"""

import pandas as pd
import logging
from pathlib import Path
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/load.log"),
    ],
)
log = logging.getLogger(__name__)

DB_PATH = Path("../data/supply_chain.db")

VIEWS = {
    "v_carrier_kpis": """
        CREATE VIEW IF NOT EXISTS v_carrier_kpis AS
        SELECT
            s.carrier_id,
            c.carrier_name,
            c.carrier_type,
            c.rating,
            COUNT(*)                                        AS total_shipments,
            ROUND(AVG(s.delay_days), 2)                     AS avg_delay_days,
            ROUND(
                100.0 * SUM(CASE WHEN s.is_on_time THEN 1 ELSE 0 END)
                / COUNT(*), 1
            )                                               AS on_time_pct,
            ROUND(AVG(s.cost_usd), 2)                       AS avg_cost_usd,
            ROUND(AVG(s.transit_days), 1)                   AS avg_transit_days,
            -- Composite performance score (higher = better)
            ROUND(
                (c.rating * 20)
                - (AVG(s.delay_days) * 5)
                + (SUM(CASE WHEN s.is_on_time THEN 1 ELSE 0 END) * 100.0 / COUNT(*))
            , 1)                                            AS performance_score
        FROM shipments s
        JOIN carriers c ON s.carrier_id = c.carrier_id
        GROUP BY s.carrier_id
        ORDER BY performance_score DESC
    """,
    "v_port_congestion": """
        CREATE VIEW IF NOT EXISTS v_port_congestion AS
        SELECT
            p.port_id,
            p.port_name,
            p.region,
            p.congestion_index,
            p.congestion_level,
            COUNT(s.shipment_id)                            AS departures,
            ROUND(AVG(s.delay_days), 2)                     AS avg_departure_delay,
            ROUND(AVG(s.cost_usd), 2)                       AS avg_cost_origin
        FROM ports p
        LEFT JOIN shipments s ON p.port_id = s.origin_port_id
        GROUP BY p.port_id
        ORDER BY p.congestion_index DESC
    """,
    "v_route_performance": """
        CREATE VIEW IF NOT EXISTS v_route_performance AS
        SELECT
            s.origin_port_id,
            op.port_name                                    AS origin_port,
            s.destination_port_id,
            dp.port_name                                    AS dest_port,
            op.region                                       AS origin_region,
            dp.region                                       AS dest_region,
            COUNT(*)                                        AS shipment_count,
            ROUND(AVG(s.delay_days), 2)                     AS avg_delay,
            ROUND(AVG(s.cost_usd), 2)                       AS avg_cost,
            ROUND(AVG(s.transit_days), 1)                   AS avg_transit_days,
            ROUND(
                100.0 * SUM(CASE WHEN s.is_on_time THEN 1 ELSE 0 END)
                / COUNT(*), 1
            )                                               AS on_time_pct
        FROM shipments s
        JOIN ports op ON s.origin_port_id      = op.port_id
        JOIN ports dp ON s.destination_port_id = dp.port_id
        GROUP BY s.origin_port_id, s.destination_port_id
        HAVING shipment_count >= 5
        ORDER BY avg_delay ASC
    """,
    "v_monthly_kpis": """
        CREATE VIEW IF NOT EXISTS v_monthly_kpis AS
        SELECT
            departure_month,
            COUNT(*)                                        AS total_shipments,
            ROUND(AVG(delay_days), 2)                       AS avg_delay_days,
            ROUND(AVG(cost_usd), 2)                         AS avg_cost_usd,
            ROUND(
                100.0 * SUM(CASE WHEN is_on_time THEN 1 ELSE 0 END)
                / COUNT(*), 1
            )                                               AS on_time_pct,
            SUM(cost_usd)                                   AS total_revenue
        FROM shipments
        GROUP BY departure_month
        ORDER BY departure_month
    """,
}

WINDOW_QUERIES = {
    "carrier_delay_rank": """
        SELECT
            carrier_id,
            carrier_name,
            avg_delay_days,
            on_time_pct,
            performance_score,
            RANK() OVER (ORDER BY performance_score DESC) AS perf_rank,
            RANK() OVER (ORDER BY avg_delay_days ASC)     AS delay_rank
        FROM v_carrier_kpis
    """,
    "monthly_delay_rolling_avg": """
        SELECT
            departure_month,
            avg_delay_days,
            ROUND(
                AVG(avg_delay_days) OVER (
                    ORDER BY departure_month
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ), 2
            ) AS rolling_3m_avg_delay
        FROM v_monthly_kpis
    """,
}


def load_to_sqlite(processed: dict) -> None:
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    log.info(f"Connected to SQLite at {DB_PATH}")

    table_order = ["carriers", "ports", "shipments", "weather", "master"]
    for name in table_order:
        if name not in processed:
            continue
        df = processed[name]
        df.to_sql(name, engine, if_exists="replace", index=False)
        log.info(f"Table '{name}' loaded → {len(df)} rows")

    # Create views
    with engine.connect() as conn:
        for view_name, ddl in VIEWS.items():
            conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
            conn.execute(text(ddl))
            log.info(f"View '{view_name}' created")
        conn.commit()

    # Quick sanity: run each window query and log row count
    with engine.connect() as conn:
        for qname, sql in WINDOW_QUERIES.items():
            result = pd.read_sql(sql, conn)
            log.info(f"Window query '{qname}' → {len(result)} rows")
            print(f"\n--- {qname} ---")
            print(result.to_string(index=False))

    log.info("=== Load complete ===")


def run_load(processed: dict):
    log.info("=== Load started ===")
    load_to_sqlite(processed)


if __name__ == "__main__":
    from ingestion import ingest_all
    from transform  import run_transforms

    raw  = ingest_all()
    proc = run_transforms(raw)
    run_load(proc)
    print(f"\nDatabase written to: {DB_PATH.resolve()}")
