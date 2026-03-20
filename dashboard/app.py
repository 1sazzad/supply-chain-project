"""
dashboard/app.py — Streamlit KPI Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = Path("../data/supply_chain.db")

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="🚢",
    layout="wide",
)

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    master   = pd.read_sql("SELECT * FROM master",             conn)
    kpis     = pd.read_sql("SELECT * FROM v_carrier_kpis",     conn)
    ports    = pd.read_sql("SELECT * FROM v_port_congestion",  conn)
    routes   = pd.read_sql("SELECT * FROM v_route_performance",conn)
    monthly  = pd.read_sql("SELECT * FROM v_monthly_kpis",     conn)
    conn.close()
    master["departure_date"] = pd.to_datetime(master["departure_date"], errors="coerce")
    return master, kpis, ports, routes, monthly

master, kpis_df, ports_df, routes_df, monthly_df = load_data()

# ──────────────────────────────────────────────
# Sidebar filters
# ──────────────────────────────────────────────
st.sidebar.title("🚢 Filters")

years = sorted(master["departure_year"].dropna().unique().astype(int).tolist())
sel_years = st.sidebar.multiselect("Year", years, default=years)

regions = sorted(master["origin_region"].dropna().unique().tolist())
sel_region = st.sidebar.multiselect("Origin Region", regions, default=regions)

df = master[
    master["departure_year"].isin(sel_years) &
    master["origin_region"].isin(sel_region)
]

# ──────────────────────────────────────────────
# KPI strip
# ──────────────────────────────────────────────
st.title("🌐 Global Supply Chain Analytics")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Shipments",    f"{len(df):,}")
c2.metric("On-Time Rate",       f"{df['is_on_time'].mean():.1%}")
c3.metric("Avg Delay (delayed)",f"{df[df.delay_days>0]['delay_days'].mean():.1f} days")
c4.metric("Avg Cost / Shipment",f"${df['cost_usd'].mean():,.0f}")
c5.metric("Avg Transit Time",   f"{df['transit_days'].mean():.1f} days")

st.divider()

# ──────────────────────────────────────────────
# Row 1: Monthly trend + Delay distribution
# ──────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("📅 Monthly On-Time Rate & Shipment Volume")
    monthly_filtered = monthly_df[monthly_df["departure_month"].str[:4].isin(map(str, sel_years))]
    fig = go.Figure()
    fig.add_bar(x=monthly_filtered["departure_month"],
                y=monthly_filtered["total_shipments"],
                name="Shipments", marker_color="#AFA9EC", opacity=0.6)
    fig.add_scatter(x=monthly_filtered["departure_month"],
                    y=monthly_filtered["on_time_pct"],
                    mode="lines+markers", name="On-Time %",
                    yaxis="y2", line=dict(color="#0F6E56", width=2))
    fig.update_layout(
        yaxis=dict(title="Shipments"),
        yaxis2=dict(title="On-Time %", overlaying="y", side="right",
                    range=[0, 100]),
        legend=dict(x=0.01, y=0.99),
        height=300, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("🔔 Delay Distribution")
    delayed = df[df["delay_days"] > 0]["delay_days"]
    fig2 = px.histogram(delayed, nbins=20, color_discrete_sequence=["#D85A30"])
    fig2.update_layout(height=300, margin=dict(t=20, b=20),
                       xaxis_title="Delay Days", yaxis_title="Shipments")
    st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────
# Row 2: Carrier ranking + Port congestion
# ──────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🏆 Carrier Performance Ranking")
    kpi_show = kpis_df[["carrier_name","on_time_pct","avg_delay_days",
                         "avg_cost_usd","performance_score"]].sort_values(
                             "performance_score", ascending=False)
    fig3 = px.bar(kpi_show, x="performance_score", y="carrier_name",
                  orientation="h", color="on_time_pct",
                  color_continuous_scale="teal",
                  labels={"performance_score":"Score","carrier_name":"Carrier",
                          "on_time_pct":"On-Time %"})
    fig3.update_layout(height=350, margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("⚓ Port Congestion Index")
    fig4 = px.scatter(ports_df, x="congestion_index", y="avg_departure_delay",
                      size="departures", color="region",
                      hover_name="port_name", size_max=40,
                      labels={"congestion_index":"Congestion Index",
                              "avg_departure_delay":"Avg Delay (days)",
                              "region":"Region"})
    fig4.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────
# Row 3: Route table + Cost vs Delay
# ──────────────────────────────────────────────
col_e, col_f = st.columns([3, 2])

with col_e:
    st.subheader("🛣️ Top Routes by On-Time Performance")
    st.dataframe(
        routes_df[["origin_port","dest_port","shipment_count",
                   "on_time_pct","avg_delay","avg_cost","avg_transit_days"]]
        .sort_values("on_time_pct", ascending=False)
        .head(15)
        .style.background_gradient(subset=["on_time_pct"], cmap="Greens")
        .format({"avg_cost": "${:,.0f}", "on_time_pct": "{:.1f}%", "avg_delay": "{:.1f}"}),
        use_container_width=True, height=320
    )

with col_f:
    st.subheader("💰 Cost vs Delay")
    fig5 = px.scatter(df.sample(min(300, len(df))), x="delay_days", y="cost_usd",
                      color="carrier_type", opacity=0.6,
                      labels={"delay_days":"Delay Days","cost_usd":"Cost (USD)"})
    fig5.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig5, use_container_width=True)

# ──────────────────────────────────────────────
# What-If Simulator
# ──────────────────────────────────────────────
st.divider()
st.subheader("🔮 What-If: Carrier Switch Simulator")

carriers_list = sorted(master["carrier_name"].dropna().unique().tolist())
c_from = st.selectbox("Switch FROM carrier:", carriers_list, index=9)
c_to   = st.selectbox("Switch TO carrier:",   carriers_list, index=0)

if c_from != c_to:
    sub_from = df[df["carrier_name"] == c_from]
    sub_to   = df[df["carrier_name"] == c_to]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{c_from} on-time rate",    f"{sub_from['is_on_time'].mean():.1%}")
    m2.metric(f"{c_to} on-time rate",      f"{sub_to['is_on_time'].mean():.1%}",
              delta=f"{(sub_to['is_on_time'].mean() - sub_from['is_on_time'].mean()):.1%}")
    m3.metric("Est. cost difference",
              f"${sub_to['cost_usd'].mean() - sub_from['cost_usd'].mean():+,.0f}/shipment")

st.caption("Supply Chain Analytics System — built with Python · SQLite · Streamlit · XGBoost")
