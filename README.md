# 🚢 Global Supply Chain Analytics & Shipment Optimization System

> End-to-end data engineering + analytics + ML system inspired by Flexport's operational stack.
> Built to demonstrate: SQL · ETL pipelines · XGBoost · Streamlit · product thinking.

---

## 📌 Business Problem

> "How can we reduce shipment delays, optimize cost, and improve delivery reliability using data?"

Global freight companies lose millions annually to delays and inefficient carrier allocation. This project builds a data system that answers:

- Which carriers consistently underperform?
- Which routes carry the highest delay risk?
- How does port congestion affect costs?
- What happens if we switch carrier on a given route?

---

## 🗂️ Dataset Design

Relational data model with 4 tables, 1 000+ shipment records spanning 2022–2024.

| Table | Rows | Description |
|---|---|---|
| `shipments` | 1 000 | Core fact table: routes, costs, dates, delays |
| `carriers` | 10 | Carrier metadata: type, rating, fleet size |
| `ports` | 12 | Port metadata: region, congestion index |
| `weather` | ~3 000 | Historical weather conditions per port |

**Key design decision:** delay is modelled as a function of carrier rating + port congestion + weather — making it learnable by ML.

---

## 📊 KPIs Defined

| KPI | Formula |
|---|---|
| On-Time Delivery Rate | `shipments with delay_days=0 / total` |
| Average Delay (delayed) | `mean(delay_days) WHERE delay_days > 0` |
| Carrier Performance Score | `rating×20 − avg_delay×5 + on_time_pct` |
| Port Congestion Index | Pre-assigned (0–1), correlated with delays in data |
| Cost per Shipment | `mean(cost_usd)` by carrier / route |

---

## 🏗️ Project Structure

```
supply_chain_project/
│
├── data/
│   ├── raw/              ← CSVs from generate_data.py
│   ├── processed/        ← Cleaned, enriched DataFrames
│   └── supply_chain.db   ← SQLite database
│
├── src/
│   ├── generate_data.py  ← Realistic dataset generator
│   ├── ingestion.py      ← ETL Step 1: load & validate
│   ├── transform.py      ← ETL Step 2: clean & derive KPIs
│   ├── load.py           ← ETL Step 3: SQLite + SQL views
│   └── models.py         ← XGBoost delay + cost models
│
├── dashboard/
│   └── app.py            ← Streamlit KPI dashboard
│
├── models/               ← Saved .pkl model files
├── logs/                 ← Pipeline logs
└── README.md
```

---

## 🧭 System Architecture & Design (Detailed)

For a full, user-friendly architecture and design walkthrough (with visual diagrams), see:

- [`docs/system_architecture_and_design.md`](docs/system_architecture_and_design.md)

---

## ⚙️ ETL Pipeline

```
Raw CSV (data/raw/)
     ↓  ingestion.py    [schema check, null audit, logging]
     ↓  transform.py    [type casting, KPI derivation, master join]
     ↓  load.py         [SQLite, SQL views, window functions]
     ↓
SQLite DB + Processed CSVs
```

---

## 🔍 SQL Layer Examples

**Carrier KPI view (window function):**
```sql
SELECT
    carrier_id,
    avg_delay_days,
    on_time_pct,
    RANK() OVER (ORDER BY performance_score DESC) AS perf_rank
FROM v_carrier_kpis;
```

**Rolling 3-month average delay:**
```sql
SELECT
    departure_month,
    AVG(avg_delay_days) OVER (
        ORDER BY departure_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_avg_delay
FROM v_monthly_kpis;
```

---

## 🤖 Machine Learning

### Model 1: Delay Prediction
- **Binary classifier** — will this shipment be delayed? (XGBoost)
- **Regressor** — how many days of delay? (XGBoost, trained on delayed-only)
- **Top features:** carrier rating, origin congestion, weather severity, destination congestion

### Model 2: Cost Prediction
- XGBoost regressor
- Features: route, carrier type, congestion, transit days

### Results (sample run)
| Model | Metric | Value |
|---|---|---|
| Delay classifier | F1 (delayed class) | ~0.72 |
| Delay regressor | MAE | ~1.4 days |
| Cost model | MAE | ~$380 |
| Cost model | R² | ~0.87 |

---

## 🔮 What-If Simulator (Product Thinking)

> "If we switch from COSCO → Maersk on Asia–North America routes, how much does delay probability drop?"

The `whatif_carrier_switch()` function re-scores existing shipments under an alternative carrier using the trained classifier — giving a data-backed answer to a real procurement question.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy sqlalchemy xgboost scikit-learn streamlit plotly

# 2. Generate data
cd src && python generate_data.py

# 3. Run full ETL
python ingestion.py
python transform.py
python load.py

# 4. Train models
python models.py

# 5. Launch dashboard
cd ../dashboard && streamlit run app.py
```

---

## 💼 Skills Demonstrated

| Area | Tools / Techniques |
|---|---|
| Data modelling | Relational schema, FK design, ER diagram |
| ETL pipeline | pandas, SQLAlchemy, logging, error handling |
| SQL | JOINs, window functions, aggregations, views |
| Machine learning | XGBoost, feature engineering, model evaluation |
| Product thinking | What-If simulator, KPI framework |
| Dashboard | Streamlit, Plotly, interactive filters |
| Software engineering | Modular code, logging, project structure |
