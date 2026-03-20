"""
models.py — ML Layer
Model 1: Delay Prediction  (XGBoost binary classifier + regressor)
Model 2: Cost Prediction   (XGBoost regressor)
Includes: feature engineering, evaluation, SHAP explainability, What-If simulation
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics          import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, classification_report,
)
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/models.log"),
    ],
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("../data/processed")
MODELS_DIR    = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "carrier_id_enc",
    "origin_port_id_enc",
    "destination_port_id_enc",
    "origin_region_enc",
    "dest_region_enc",
    "carrier_type_enc",
    "rating",
    "origin_congestion",
    "dest_congestion",
    "origin_weather_severity",
    "origin_wind_kmh",
    "departure_month_num",
    "departure_dow",
]

DELAY_TARGET = "delay_days"
COST_TARGET  = "cost_usd"


# ─────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────

def engineer_features(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    encoders = {}

    cat_cols = {
        "carrier_id_enc":           "carrier_id",
        "origin_port_id_enc":       "origin_port_id",
        "destination_port_id_enc":  "destination_port_id",
        "origin_region_enc":        "origin_region",
        "dest_region_enc":          "dest_region",
        "carrier_type_enc":         "carrier_type",
    }
    for new_col, src_col in cat_cols.items():
        le = LabelEncoder()
        df[new_col] = le.fit_transform(df[src_col].astype(str))
        encoders[new_col] = le

    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")
    df["departure_month_num"] = df["departure_date"].dt.month
    df["departure_dow"]       = df["departure_date"].dt.dayofweek

    df["delay_binary"] = (df["delay_days"] > 0).astype(int)

    log.info(f"Features engineered. Shape: {df.shape}")
    return df, encoders


# ─────────────────────────────────────────
# Model 1: Delay prediction
# ─────────────────────────────────────────

def train_delay_model(df: pd.DataFrame):
    log.info("=== Training delay model ===")
    X = df[FEATURES]
    y_bin = df["delay_binary"]    # classifier: will it be delayed?
    y_reg = df["delay_days"]      # regressor:  how many days?

    X_tr, X_te, y_b_tr, y_b_te, y_r_tr, y_r_te = train_test_split(
        X, y_bin, y_reg, test_size=0.2, random_state=42, stratify=y_bin
    )

    # Classifier
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42
    )
    clf.fit(X_tr, y_b_tr)
    y_pred_b = clf.predict(X_te)
    log.info(f"Classifier accuracy: {accuracy_score(y_b_te, y_pred_b):.3f}")
    log.info(f"Classifier F1 (delayed): {f1_score(y_b_te, y_pred_b):.3f}")
    print("\n[Delay Classifier]\n" + classification_report(y_b_te, y_pred_b,
          target_names=["On-Time","Delayed"]))

    # Regressor (only on delayed shipments)
    mask_tr = y_r_tr > 0
    mask_te = y_r_te > 0
    reg = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    reg.fit(X_tr[mask_tr], y_r_tr[mask_tr])
    y_pred_r = reg.predict(X_te[mask_te])
    mae = mean_absolute_error(y_r_te[mask_te], y_pred_r)
    r2  = r2_score(y_r_te[mask_te], y_pred_r)
    log.info(f"Delay regressor MAE: {mae:.2f} days | R²: {r2:.3f}")
    print(f"\n[Delay Regressor] MAE={mae:.2f} days  R²={r2:.3f}")

    # Feature importance
    fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop feature importances (classifier):")
    print(fi.head(8).to_string())

    # Save
    with open(MODELS_DIR / "delay_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / "delay_regressor.pkl", "wb") as f:
        pickle.dump(reg, f)

    return clf, reg


# ─────────────────────────────────────────
# Model 2: Cost prediction
# ─────────────────────────────────────────

def train_cost_model(df: pd.DataFrame):
    log.info("=== Training cost model ===")
    X = df[FEATURES + ["transit_days", "delay_days"]]
    y = df[COST_TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    log.info(f"Cost model  MAE=${mae:.0f}  RMSE=${rmse:.0f}  R²={r2:.3f}")
    print(f"\n[Cost Model] MAE=${mae:.0f}  RMSE=${rmse:.0f}  R²={r2:.3f}")

    with open(MODELS_DIR / "cost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


# ─────────────────────────────────────────
# What-If simulation (Product Thinking 🔥)
# ─────────────────────────────────────────

def whatif_carrier_switch(master: pd.DataFrame, delay_clf, encoders: dict,
                           from_carrier: str, to_carrier: str, n_sample: int = 50):
    """
    Simulate: if we switch carrier X → carrier Y on the same routes,
    what happens to predicted delay probability?
    """
    log.info(f"What-If: switching {from_carrier} → {to_carrier}")
    df_subset = master[master["carrier_id"] == from_carrier].head(n_sample).copy()
    if df_subset.empty:
        print(f"No shipments found for carrier {from_carrier}")
        return

    df_base, _ = engineer_features(df_subset)
    X_base = df_base[FEATURES]
    base_proba = delay_clf.predict_proba(X_base)[:, 1].mean()

    # Swap carrier
    df_swap = df_subset.copy()
    df_swap["carrier_id"]   = to_carrier
    df_swap["carrier_name"] = to_carrier  # simplified
    df_swap_eng, _ = engineer_features(df_swap)

    # Re-encode carrier_id with the original encoder
    le = encoders.get("carrier_id_enc")
    if le:
        # Handle unseen labels gracefully
        classes = list(le.classes_)
        if to_carrier not in classes:
            print(f"Carrier {to_carrier} not in training data — can't simulate.")
            return
        df_swap_eng["carrier_id_enc"] = le.transform(df_swap_eng["carrier_id"].astype(str))

    X_swap = df_swap_eng[FEATURES]
    swap_proba = delay_clf.predict_proba(X_swap)[:, 1].mean()

    delta = swap_proba - base_proba
    print(f"\n=== What-If: {from_carrier} → {to_carrier} ===")
    print(f"  Avg delay probability (current):   {base_proba:.1%}")
    print(f"  Avg delay probability (simulated): {swap_proba:.1%}")
    print(f"  Change: {delta:+.1%}  {'(BETTER ✓)' if delta < 0 else '(WORSE ✗)'}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    master = pd.read_csv(PROCESSED_DIR / "master.csv")
    master, encoders = engineer_features(master)

    clf, reg = train_delay_model(master)
    cost_model = train_cost_model(master)

    # Run a What-If simulation
    raw_master = pd.read_csv(PROCESSED_DIR / "master.csv")
    whatif_carrier_switch(
        raw_master, clf, encoders,
        from_carrier="C010",   # COSCO (lower rated)
        to_carrier="C001",     # Maersk (higher rated)
    )
