"""Module AI Prediksi Jam Sibuk (Busy Hour Forecasting Engine).

Menggunakan ensemble ML (Random Forest, Gradient Boosting, Ridge Regression)
untuk memprediksi:
- Jam-jam sibuk per hari untuk 14 hari ke depan
- Prediksi jumlah transaksi per jam
- Prediksi produk apa saja yang akan terjual per jam
- Revenue forecast per jam
- Confidence score & model accuracy metrics
"""

from datetime import datetime, timedelta
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

BUSY_LEVEL_MAP = {
    "PEAK":   {"level": "PEAK",   "point": 4, "emoji": "[!!!]"},
    "HIGH":   {"level": "HIGH",   "point": 3, "emoji": "[!!]"},
    "MEDIUM": {"level": "MEDIUM", "point": 2, "emoji": "[!]"},
    "LOW":    {"level": "LOW",    "point": 1, "emoji": "[~]"},
    "CLOSED": {"level": "CLOSED", "point": 0, "emoji": "[-]"},
}

OPERATING_HOURS = list(range(7, 21))  # 07:00 - 20:00


# ─── Helper: Parse & Build DataFrames ────────────────────────────────────────

def normalize_transactions(raw_data: dict | list) -> list[dict]:
    if isinstance(raw_data, dict):
        return raw_data.get("data", [])
    return raw_data


def _parse_hour(dt_str: str) -> int:
    """Extract hour from datetime string."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt).hour
        except (ValueError, TypeError):
            continue
    return 12  # fallback


def build_hourly_dataframe(transactions: list[dict]) -> pd.DataFrame:
    """Build hourly transaction DataFrame from raw data. SALE only."""
    records = []
    for trx in transactions:
        if trx.get("trx_type") != "SALE":
            continue
        trx_date = trx["trx_date"]
        paid_at = trx.get("paid_at", trx.get("created_at", ""))
        hour = _parse_hour(paid_at)
        total = float(trx.get("total_amount", "0").replace(",", ""))
        item_count = sum(it.get("quantity", 0) for it in trx.get("items", []))
        product_ids = [it["product_id"] for it in trx.get("items", [])]

        records.append({
            "date": pd.to_datetime(trx_date),
            "hour": hour,
            "total_amount": total,
            "item_count": item_count,
            "trx_count": 1,
            "product_ids": product_ids,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def build_product_hour_matrix(transactions: list[dict]) -> pd.DataFrame:
    """Build product-level hourly sales matrix."""
    records = []
    for trx in transactions:
        if trx.get("trx_type") != "SALE":
            continue
        paid_at = trx.get("paid_at", trx.get("created_at", ""))
        hour = _parse_hour(paid_at)
        trx_date = trx["trx_date"]
        for item in trx.get("items", []):
            pid = item["product_id"]
            qty = item.get("quantity", 0)
            pname = "Unknown"
            pprice = 0.0
            prod = item.get("product")
            if prod and isinstance(prod, dict):
                pname = prod.get("name", pname)
                pprice = float(prod.get("price", "0"))
            records.append({
                "date": pd.to_datetime(trx_date),
                "hour": hour,
                "product_id": pid,
                "product_name": pname,
                "product_price": pprice,
                "quantity": qty,
            })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _extract_product_catalog(transactions: list[dict]) -> dict:
    """Extract unique product info from transactions."""
    catalog = {}
    for trx in transactions:
        for item in trx.get("items", []):
            pid = item["product_id"]
            prod = item.get("product")
            if prod and isinstance(prod, dict) and pid not in catalog:
                catalog[pid] = {
                    "id": pid,
                    "name": prod.get("name", f"Product #{pid}"),
                    "price": float(prod.get("price", "0")),
                }
    return catalog


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (date, hour) level and engineer features."""
    agg = df.groupby(["date", "hour"]).agg(
        trx_count=("trx_count", "sum"),
        total_amount=("total_amount", "sum"),
        item_count=("item_count", "sum"),
    ).reset_index()

    # Fill missing hours for each date
    dates = agg["date"].unique()
    full_rows = []
    for d in dates:
        for h in OPERATING_HOURS:
            full_rows.append({"date": d, "hour": h})
    full = pd.DataFrame(full_rows)
    agg = full.merge(agg, on=["date", "hour"], how="left").fillna(0)

    # Features
    agg["day_of_week"] = pd.to_datetime(agg["date"]).dt.dayofweek
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(int)
    agg["day_index"] = (pd.to_datetime(agg["date"]) - pd.to_datetime(agg["date"]).min()).dt.days
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24)
    agg["is_lunch"] = ((agg["hour"] >= 11) & (agg["hour"] <= 13)).astype(int)
    agg["is_morning"] = ((agg["hour"] >= 8) & (agg["hour"] <= 10)).astype(int)
    agg["is_evening"] = ((agg["hour"] >= 16) & (agg["hour"] <= 19)).astype(int)

    # Rolling average per hour (across days)
    agg = agg.sort_values(["hour", "date"])
    agg["rolling_avg_trx"] = agg.groupby("hour")["trx_count"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    agg["rolling_avg_revenue"] = agg.groupby("hour")["total_amount"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    return agg.sort_values(["date", "hour"]).reset_index(drop=True)


FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "day_index",
    "hour_sin", "hour_cos", "is_lunch", "is_morning", "is_evening",
    "rolling_avg_trx", "rolling_avg_revenue",
]


# ─── Ensemble Model ─────────────────────────────────────────────────────────

class BusyHourEnsemble:
    """Ensemble of RF + GBR + Ridge for transaction count prediction."""

    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.gbr = GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)
        self.ridge = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.weights = [0.45, 0.40, 0.15]  # RF, GBR, Ridge
        self.is_fitted = False
        self.metrics = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.gbr.fit(X_scaled, y)
        self.ridge.fit(X_scaled, y)
        self.is_fitted = True
        self._compute_metrics(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        X_scaled = self.scaler.transform(X)
        p_rf = self.rf.predict(X_scaled)
        p_gbr = self.gbr.predict(X_scaled)
        p_ridge = self.ridge.predict(X_scaled)
        ensemble = (self.weights[0] * p_rf +
                    self.weights[1] * p_gbr +
                    self.weights[2] * p_ridge)
        return np.maximum(0, ensemble)

    def _compute_metrics(self, X: np.ndarray, y: np.ndarray):
        """Cross-validated accuracy metrics."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.predict(X)

        # In-sample metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred) if len(y) > 1 else 0.0
        mape = np.mean(np.abs((y - y_pred) / np.where(y == 0, 1, y))) * 100

        # Cross-validation
        n_splits = min(3, max(2, len(y) // 5))
        cv_scores = {}
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for name, model in [("RF", self.rf), ("GBR", self.gbr), ("Ridge", self.ridge)]:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="r2")
                cv_scores[name] = {"mean_r2": round(float(np.mean(scores)), 4),
                                   "std_r2": round(float(np.std(scores)), 4)}
        except Exception:
            cv_scores = {"note": "Insufficient data for cross-validation"}

        accuracy_pct = max(0, min(100, (1 - mape / 100) * 100))

        self.metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "mape_percent": round(mape, 2),
            "accuracy_percent": round(accuracy_pct, 2),
            "training_samples": len(y),
            "cross_validation": cv_scores,
            "ensemble_weights": {"RF": self.weights[0], "GBR": self.weights[1], "Ridge": self.weights[2]},
            "feature_importance_rf": {},
        }

        # Feature importance from RF
        importances = self.rf.feature_importances_
        for i, col in enumerate(FEATURE_COLS):
            if i < len(importances):
                self.metrics["feature_importance_rf"][col] = round(float(importances[i]), 4)


# ─── Revenue Model ───────────────────────────────────────────────────────────

class RevenueModel:
    """Simpler model for revenue prediction per hour."""

    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=60, max_depth=4, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        return np.maximum(0, self.model.predict(self.scaler.transform(X)))


# ─── Product Probability Model ───────────────────────────────────────────────

def build_product_hour_probabilities(prod_df: pd.DataFrame, catalog: dict) -> dict:
    """
    Calculate P(product sold | hour, day_of_week) and avg qty.
    Returns: {product_id: {hour: {dow: {prob, avg_qty, name, price}}}}
    """
    if prod_df.empty:
        return {}

    prod_df = prod_df.copy()
    prod_df["day_of_week"] = pd.to_datetime(prod_df["date"]).dt.dayofweek

    # Count occurrences per (product, hour, dow)
    grp = prod_df.groupby(["product_id", "hour", "day_of_week"]).agg(
        total_qty=("quantity", "sum"),
        occurrence=("quantity", "count"),
    ).reset_index()

    # Total transactions per (hour, dow)
    trx_per_hd = prod_df.drop_duplicates(subset=["date", "hour"]).groupby(
        ["hour", "day_of_week"]
    ).size().reset_index(name="total_slots")

    grp = grp.merge(trx_per_hd, on=["hour", "day_of_week"], how="left")
    grp["probability"] = (grp["occurrence"] / grp["total_slots"]).clip(0, 1)
    grp["avg_qty"] = grp["total_qty"] / grp["occurrence"]

    result = {}
    for _, row in grp.iterrows():
        pid = int(row["product_id"])
        h = int(row["hour"])
        dow = int(row["day_of_week"])
        if pid not in result:
            result[pid] = {}
        if h not in result[pid]:
            result[pid][h] = {}
        info = catalog.get(pid, {"name": f"Product #{pid}", "price": 0})
        result[pid][h][dow] = {
            "probability": round(float(row["probability"]), 3),
            "avg_qty": round(float(row["avg_qty"]), 1),
            "product_name": info["name"],
            "product_price": info["price"],
        }
    return result


# ─── Busy Level Classification ───────────────────────────────────────────────

def classify_busy_level(trx_count: float, percentiles: dict) -> dict:
    """Classify busyness based on predicted trx count vs historical percentiles."""
    if trx_count <= 0:
        return BUSY_LEVEL_MAP["CLOSED"]
    if trx_count >= percentiles.get("p90", 2):
        return BUSY_LEVEL_MAP["PEAK"]
    if trx_count >= percentiles.get("p70", 1.5):
        return BUSY_LEVEL_MAP["HIGH"]
    if trx_count >= percentiles.get("p40", 0.8):
        return BUSY_LEVEL_MAP["MEDIUM"]
    return BUSY_LEVEL_MAP["LOW"]


# ─── Main Entry Point ────────────────────────────────────────────────────────

def analyze_busy_hours(
    transactions: list[dict],
    forecast_days: int = 14,
) -> dict:
    """
    Main entry point: Full busy hour analysis & prediction.

    Returns complex analysis including:
    - 14-day hourly forecast with busy levels
    - Per-product predictions per hour
    - Revenue forecasts
    - Historical pattern analysis
    - Model accuracy metrics
    - Peak hour rankings
    """
    print("\n" + "=" * 70)
    print("[BUSY HOUR] PREDICTION ENGINE - Starting Analysis")
    print("=" * 70)

    # 1. Parse data
    hourly_df = build_hourly_dataframe(transactions)
    if hourly_df.empty:
        return {"error": "Tidak ada data transaksi SALE."}

    prod_df = build_product_hour_matrix(transactions)
    catalog = _extract_product_catalog(transactions)

    print(f"[DATA] Loaded: {len(hourly_df)} sale transactions")
    print(f"[PRODUCTS] Found: {len(catalog)}")

    # 2. Feature engineering
    features_df = build_hourly_features(hourly_df)
    X = features_df[FEATURE_COLS].values
    y_trx = features_df["trx_count"].values
    y_rev = features_df["total_amount"].values

    # 3. Train models
    trx_model = BusyHourEnsemble()
    trx_model.fit(X, y_trx)

    rev_model = RevenueModel()
    rev_model.fit(X, y_rev)

    print(f"[MODEL] Trained | Accuracy: {trx_model.metrics.get('accuracy_percent', 0)}%")
    print(f"   MAE={trx_model.metrics.get('mae', 0)}, RMSE={trx_model.metrics.get('rmse', 0)}, R2={trx_model.metrics.get('r2_score', 0)}")

    # 4. Historical pattern analysis
    hist_hourly = features_df.groupby("hour").agg(
        avg_trx=("trx_count", "mean"),
        max_trx=("trx_count", "max"),
        avg_revenue=("total_amount", "mean"),
    ).reset_index()

    percentiles = {
        "p40": float(np.percentile(y_trx[y_trx > 0], 40)) if np.any(y_trx > 0) else 0.5,
        "p70": float(np.percentile(y_trx[y_trx > 0], 70)) if np.any(y_trx > 0) else 1.0,
        "p90": float(np.percentile(y_trx[y_trx > 0], 90)) if np.any(y_trx > 0) else 1.5,
    }

    # Historical peak hours
    historical_patterns = []
    for _, row in hist_hourly.iterrows():
        bl = classify_busy_level(row["avg_trx"], percentiles)
        historical_patterns.append({
            "hour": f"{int(row['hour']):02d}:00",
            "avg_transactions": round(row["avg_trx"], 2),
            "max_transactions": int(row["max_trx"]),
            "avg_revenue": round(row["avg_revenue"], 0),
            "busy_level": bl["level"],
            "emoji": bl["emoji"],
        })

    # 5. Product probabilities
    product_probs = build_product_hour_probabilities(prod_df, catalog)

    # 6. Generate 14-day forecast
    last_date = features_df["date"].max()
    max_day_idx = features_df["day_index"].max()

    daily_forecasts = []
    all_peak_hours = []

    for d in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=d)
        dow = future_date.weekday()
        is_wknd = 1 if dow >= 5 else 0
        day_idx = max_day_idx + d
        day_name = future_date.strftime("%A")
        date_str = future_date.strftime("%Y-%m-%d")

        hourly_preds = []
        day_total_trx = 0
        day_total_rev = 0

        for h in OPERATING_HOURS:
            # Build feature vector
            h_sin = np.sin(2 * np.pi * h / 24)
            h_cos = np.cos(2 * np.pi * h / 24)
            is_lunch = 1 if 11 <= h <= 13 else 0
            is_morn = 1 if 8 <= h <= 10 else 0
            is_eve = 1 if 16 <= h <= 19 else 0

            # Use historical rolling averages for this hour
            hist_h = features_df[features_df["hour"] == h]
            roll_trx = float(hist_h["rolling_avg_trx"].iloc[-1]) if len(hist_h) > 0 else 0
            roll_rev = float(hist_h["rolling_avg_revenue"].iloc[-1]) if len(hist_h) > 0 else 0

            feat = np.array([[h, dow, is_wknd, day_idx,
                              h_sin, h_cos, is_lunch, is_morn, is_eve,
                              roll_trx, roll_rev]])

            pred_trx = float(trx_model.predict(feat)[0])
            pred_rev = float(rev_model.predict(feat)[0])
            bl = classify_busy_level(pred_trx, percentiles)

            # Product predictions for this hour
            predicted_products = []
            for pid, hour_data in product_probs.items():
                if h in hour_data:
                    dow_data = hour_data[h].get(dow)
                    if not dow_data:
                        # Fallback: average across all days for this hour
                        all_dows = hour_data[h]
                        avg_prob = np.mean([v["probability"] for v in all_dows.values()])
                        avg_qty = np.mean([v["avg_qty"] for v in all_dows.values()])
                        pinfo = catalog.get(pid, {"name": f"P#{pid}", "price": 0})
                        if avg_prob > 0.1:
                            est_qty = round(avg_qty * pred_trx, 1) if pred_trx > 0 else round(avg_qty, 1)
                            predicted_products.append({
                                "product_id": pid,
                                "product_name": pinfo["name"],
                                "probability": round(float(avg_prob), 3),
                                "estimated_qty": max(0, est_qty),
                                "estimated_revenue": round(est_qty * pinfo["price"], 0),
                            })
                    else:
                        if dow_data["probability"] > 0.1:
                            est_qty = round(dow_data["avg_qty"] * pred_trx, 1) if pred_trx > 0 else round(dow_data["avg_qty"], 1)
                            predicted_products.append({
                                "product_id": pid,
                                "product_name": dow_data["product_name"],
                                "probability": dow_data["probability"],
                                "estimated_qty": max(0, est_qty),
                                "estimated_revenue": round(est_qty * dow_data["product_price"], 0),
                            })

            predicted_products.sort(key=lambda x: x["probability"], reverse=True)

            hour_entry = {
                "hour": f"{h:02d}:00",
                "hour_int": h,
                "predicted_transactions": round(pred_trx, 2),
                "predicted_revenue": round(pred_rev, 0),
                "busy_level": bl["level"],
                "busy_point": bl["point"],
                "busy_emoji": bl["emoji"],
                "predicted_products": predicted_products[:6],
            }
            hourly_preds.append(hour_entry)
            day_total_trx += pred_trx
            day_total_rev += pred_rev

            if bl["level"] in ("PEAK", "HIGH"):
                all_peak_hours.append({
                    "date": date_str,
                    "day_name": day_name,
                    "hour": f"{h:02d}:00",
                    "level": bl["level"],
                    "predicted_trx": round(pred_trx, 2),
                })

        # Find peak hour of the day
        peak = max(hourly_preds, key=lambda x: x["predicted_transactions"])
        # Day-level busy score
        busy_hours_count = sum(1 for x in hourly_preds if x["busy_level"] in ("PEAK", "HIGH"))

        daily_forecasts.append({
            "date": date_str,
            "day_name": day_name,
            "day_of_week": dow,
            "is_weekend": bool(is_wknd),
            "total_predicted_transactions": round(day_total_trx, 1),
            "total_predicted_revenue": round(day_total_rev, 0),
            "peak_hour": peak["hour"],
            "peak_hour_transactions": peak["predicted_transactions"],
            "busy_hours_count": busy_hours_count,
            "hourly_breakdown": hourly_preds,
        })

    # 7. Summary statistics
    all_peak_hours.sort(key=lambda x: x["predicted_trx"], reverse=True)
    busiest_day = max(daily_forecasts, key=lambda x: x["total_predicted_transactions"])
    quietest_day = min(daily_forecasts, key=lambda x: x["total_predicted_transactions"])

    # Day-of-week pattern
    dow_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    dow_pattern = []
    for dow_i in range(7):
        days_data = [d for d in daily_forecasts if d["day_of_week"] == dow_i]
        if days_data:
            avg_trx = np.mean([d["total_predicted_transactions"] for d in days_data])
            dow_pattern.append({"day": dow_names[dow_i], "avg_predicted_trx": round(avg_trx, 1)})

    # Print summary
    print(f"\n[FORECAST] Generated for {forecast_days} days")
    print(f"[BUSIEST] {busiest_day['date']} ({busiest_day['day_name']}) "
          f"- {busiest_day['total_predicted_transactions']} trx")
    print(f"[QUIETEST] {quietest_day['date']} ({quietest_day['day_name']}) "
          f"- {quietest_day['total_predicted_transactions']} trx")
    print(f"[PEAK HOURS] Total in {forecast_days} days: {len(all_peak_hours)}")

    # 8. Compile result
    result = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_days": forecast_days,
        "data_range": {
            "from": features_df["date"].min().strftime("%Y-%m-%d"),
            "to": features_df["date"].max().strftime("%Y-%m-%d"),
            "total_training_days": int(features_df["date"].nunique()),
            "total_sale_transactions": len(hourly_df),
        },
        "model_accuracy": trx_model.metrics,
        "historical_hourly_pattern": historical_patterns,
        "day_of_week_pattern": dow_pattern,
        "percentile_thresholds": percentiles,
        "summary": {
            "busiest_predicted_day": {
                "date": busiest_day["date"],
                "day_name": busiest_day["day_name"],
                "total_transactions": busiest_day["total_predicted_transactions"],
                "peak_hour": busiest_day["peak_hour"],
            },
            "quietest_predicted_day": {
                "date": quietest_day["date"],
                "day_name": quietest_day["day_name"],
                "total_transactions": quietest_day["total_predicted_transactions"],
            },
            "total_peak_hours": len(all_peak_hours),
            "top_5_peak_hours": all_peak_hours[:5],
            "avg_daily_transactions": round(np.mean([d["total_predicted_transactions"] for d in daily_forecasts]), 1),
            "avg_daily_revenue": round(np.mean([d["total_predicted_revenue"] for d in daily_forecasts]), 0),
        },
        "daily_forecasts": daily_forecasts,
    }

    print("=" * 70)
    print("[DONE] Analysis complete!")
    print("=" * 70 + "\n")

    return result
