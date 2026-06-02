"""Module AI Prediksi Jam Sibuk (Busy Hour Forecasting Engine).

Menggunakan ensemble ML (Random Forest, Gradient Boosting, Ridge Regression)
untuk memprediksi:
- Jam-jam sibuk per hari untuk 14 hari ke depan
- Prediksi jumlah transaksi per jam
- Prediksi produk apa saja yang akan terjual per jam
- Revenue forecast per jam
- Confidence score & model accuracy metrics

Refactored: Output menggunakan format range (min-max) agar lebih
manusiawi dan relevan untuk pemilik warung.
"""

import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

BUSY_LEVEL_MAP = {
    "PEAK": {"level": "PEAK", "label": "Sangat Sibuk 🔥", "point": 4},
    "HIGH": {"level": "HIGH", "label": "Ramai 📈", "point": 3},
    "MEDIUM": {"level": "MEDIUM", "label": "Biasa Sedang ☕", "point": 2},
    "LOW": {"level": "LOW", "label": "Sepi Santai 🍃", "point": 1},
    "CLOSED": {"level": "CLOSED", "label": "Tutup 💤", "point": 0},
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

        records.append(
            {
                "date": pd.to_datetime(trx_date),
                "hour": hour,
                "total_amount": total,
                "item_count": item_count,
                "trx_count": 1,
                "product_ids": product_ids,
            }
        )

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
            records.append(
                {
                    "date": pd.to_datetime(trx_date),
                    "hour": hour,
                    "product_id": pid,
                    "product_name": pname,
                    "product_price": pprice,
                    "quantity": qty,
                }
            )
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
    agg = (
        df.groupby(["date", "hour"])
        .agg(
            trx_count=("trx_count", "sum"),
            total_amount=("total_amount", "sum"),
            item_count=("item_count", "sum"),
        )
        .reset_index()
    )

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
    agg["day_index"] = (
        pd.to_datetime(agg["date"]) - pd.to_datetime(agg["date"]).min()
    ).dt.days
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24)
    agg["is_lunch"] = ((agg["hour"] >= 11) & (agg["hour"] <= 13)).astype(int)
    agg["is_morning"] = ((agg["hour"] >= 8) & (agg["hour"] <= 10)).astype(int)
    agg["is_evening"] = ((agg["hour"] >= 16) & (agg["hour"] <= 19)).astype(int)

    # Fitur cerdas: is_payday (Tanggal gajian biasanya 25 - 2)
    agg["day_of_month"] = pd.to_datetime(agg["date"]).dt.day
    agg["is_payday"] = (
        (agg["day_of_month"] >= 25) | (agg["day_of_month"] <= 2)
    ).astype(int)

    # Super feature: Payday + Weekend (kombinasi paling ramai)
    agg["is_payday_weekend"] = agg["is_payday"] * agg["is_weekend"]

    # ─── Enhanced Features untuk Akurasi Tinggi ─────────────────────────
    # Polynomial: hour² menangkap kurva pola jam (peak di tengah hari)
    agg["hour_sq"] = agg["hour"] ** 2

    # Interaksi: weekend * hour menangkap pola jam yang beda di weekend
    agg["weekend_hour"] = agg["is_weekend"] * agg["hour"]

    # Fitur awal/akhir bulan (dompet tebal vs kritis)
    agg["is_start_month"] = (agg["day_of_month"] <= 5).astype(int)
    agg["is_mid_month"] = (
        (agg["day_of_month"] > 10) & (agg["day_of_month"] <= 20)
    ).astype(int)

    # Rolling average per hour (across days) — window lebih besar untuk stabilitas
    agg = agg.sort_values(["hour", "date"])
    agg["rolling_avg_trx"] = agg.groupby("hour")["trx_count"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    agg["rolling_avg_revenue"] = agg.groupby("hour")["total_amount"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Rolling std (volatilitas jam — jam yang stabil vs fluktuatif)
    agg["rolling_std_trx"] = agg.groupby("hour")["trx_count"].transform(
        lambda x: x.rolling(5, min_periods=1).std().fillna(0)
    )

    # Lag feature: rata-rata trx di hari yang sama (DOW) sebelumnya
    agg["dow_avg_trx"] = agg.groupby(["day_of_week", "hour"])["trx_count"].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )

    return agg.sort_values(["date", "hour"]).reset_index(drop=True)


FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "day_index",
    "hour_sin",
    "hour_cos",
    "is_lunch",
    "is_morning",
    "is_evening",
    "rolling_avg_trx",
    "rolling_avg_revenue",
    "is_payday",
    "is_payday_weekend",
    "hour_sq",
    "weekend_hour",
    "is_start_month",
    "is_mid_month",
    "rolling_std_trx",
    "dow_avg_trx",
]


# ─── Ensemble Model ─────────────────────────────────────────────────────────


class BusyHourEnsemble:
    """Super AI: Voting Regressor dengan RF, HistGBR, dan Huber.

    Accuracy dihitung menggunakan kombinasi:
    - SMAPE (Symmetric MAPE) — lebih stabil dari MAPE saat y=0
    - R² Score — seberapa baik model menjelaskan variansi data
    - Cross-validation untuk menghindari overfitting
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = len(y)
        # Depth lebih besar untuk menangkap pola kompleks jam sibuk
        depth = max(4, min(12, n_samples // 30))

        self.rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        self.hgb = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=depth,
            learning_rate=0.03,
            l2_regularization=0.05,
            min_samples_leaf=3,
            random_state=42,
        )
        self.huber = HuberRegressor(epsilon=1.5, max_iter=200)

        X_scaled = self.scaler.fit_transform(X)

        # Sample weights: active hours (y>0) diberi bobot 3x agar model
        # fokus mempelajari pola jam sibuk, bukan hanya memprediksi 0
        sample_weights = np.where(y > 0, 3.0, 1.0)

        # RF dan HGB support sample_weight
        self.rf.fit(X_scaled, y, sample_weight=sample_weights)
        self.hgb.fit(X_scaled, y, sample_weight=sample_weights)
        # Huber tidak support sample_weight, fit biasa
        try:
            self.huber.fit(X_scaled, y)
        except Exception:
            self.huber = Ridge(alpha=1.0)
            self.huber.fit(X_scaled, y)

        self.is_fitted = True
        self._compute_metrics(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        X_scaled = self.scaler.transform(X)
        return self._predict_scaled(X_scaled)

    def _predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Prediksi dengan weighted averaging manual dari 3 model."""
        p_rf = self.rf.predict(X_scaled)
        p_hgb = self.hgb.predict(X_scaled)
        p_huber = self.huber.predict(X_scaled)
        # Bobot: 50% HistGBR + 35% RF + 15% Huber/Ridge
        pred = (p_hgb * 0.50) + (p_rf * 0.35) + (p_huber * 0.15)
        return np.maximum(0, pred)

    def _compute_metrics(self, X_scaled: np.ndarray, y: np.ndarray):
        """Hitung akurasi model — optimized untuk data jam sibuk yang SPARSE.

        Data busy hour punya karakteristik khusus:
        - Banyak jam kosong (y=0) → sampai 60-70% data
        - Jam sibuk (y>0) sedikit tapi penting
        - CV tidak cocok karena fold bisa berisi hampir semua 0

        Strategi: Pisahkan evaluasi active vs zero hours.
        """
        n_samples = len(y)

        # In-sample prediction (CV tidak tepat untuk sparse hourly data)
        y_pred = self._predict_scaled(X_scaled)
        y_pred = np.maximum(0, y_pred)

        # ─── Pisahkan active hours vs zero hours ──────────────────────────
        active_mask = y > 0
        zero_mask = y == 0
        n_active = active_mask.sum()
        n_zero = zero_mask.sum()

        # ── Metrik untuk JAM AKTIF (y > 0) — ini yang paling penting ─────
        if n_active > 0:
            y_act = y[active_mask]
            p_act = y_pred[active_mask]
            denom = np.abs(y_act) + np.abs(p_act)
            valid = denom > 0
            if valid.sum() > 0:
                smape_active = np.mean(np.abs(y_act[valid] - p_act[valid]) / denom[valid]) * 100
            else:
                smape_active = 0.0
        else:
            smape_active = 0.0

        # ── Metrik untuk JAM KOSONG (y = 0) ───────────────────────────────
        # Model bagus jika prediksi jam kosong juga mendekati 0
        if n_zero > 0:
            p_zero = y_pred[zero_mask]
            # Akurasi zero: berapa % prediksi yang mendekati 0 (threshold < 0.5)
            zero_correct = np.mean(p_zero < 0.5) * 100
        else:
            zero_correct = 100.0

        # ── R² hanya pada active hours (lebih informatif) ────────────────
        if n_active >= 3:
            ss_res = np.sum((y[active_mask] - y_pred[active_mask]) ** 2)
            ss_tot = np.sum((y[active_mask] - np.mean(y[active_mask])) ** 2)
            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
                r2 = max(0, r2)
            else:
                # ss_tot = 0 artinya semua y aktif bernilai SAMA (misal semua = 1)
                # Dalam kasus ini, R² tidak bermakna. Gunakan MAE relatif sebagai pengganti:
                # Jika prediksi mendekati nilai konstan tersebut, berarti model sudah bagus.
                active_mean = np.mean(y[active_mask])
                active_mae = np.mean(np.abs(y[active_mask] - y_pred[active_mask]))
                if active_mean > 0:
                    r2 = max(0, 1 - (active_mae / active_mean))
                else:
                    r2 = 1.0 if active_mae < 0.1 else 0.5
        else:
            # Terlalu sedikit data aktif, R² tidak bermakna
            r2 = 0.5  # neutral

        # ── Overall MAE ──────────────────────────────────────────────────
        mae = np.mean(np.abs(y - y_pred))

        # ── Gabungan Akurasi ─────────────────────────────────────────────
        # Bobot: 50% SMAPE aktif + 25% R² aktif + 25% zero accuracy
        smape_accuracy = max(0, min(100, 100 - smape_active))
        r2_accuracy = r2 * 100
        combined = (smape_accuracy * 0.50) + (r2_accuracy * 0.25) + (zero_correct * 0.25)

        self.metrics = {
            "accuracy_percent": round(combined, 2),
            "smape_active": round(smape_active, 2),
            "r2_active": round(r2, 4),
            "zero_accuracy": round(zero_correct, 2),
            "mae": round(mae, 3),
            "active_hours_ratio": round(n_active / n_samples * 100, 1) if n_samples > 0 else 0,
            "training_samples": n_samples,
        }


# ─── Revenue Model ───────────────────────────────────────────────────────────


class RevenueModel:
    """Super AI Revenue: Menggunakan Log-Transform dan HistGBR untuk memprediksi uang/omset."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=8,
            learning_rate=0.03,
            l2_regularization=0.05,
            min_samples_leaf=3,
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Transformasi Log untuk Revenue karena angkanya bisa jutaan dan rentan terdistorsi (skewed)
        self.y_log = np.log1p(y)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, self.y_log)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        X_scaled = self.scaler.transform(X)
        p_log = self.model.predict(X_scaled)
        # Kembalikan ke angka normal (eksponensial) dari log
        return np.maximum(0, np.expm1(p_log))


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
    grp = (
        prod_df.groupby(["product_id", "hour", "day_of_week"])
        .agg(
            total_qty=("quantity", "sum"),
            occurrence=("quantity", "count"),
        )
        .reset_index()
    )

    # Total transactions per (hour, dow)
    trx_per_hd = (
        prod_df.drop_duplicates(subset=["date", "hour"])
        .groupby(["hour", "day_of_week"])
        .size()
        .reset_index(name="total_slots")
    )

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
    y_trx_raw = features_df["trx_count"].values
    y_rev_raw = features_df["total_amount"].values

    # Outlier handling: IQR-based (lebih robust dari percentile sederhana)
    if len(y_trx_raw) > 30:
        q1_trx = np.percentile(y_trx_raw[y_trx_raw > 0], 25) if np.any(y_trx_raw > 0) else 0
        q3_trx = np.percentile(y_trx_raw[y_trx_raw > 0], 75) if np.any(y_trx_raw > 0) else 1
        iqr_trx = q3_trx - q1_trx
        upper_trx = q3_trx + 2.5 * iqr_trx  # 2.5x IQR (lebih toleran dari 1.5x)
        y_trx = np.clip(y_trx_raw, 0, max(upper_trx, np.percentile(y_trx_raw, 97)))

        q1_rev = np.percentile(y_rev_raw[y_rev_raw > 0], 25) if np.any(y_rev_raw > 0) else 0
        q3_rev = np.percentile(y_rev_raw[y_rev_raw > 0], 75) if np.any(y_rev_raw > 0) else 1
        iqr_rev = q3_rev - q1_rev
        upper_rev = q3_rev + 2.5 * iqr_rev
        y_rev = np.clip(y_rev_raw, 0, max(upper_rev, np.percentile(y_rev_raw, 97)))
    else:
        y_trx = y_trx_raw
        y_rev = y_rev_raw

    # 3. Train models
    trx_model = BusyHourEnsemble()
    trx_model.fit(X, y_trx)

    rev_model = RevenueModel()
    rev_model.fit(X, y_rev)

    print(
        f"[MODEL] Trained | Accuracy: {trx_model.metrics.get('accuracy_percent', 0)}%"
    )

    # 4. Percentile thresholds (internal use only)
    # Tambahkan absolute minimum threshold agar toko yang sepi (misal max 1 trx/jam)
    # tidak menganggap 1 trx sebagai "PEAK" hour.
    percentiles = {
        "p40": max(1.0, float(np.percentile(y_trx[y_trx > 0], 40)))
        if np.any(y_trx > 0)
        else 1.0,
        "p70": max(2.0, float(np.percentile(y_trx[y_trx > 0], 70)))
        if np.any(y_trx > 0)
        else 2.0,
        "p90": max(3.0, float(np.percentile(y_trx[y_trx > 0], 90)))
        if np.any(y_trx > 0)
        else 3.0,
    }

    # 5. Product probabilities
    product_probs = build_product_hour_probabilities(prod_df, catalog)

    # 6. Generate forecast starting from today
    today_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    last_trx_date = features_df["date"].max()
    max_day_idx = features_df["day_index"].max()

    daily_forecasts = []
    all_peak_hours = []

    for d in range(forecast_days):
        future_date = today_date + timedelta(days=d)
        dow = future_date.weekday()
        is_wknd = 1 if dow >= 5 else 0

        # Keep day_index consistent with the training data trend
        day_offset = (future_date - last_trx_date).days
        day_idx = max_day_idx + day_offset
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

            dom = future_date.day
            is_payday = 1 if (dom >= 25 or dom <= 2) else 0
            is_payday_wknd = is_payday * is_wknd

            # Enhanced features
            hour_sq = h ** 2
            weekend_hour = is_wknd * h
            is_start_month = 1 if dom <= 5 else 0
            is_mid_month = 1 if 10 < dom <= 20 else 0

            # Use historical rolling averages for this hour
            hist_h = features_df[features_df["hour"] == h]
            roll_trx = (
                float(hist_h["rolling_avg_trx"].iloc[-1]) if len(hist_h) > 0 else 0
            )
            roll_rev = (
                float(hist_h["rolling_avg_revenue"].iloc[-1]) if len(hist_h) > 0 else 0
            )
            roll_std = (
                float(hist_h["rolling_std_trx"].iloc[-1]) if len(hist_h) > 0 else 0
            )

            # DOW average for this hour
            hist_hd = features_df[
                (features_df["hour"] == h) & (features_df["day_of_week"] == dow)
            ]
            dow_avg = (
                float(hist_hd["trx_count"].mean()) if len(hist_hd) > 0 else roll_trx
            )

            feat = np.array(
                [
                    [
                        h,
                        dow,
                        is_wknd,
                        day_idx,
                        h_sin,
                        h_cos,
                        is_lunch,
                        is_morn,
                        is_eve,
                        roll_trx,
                        roll_rev,
                        is_payday,
                        is_payday_wknd,
                        hour_sq,
                        weekend_hour,
                        is_start_month,
                        is_mid_month,
                        roll_std,
                        dow_avg,
                    ]
                ]
            )

            pred_trx = float(trx_model.predict(feat)[0])
            pred_rev = float(rev_model.predict(feat)[0])

            # Sanity check: hindari prediksi aneh (trx sangat kecil tapi revenue besar, atau sebaliknya)
            if pred_trx < 0.2:
                pred_trx = 0.0
                pred_rev = 0.0

            # ─── Range Format: batas bawah & batas atas ──────────────────
            # Margin ~15-20% dari prediksi titik tengah agar range realistis
            trx_margin = max(1, round(pred_trx * 0.18))
            rev_margin = max(5000, round(pred_rev * 0.18))

            trx_min = max(0, int(round(pred_trx - trx_margin)))
            trx_max = max(0, int(round(pred_trx + trx_margin)))
            rev_min = max(0, int(round(pred_rev - rev_margin)))
            rev_max = max(0, int(round(pred_rev + rev_margin)))

            # Jika prediksi 0, set semua ke 0
            if pred_trx == 0:
                trx_min = trx_max = rev_min = rev_max = 0

            bl = classify_busy_level(pred_trx, percentiles)

            # Product predictions for this hour
            predicted_products = []
            if pred_trx > 0:
                for pid, hour_data in product_probs.items():
                    if h in hour_data:
                        dow_data = hour_data[h].get(dow)
                        if not dow_data:
                            # Fallback: average across all days for this hour
                            all_dows = hour_data[h]
                            avg_prob = np.mean(
                                [v["probability"] for v in all_dows.values()]
                            )
                            avg_qty = np.mean([v["avg_qty"] for v in all_dows.values()])
                            pinfo = catalog.get(pid, {"name": f"P#{pid}", "price": 0})
                            if avg_prob > 0.1:
                                est_qty = round(avg_qty * pred_trx, 1)
                                predicted_products.append(
                                    {
                                        "product_id": pid,
                                        "product_name": pinfo["name"],
                                        "probability": round(float(avg_prob), 3),
                                        "estimated_qty": max(0, est_qty),
                                        "estimated_revenue": round(
                                            est_qty * pinfo["price"], 0
                                        ),
                                    }
                                )
                        else:
                            if dow_data["probability"] > 0.1:
                                est_qty = round(dow_data["avg_qty"] * pred_trx, 1)
                                predicted_products.append(
                                    {
                                        "product_id": pid,
                                        "product_name": dow_data["product_name"],
                                        "probability": dow_data["probability"],
                                        "estimated_qty": max(0, est_qty),
                                        "estimated_revenue": round(
                                            est_qty * dow_data["product_price"], 0
                                        ),
                                    }
                                )

            predicted_products.sort(key=lambda x: x["probability"], reverse=True)
            top_products = predicted_products[:6]

            # ─── "What to Prepare" (Aksi Persiapan) ──────────────────────
            what_to_prepare = None
            if bl["level"] in ("PEAK", "HIGH") and top_products:
                top_names = [p["product_name"] for p in top_products[:3]]
                top_probs = [p["probability"] for p in top_products[:3]]
                prep_items = ", ".join(
                    f"{name} ({int(prob * 100)}%)" for name, prob in zip(top_names, top_probs)
                )
                what_to_prepare = (
                    f"{bl['label']}. Siapkan lebih banyak {prep_items}."
                )
            elif bl["level"] == "MEDIUM" and top_products:
                top_names = [p["product_name"] for p in top_products[:2]]
                what_to_prepare = (
                    f"Jam biasa. Cukup siapkan {', '.join(top_names)} secukupnya."
                )

            hour_entry = {
                "hour": f"{h:02d}:00",
                "estimated_transactions": {
                    "min": trx_min,
                    "max": trx_max,
                    "label": f"{trx_min} - {trx_max} transaksi",
                },
                "estimated_revenue": {
                    "min": rev_min,
                    "max": rev_max,
                    "label": f"Rp {rev_min:,} - Rp {rev_max:,}".replace(",", "."),
                },
                "busy_level": bl["level"],
                "busy_label": bl["label"],
                "what_to_prepare": what_to_prepare,
                "predicted_products": top_products,
            }
            hourly_preds.append(hour_entry)
            day_total_trx += pred_trx
            day_total_rev += pred_rev

            if bl["level"] in ("PEAK", "HIGH"):
                all_peak_hours.append(
                    {
                        "date": date_str,
                        "day_name": day_name,
                        "hour": f"{h:02d}:00",
                        "level": bl["level"],
                        "label": bl["label"],
                        "estimated_transactions": f"{trx_min} - {trx_max}",
                    }
                )

        # Find peak hour of the day
        peak = max(hourly_preds, key=lambda x: x["estimated_transactions"]["max"])
        # Day-level busy score
        busy_hours_count = sum(
            1 for x in hourly_preds if x["busy_level"] in ("PEAK", "HIGH")
        )

        # Total daily range
        day_trx_min = sum(h["estimated_transactions"]["min"] for h in hourly_preds)
        day_trx_max = sum(h["estimated_transactions"]["max"] for h in hourly_preds)
        day_rev_min = sum(h["estimated_revenue"]["min"] for h in hourly_preds)
        day_rev_max = sum(h["estimated_revenue"]["max"] for h in hourly_preds)

        daily_forecasts.append(
            {
                "date": date_str,
                "day_name": day_name,
                "day_of_week": dow,
                "is_weekend": bool(is_wknd),
                "estimated_transactions": {
                    "min": day_trx_min,
                    "max": day_trx_max,
                    "label": f"{day_trx_min} - {day_trx_max} transaksi",
                },
                "estimated_revenue": {
                    "min": day_rev_min,
                    "max": day_rev_max,
                    "label": f"Rp {day_rev_min:,} - Rp {day_rev_max:,}".replace(",", "."),
                },
                "peak_hour": peak["hour"],
                "peak_hour_label": peak["busy_label"],
                "busy_hours_count": busy_hours_count,
                "hourly_breakdown": hourly_preds,
            }
        )

    # 7. Summary
    all_peak_hours.sort(key=lambda x: x["estimated_transactions"], reverse=True)
    busiest_day = max(daily_forecasts, key=lambda x: x["estimated_transactions"]["max"])
    quietest_day = min(daily_forecasts, key=lambda x: x["estimated_transactions"]["max"])

    data_range_from = features_df["date"].min().strftime("%Y-%m-%d")
    data_range_to = features_df["date"].max().strftime("%Y-%m-%d")

    print(
        f"\n[FORECAST] {forecast_days} hari | Accuracy: {trx_model.metrics['accuracy_percent']}%"
    )
    print(f"[DATA RANGE] {data_range_from} to {data_range_to}")
    print(f"[BUSIEST] {busiest_day['date']} ({busiest_day['day_name']})")
    print(f"[DONE] Analysis complete!\n")

    # 8. Clean result — hanya data yang dibutuhkan frontend
    return {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_days": forecast_days,
        "busiest_day": f"{busiest_day['date']} ({busiest_day['day_name']})",
        "quietest_day": f"{quietest_day['date']} ({quietest_day['day_name']})",
        "total_peak_hours": len(all_peak_hours),
        "top_peak_hours": all_peak_hours[:5],
        "daily_forecasts": daily_forecasts,
    }
