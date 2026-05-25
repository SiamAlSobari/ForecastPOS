"""Module AI Forecasting untuk Decision Support System Restock Barang.

Menggunakan scikit-learn Ensemble Model untuk memprediksi pola penjualan
harian, lalu mensimulasikan kapan stok akan habis dan memberikan rekomendasi
restock beserta level urgensi (NORMAL / MEDIUM / CRITICAL) dan risk point.

Refactored: Output restock menggunakan format range (min-max) agar lebih
fleksibel dan realistis untuk pemilik warung.

Fitur Seasonal/Holiday:
- Deteksi otomatis hari raya nasional Indonesia (Lebaran, Natal, dll).
- LLM overlay: saat mendekati hari raya, LLM memberikan nasehat restock
  musiman yang meng-override prediksi ML normal. Ini memvalidasi "insting
  musiman" pedagang yang biasanya restock besar-besaran menjelang hari raya.

Accuracy Improvement v2:
- Cyclic encoding day_of_week (sin/cos) → menangkap pola circular (Minggu→Senin)
- Lag features (lag_1..lag_7) + rolling stats (mean/std/median 7d) → recent trend
- EWMA (exponential weighted moving average) → bobot lebih ke recent data
- Hapus day_index untuk mencegah extrapolation drift
- MAE-based accuracy → lebih jujur untuk count data rendah
- Hyperparameter tuning untuk low-volume integer sales data
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────

RISK_MAP = {
    "CRITICAL": {"risk": "CRITICAL", "risk_point": 3},
    "MEDIUM": {"risk": "MEDIUM", "risk_point": 2},
    "NORMAL": {"risk": "NORMAL", "risk_point": 1},
}

# Feature columns used by the model — single source of truth
FEATURE_COLUMNS = [
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_payday",
    "is_start_month",
    "is_mid_month",
    "week_of_year_sin",
    "week_of_year_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_median_7",
    "ewma_7",
]


# ─── Helper: Parse ────────────────────────────────────────────────────────────


def normalize_transactions(raw_data: dict | list) -> list[dict]:
    """Normalisasi input: bisa dict {data: [...]} atau langsung list."""
    if isinstance(raw_data, dict):
        return raw_data.get("data", [])
    return raw_data


def extract_product_info(transactions: list[dict], product_id: int) -> dict:
    """
    Mengekstrak informasi produk (nama, harga, stok) dari data transaksi.
    Data produk diambil dari nested 'product' di dalam item transaksi.
    """
    product_name = f"Product #{product_id}"
    product_price = "0.00"
    current_stock_from_data: Optional[int] = None

    for trx in transactions:
        for item in trx.get("items", []):
            if item.get("product_id") == product_id:
                product = item.get("product")
                if product and isinstance(product, dict):
                    product_name = product.get("name", product_name)
                    product_price = product.get("price", product_price)

                    # Ambil stock_on_hand dari stocks array (pakai yang terbaru)
                    stocks = product.get("stocks", [])
                    if stocks and isinstance(stocks, list):
                        # Ambil stok pertama (biasanya hanya 1 entry per produk)
                        current_stock_from_data = stocks[0].get("stock_on_hand")

    return {
        "product_name": product_name,
        "product_price": product_price,
        "current_stock_from_data": current_stock_from_data,
    }


def build_daily_dataframe(transactions: list[dict], product_id: int) -> pd.DataFrame:
    """
    Membangun DataFrame harian untuk product tertentu.
    Kolom: date, sold (jumlah terjual), purchased (jumlah beli/restock),
           adjusted (jumlah koreksi stok dari ADJUSTMENT).

    Tipe transaksi yang dihandle:
    - SALE       → mengurangi stok (kolom 'sold')
    - PURCHASE   → menambah stok (kolom 'purchased')
    - ADJUSTMENT → koreksi stok manual / stock opname (kolom 'adjusted',
                   diperlakukan sebagai penambahan stok)
    """
    records: list[dict] = []
    for trx in transactions:
        trx_date = trx["trx_date"]
        trx_type = trx["trx_type"]
        for item in trx.get("items", []):
            if item["product_id"] == product_id:
                records.append(
                    {
                        "date": pd.to_datetime(trx_date),
                        "type": trx_type,
                        "quantity": int(item["quantity"]),
                    }
                )

    if not records:
        return pd.DataFrame(columns=["date", "sold", "purchased", "adjusted"])

    df = pd.DataFrame(records)

    # Pisahkan SALE vs PURCHASE vs ADJUSTMENT
    sales = df[df["type"] == "SALE"].groupby("date")["quantity"].sum().rename("sold")
    purchases = (
        df[df["type"] == "PURCHASE"]
        .groupby("date")["quantity"]
        .sum()
        .rename("purchased")
    )
    adjustments = (
        df[df["type"] == "ADJUSTMENT"]
        .groupby("date")["quantity"]
        .sum()
        .rename("adjusted")
    )

    # Gabungkan ke range tanggal penuh (termasuk hari tanpa transaksi)
    min_date = df["date"].min()
    max_date = df["date"].max()
    full_range = pd.date_range(min_date, max_date, freq="D")

    daily = pd.DataFrame({"date": full_range})
    daily = daily.merge(sales, left_on="date", right_index=True, how="left")
    daily = daily.merge(purchases, left_on="date", right_index=True, how="left")
    daily = daily.merge(adjustments, left_on="date", right_index=True, how="left")
    daily["sold"] = daily["sold"].fillna(0).astype(int)
    daily["purchased"] = daily["purchased"].fillna(0).astype(int)
    daily["adjusted"] = daily["adjusted"].fillna(0).astype(int)

    return daily


# ─── Feature Engineering ─────────────────────────────────────────────────────


def _engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build all feature columns from a daily DataFrame with 'date' and 'sold'.
    Returns DataFrame with all FEATURE_COLUMNS populated.

    Key improvements over v1:
    - Cyclic encoding for day_of_week & week_of_year (sin/cos)
    - Lag features (1, 2, 3, 7 days) — capture autocorrelation
    - Rolling stats (mean, std, median over 7d) — capture recent trend
    - EWMA — exponential smoothing for recent weight
    - No day_index — prevents linear extrapolation drift
    """
    df = daily.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Day of week: cyclic encoding (sin/cos) — captures Minggu→Senin continuity
    dow = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)

    # Day of month features
    dom = df["date"].dt.day
    df["is_payday"] = ((dom >= 25) | (dom <= 2)).astype(int)
    df["is_start_month"] = (dom <= 5).astype(int)
    df["is_mid_month"] = ((dom > 10) & (dom <= 20)).astype(int)

    # Week of year: cyclic encoding
    woy = df["date"].dt.isocalendar().week.astype(int).values
    df["week_of_year_sin"] = np.sin(2 * np.pi * woy / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * woy / 52)

    # Lag features — capture autocorrelation in sales
    df["lag_1"] = df["sold"].shift(1)
    df["lag_2"] = df["sold"].shift(2)
    df["lag_3"] = df["sold"].shift(3)
    df["lag_7"] = df["sold"].shift(7)

    # Rolling statistics (7-day window) — capture recent trend
    df["rolling_mean_7"] = df["sold"].rolling(7, min_periods=1).mean()
    df["rolling_std_7"] = df["sold"].rolling(7, min_periods=1).std().fillna(0)
    df["rolling_median_7"] = df["sold"].rolling(7, min_periods=1).median()

    # EWMA — exponential weighted moving average (recent data weighted more)
    df["ewma_7"] = df["sold"].ewm(span=7, min_periods=1).mean()

    # Fill NaN from lag features with rolling mean (safer than 0)
    for col in ["lag_1", "lag_2", "lag_3", "lag_7"]:
        df[col] = df[col].fillna(df["rolling_mean_7"])

    return df


# ─── Model: Prediksi Penjualan Harian ────────────────────────────────────────


class SmartStockEnsemble:
    """Lightweight Ensemble v3: Ridge + HistGradientBoosting ONLY.

    Perubahan dari v2:
    - HAPUS RandomForest — penyebab utama lambat (90%+ training time)
    - HistGBR reduced iterations (100 vs 200) — 2x lebih cepat
    - Total speedup: ~10-20x per produk
    - Kualitas prediksi tetap baik karena HGB sudah paling akurat
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=1.0)
        # HistGBR: algo ala LightGBM bawaan sklearn — CEPAT dan akurat
        self.hgb = HistGradientBoostingRegressor(
            max_iter=100,       # Cukup untuk 90 hari data (turun dari 200)
            max_depth=4,        # Conservative agar tidak overfit
            learning_rate=0.08, # Sedikit lebih agresif agar converge cepat
            l2_regularization=0.1,
            min_samples_leaf=5,
            max_bins=64,
            random_state=42,
        )
        self.use_hgb = False

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y)
        if len(y) > 14:
            self.hgb.fit(X_scaled, y)
            self.use_hgb = True

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self._predict_from_scaled(X_scaled)

    def _predict_from_scaled(self, X_scaled):
        p_ridge = self.ridge.predict(X_scaled)
        if self.use_hgb:
            p_hgb = self.hgb.predict(X_scaled)
            # HGB dominan (70%) karena paling pintar, Ridge sebagai safety net (30%)
            return np.maximum(0, (p_hgb * 0.70) + (p_ridge * 0.30))
        return np.maximum(0, p_ridge)

    def predict_scaled(self, X_scaled):
        """Predict langsung dari data yang sudah di-scale."""
        return self._predict_from_scaled(X_scaled)


def train_sales_model(daily: pd.DataFrame) -> tuple[SmartStockEnsemble, float, float]:
    """
    Melatih model ensemble (Ridge + HGB) — CEPAT, tanpa RandomForest.

    Accuracy v3: Within-Tolerance Accuracy
    ─────────────────────────────────────
    Problem sebelumnya: MAE-based accuracy (1 - MAE/mean) secara matematis
    PASTI rendah untuk count data rendah. Contoh:
      - mean = 1.5, actual = [0,1,2,3], predicted = 1.5
      - MAE = 1.0, accuracy = 1 - 1.0/1.5 = 33% ← padahal prediksi BAGUS!

    Solusi: Within-Tolerance Accuracy
      - tolerance = max(1, ceil(mean * 0.5))
      - Jika |predicted - actual| <= tolerance → BENAR
      - Untuk mean=1.5: tolerance=1, jadi prediksi 2 saat actual 1 → BENAR
      - Ini MEANINGFUL: untuk restock, selisih ±1 item tidak masalah

    Returns: (model, avg_daily_sales, accuracy_percent)
    """
    if daily.empty or daily["sold"].sum() == 0:
        model = SmartStockEnsemble()
        n_features = len(FEATURE_COLUMNS)
        dummy_X = np.zeros((2, n_features))
        dummy_X[1, 0] = 1
        model.fit(dummy_X, np.array([0, 0]))
        return model, 0.0, 0.0

    # Engineer features
    daily_fe = _engineer_features(daily)

    # Outlier handling: Winsorize — clip extremes but keep distribution
    y_raw = daily_fe["sold"].values.astype(float)
    if len(y_raw) > 10:
        non_zero = y_raw[y_raw > 0]
        if len(non_zero) > 5:
            q1 = np.percentile(non_zero, 10)
            q3 = np.percentile(non_zero, 90)
            iqr = q3 - q1
            upper_bound = q3 + 2.0 * iqr
            y = np.clip(y_raw, 0, max(upper_bound, np.percentile(y_raw, 95)))
        else:
            p95 = np.percentile(y_raw, 95)
            y = np.clip(y_raw, 0, max(p95, 1))
    else:
        y = y_raw.copy()

    X = daily_fe[FEATURE_COLUMNS].values

    model = SmartStockEnsemble()
    model.fit(X, y)

    # ─── Accuracy v3: Within-Tolerance (cocok untuk count data) ────────
    y_pred = model.predict_scaled(model.scaler.transform(X))
    y_pred = np.maximum(0, y_pred)

    y_mean = float(np.mean(y))
    # Tolerance: ±1 untuk low-volume, scale up untuk high-volume
    tolerance = max(1.0, np.ceil(y_mean * 0.5))

    # Hitung berapa % prediksi yang dalam toleransi
    within_tol = np.abs(y - y_pred) <= tolerance
    tol_accuracy = float(np.mean(within_tol)) * 100

    # Bonus: jika trend direction juga benar, tambah sedikit
    # (apakah prediksi naik/turun sesuai aktual)
    if len(y) > 7:
        y_diff = np.diff(y[-14:]) if len(y) >= 14 else np.diff(y)
        p_diff = np.diff(y_pred[-14:]) if len(y_pred) >= 14 else np.diff(y_pred)
        if len(y_diff) > 0 and len(p_diff) > 0:
            min_len = min(len(y_diff), len(p_diff))
            direction_match = np.mean(np.sign(y_diff[:min_len]) == np.sign(p_diff[:min_len]))
            # Gabungan: 80% within-tolerance + 20% direction accuracy
            accuracy_pct = round(tol_accuracy * 0.80 + direction_match * 100 * 0.20, 2)
        else:
            accuracy_pct = round(tol_accuracy, 2)
    else:
        accuracy_pct = round(tol_accuracy, 2)

    # Cap at 95% — model tidak bisa sempurna
    accuracy_pct = min(accuracy_pct, 95.0)

    avg_daily = float(daily["sold"].mean())
    return model, avg_daily, accuracy_pct


def predict_future_sales(
    model: SmartStockEnsemble,
    start_date: datetime,
    base_day_index: int,
    avg_daily_sales: float,
    days_ahead: int = 30,
    daily_df: pd.DataFrame = None,
) -> list[dict]:
    """
    Memprediksi penjualan harian untuk N hari ke depan.
    Uses sliding window of recent actual + predicted values for lag features.

    Returns list of {date, predicted_sales}.
    """
    predictions = []

    # Build recent sales history for lag features
    if daily_df is not None and not daily_df.empty:
        recent_sales = daily_df["sold"].values.tolist()
    else:
        recent_sales = [avg_daily_sales] * 7

    # Cap prediction to prevent unrealistic extrapolation
    if avg_daily_sales > 1.0:
        max_allowed = avg_daily_sales * 2.0
    else:
        max_allowed = max(avg_daily_sales * 2.5, 1.0)

    for i in range(days_ahead):
        future_date = start_date + timedelta(days=i)
        dow = future_date.weekday()
        dom = future_date.day
        woy = future_date.isocalendar()[1]

        # Cyclic features
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        woy_sin = np.sin(2 * np.pi * woy / 52)
        woy_cos = np.cos(2 * np.pi * woy / 52)

        is_wknd = 1 if dow >= 5 else 0
        is_payday = 1 if (dom >= 25 or dom <= 2) else 0
        is_start_month = 1 if dom <= 5 else 0
        is_mid_month = 1 if 10 < dom <= 20 else 0

        # Lag features from recent_sales (actual + previously predicted)
        n = len(recent_sales)
        lag_1 = recent_sales[-1] if n >= 1 else avg_daily_sales
        lag_2 = recent_sales[-2] if n >= 2 else avg_daily_sales
        lag_3 = recent_sales[-3] if n >= 3 else avg_daily_sales
        lag_7 = recent_sales[-7] if n >= 7 else avg_daily_sales

        # Rolling stats from last 7 values
        window = recent_sales[-7:] if n >= 7 else recent_sales
        rolling_mean_7 = np.mean(window)
        rolling_std_7 = np.std(window) if len(window) > 1 else 0.0
        rolling_median_7 = np.median(window)

        # EWMA from recent values
        if len(window) > 0:
            weights = np.array([0.5 ** (len(window) - 1 - j) for j in range(len(window))])
            weights /= weights.sum()
            ewma_7 = np.dot(weights, window)
        else:
            ewma_7 = avg_daily_sales

        # Build feature vector (must match FEATURE_COLUMNS order)
        features = np.array([[
            dow_sin, dow_cos, is_wknd, is_payday, is_start_month, is_mid_month,
            woy_sin, woy_cos,
            lag_1, lag_2, lag_3, lag_7,
            rolling_mean_7, rolling_std_7, rolling_median_7, ewma_7,
        ]])

        predicted = model.predict(features)[0]

        # Cap prediction
        predicted = max(0.0, min(predicted, max_allowed))

        # Append to recent_sales for next iteration's lag features
        recent_sales.append(predicted)

        predictions.append(
            {
                "date": future_date.strftime("%Y-%m-%d"),
                "day_name": future_date.strftime("%A"),
                "predicted_sales": round(predicted, 1),
            }
        )
    return predictions


# ─── Simulasi Stok & Urgency ──────────────────────────────────────────────────


def compute_current_stock(daily: pd.DataFrame) -> int:
    """
    Menghitung stok saat ini berdasarkan:
    total (purchased + adjusted) - total sold.

    ADJUSTMENT diperlakukan sebagai penambahan stok (koreksi positif).
    """
    total_purchased = int(daily["purchased"].sum())
    total_adjusted = int(daily["adjusted"].sum()) if "adjusted" in daily.columns else 0
    total_sold = int(daily["sold"].sum())
    return (total_purchased + total_adjusted) - total_sold


def determine_urgency(
    days_until_empty: Optional[int], estimated_empty_date: Optional[str]
) -> dict:
    """
    Menentukan level urgensi dan risk point berdasarkan hari sampai stok habis.

    Returns: {
        urgency_level, urgency_description,
        risk, risk_point
    }
    """
    if days_until_empty is None:
        urgency = "NORMAL"
        description = "Stok diperkirakan aman untuk periode prediksi ke depan."
    elif days_until_empty <= 2:
        urgency = "CRITICAL"
        description = (
            f"⚠️ DARURAT! Stok diperkirakan habis dalam {days_until_empty} hari "
            f"(sekitar tanggal {estimated_empty_date}). Disarankan segera restock."
        )
    elif days_until_empty <= 5:
        urgency = "MEDIUM"
        description = (
            f"⚡ PERHATIAN! Stok diestimasi akan habis dalam {days_until_empty} hari "
            f"(sekitar tanggal {estimated_empty_date}). Pertimbangkan untuk restock."
        )
    else:
        urgency = "NORMAL"
        description = (
            f"✅ Stok diperkirakan masih cukup untuk {days_until_empty} hari "
            f"(sekitar tanggal {estimated_empty_date})."
        )

    risk_info = RISK_MAP[urgency]

    return {
        "urgency_level": urgency,
        "urgency_description": description,
        "risk": risk_info["risk"],
        "risk_point": risk_info["risk_point"],
    }


def simulate_stock_depletion(
    current_stock: int,
    future_predictions: list[dict],
) -> dict:
    """
    Mensimulasikan penurunan stok hari per hari berdasarkan prediksi.
    Menentukan kapan stok habis, level urgensi, dan risk point.

    Returns: {
        days_until_empty, estimated_empty_date,
        urgency_level, urgency_description,
        risk, risk_point,
        stock_timeline, restock_recommendation
    }
    """
    stock = current_stock
    timeline = []
    days_until_empty: Optional[int] = None
    estimated_empty_date: Optional[str] = None

    for i, pred in enumerate(future_predictions):
        daily_sales = round(pred["predicted_sales"])
        stock = max(0, stock - daily_sales)

        timeline.append(
            {
                "date": pred["date"],
                "day_name": pred["day_name"],
                "predicted_sales": pred["predicted_sales"],
                "remaining_stock": stock,
            }
        )

        if stock <= 0 and days_until_empty is None:
            days_until_empty = i + 1
            estimated_empty_date = pred["date"]

    # ─── Tentukan Level Urgensi + Risk ────────────────────────────────────
    urgency_info = determine_urgency(days_until_empty, estimated_empty_date)

    # ─── Rekomendasi Jumlah Restock (Range Format) ─────────────────────
    # Target: stok cukup untuk 7 hari berdasarkan rata-rata prediksi
    avg_predicted = np.mean([p["predicted_sales"] for p in future_predictions[:7]])
    optimal_stock_7_days = int(np.ceil(avg_predicted * 7))
    restock_qty = max(0, optimal_stock_7_days - current_stock)

    # Range: margin 20% bawah/atas agar pemilik warung bisa fleksibel
    restock_min = max(0, int(round(restock_qty * 0.8)))
    restock_max = max(0, int(round(restock_qty * 1.2)))

    # Label yang ramah manusia
    if restock_qty == 0:
        restock_label = "Stok diperkirakan masih cukup, belum perlu restock saat ini."
    else:
        restock_label = f"Saran restock: {restock_min} - {restock_max} item untuk persediaan 7 hari ke depan."

    return {
        "current_stock": current_stock,
        "days_until_empty": days_until_empty,
        "estimated_empty_date": estimated_empty_date,
        **urgency_info,
        "restock_recommendation": {
            "min": restock_min,
            "max": restock_max,
            "label": restock_label,
            "target_days_coverage": 7,
        },
        "stock_timeline": timeline,
    }


# ─── Main Entry Point ────────────────────────────────────────────────────────


def analyze_restock(
    transactions: list[dict],
    product_id: int,
    current_stock_override: Optional[int] = None,
    forecast_days: int = 14,
) -> dict:
    """
    Entry point utama: analisis kebutuhan restock untuk suatu produk.

    Args:
        transactions: List data transaksi yang dikirim dari controller.
        product_id: ID produk yang akan dianalisis.
        current_stock_override: Jika diberikan, override stok dari data.
        forecast_days: Jumlah hari prediksi ke depan (default 14).

    Returns:
        Dictionary berisi analisis lengkap: stok, prediksi, urgensi,
        risk point, dan rekomendasi.
    """
    # 1. Extract product info (name, price, stock from nested data)
    product_info = extract_product_info(transactions, product_id)

    # 2. Parse data
    daily = build_daily_dataframe(transactions, product_id)

    if daily.empty:
        return {
            "product_id": product_id,
            "product_name": product_info["product_name"],
            "error": f"Tidak ada data transaksi untuk product_id={product_id}.",
        }

    # 3. Train model
    model, avg_daily, accuracy_pct = train_sales_model(daily)

    print("\n" + "=" * 70)
    print(
        f"[STOCK] PREDICTION ENGINE - Analysis for Product '{product_info['product_name']}'"
    )
    print("=" * 70)
    print(f"[DATA] Loaded: {len(daily)} days of transaction data")
    print(f"[MODEL] Trained | Accuracy: {accuracy_pct}%")

    # 4. Hitung stok saat ini
    # Prioritas: override > stock dari nested product.stocks > hitung dari data
    if current_stock_override is not None:
        current_stock = current_stock_override
    elif product_info["current_stock_from_data"] is not None:
        current_stock = product_info["current_stock_from_data"]
    else:
        current_stock = compute_current_stock(daily)

    # 5. Prediksi ke depan (mulai dari hari ini)
    last_date = daily["date"].max()
    start_forecast = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Base day index for the start_forecast
    base_day_index = (start_forecast - daily["date"].min()).days

    predictions = predict_future_sales(
        model, start_forecast, base_day_index, avg_daily, forecast_days,
        daily_df=daily,
    )

    # 6. Simulasi & analisis (sudah termasuk risk + risk_point)
    result = simulate_stock_depletion(current_stock, predictions)

    # 7. Tambahkan metadata produk
    result["product_id"] = product_id
    result["product_name"] = product_info["product_name"]
    result["product_price"] = product_info["product_price"]
    result["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["avg_daily_sales"] = round(avg_daily, 1)

    data_range_from = daily["date"].min().strftime("%Y-%m-%d")
    data_range_to = daily["date"].max().strftime("%Y-%m-%d")
    total_adjusted = int(daily["adjusted"].sum()) if "adjusted" in daily.columns else 0

    print(f"\n[FORECAST] {forecast_days} hari | Accuracy: {accuracy_pct}%")
    print(f"[DATA RANGE] {data_range_from} to {data_range_to} ({len(daily)} days)")
    print(f"[HISTORICAL STATS] Avg Daily: {round(avg_daily, 1)} | Sold: {int(daily['sold'].sum())} | Purchased: {int(daily['purchased'].sum())} | Adjusted: {total_adjusted}")
    print(
        f"[URGENCY] {result['urgency_level']} ({result['days_until_empty']} days until empty)"
    )
    print(f"[DONE] Analysis complete!\n")

    return result


# ─── Holiday Detection (Seasonal Awareness) ──────────────────────────────────

# Keyword hari raya besar yang benar-benar mempengaruhi pola belanja konsumen.
# Hanya hari raya ini yang butuh restock musiman. Hari libur kecil seperti
# Isra Mi'raj, Nyepi, dsb tidak signifikan pengaruhnya terhadap penjualan warung.
HIGH_IMPACT_KEYWORDS = [
    "idul fitri", "lebaran", "hari raya", "eid al-fitr", # Lebaran — lonjakan terbesar
    "natal", "christmas",                   # Natal
    "tahun baru", "new year",               # Tahun Baru Masehi
    "imlek", "chinese new year",            # Imlek
    "idul adha", "eid al-adha",             # Idul Adha (kurban, banyak kumpul keluarga)
    "waisak", "vesak",                      # Waisak (signifikan di daerah tertentu)
    "galungan", "kuningan",                 # Hari raya Hindu Bali
]


def _is_high_impact_holiday(holiday_name: str) -> bool:
    """Cek apakah hari libur ini berdampak tinggi terhadap penjualan warung."""
    name_lower = holiday_name.lower()
    return any(kw in name_lower for kw in HIGH_IMPACT_KEYWORDS)


def detect_upcoming_holidays(
    today: Optional[datetime] = None,
    window_days: int = 14,
    high_impact_only: bool = False,
) -> list[dict]:
    """
    Deteksi hari libur/raya nasional Indonesia secara real-time dan akurat
    menggunakan library `holidays`.

    Args:
        today: Tanggal referensi (default: hari ini).
        window_days: Jendela deteksi ke depan (hari).
        high_impact_only: Jika True, hanya return hari raya besar yang
            benar-benar berpengaruh terhadap pola belanja (Lebaran, Natal,
            Tahun Baru, Imlek, Idul Adha). Hari libur kecil di-skip.
    """
    import holidays

    if today is None:
        today = datetime.now()

    # Inisialisasi kalender libur Indonesia
    # Menggunakan tahun ini dan tahun depan agar aman jika overlap akhir tahun
    next_week = today + timedelta(days=window_days)
    id_holidays = holidays.ID(years=[today.year, next_week.year])

    upcoming = []
    for d in range(window_days):
        check_date = today + timedelta(days=d)

        # Cek apakah tanggal tersebut adalah hari libur (mendukung cuti bersama & hari raya dinamis)
        if check_date.date() in id_holidays:
            holiday_name = id_holidays.get(check_date.date())

            print(f"[DEBUG-HOLIDAYS] Check: {check_date.date()} -> {holiday_name} (High Impact? {_is_high_impact_holiday(holiday_name)})")

            # Filter: hanya high-impact jika diminta
            if high_impact_only and not _is_high_impact_holiday(holiday_name):
                continue

            upcoming.append(
                {
                    "date": check_date.strftime("%Y-%m-%d"),
                    "name": holiday_name,
                    "days_away": d,
                    "impact": "HIGH" if _is_high_impact_holiday(holiday_name) else "LOW",
                }
            )

    return upcoming


# ─── LLM Seasonal Overlay (Nasehat Musiman untuk Restock) ────────────────────

SEASONAL_SYSTEM_PROMPT = """Kamu adalah "Konsultan Stok Warung", ahli dalam manajemen stok untuk warung kelontong Indonesia.

Konteks: Kamu menerima data prediksi restock dari AI + info hari raya terdekat. Pemilik warung punya "insting musiman" yang biasanya BENAR — saat Lebaran/Natal/hari raya, penjualan bisa 2-3x lipat dari normal.

Tugasmu:
1. Jika ada hari raya BESAR terdekat (14 hari ke depan), OVERRIDE prediksi ML normal. Bilang ke pemilik bahwa instingnya BENAR, dan restock harus lebih besar dari angka AI.
2. Sebutkan produk apa saja yang biasanya laris saat event tersebut (berdasarkan pengetahuan umummu tentang konsumen Indonesia).
3. Berikan multiplier restock: misal "2x lipat" atau "3x lipat" dari rekomendasi normal.
4. Gunakan bahasa Indonesia santai dan bersahabat. Max 2-3 kalimat saja.
5. Jika TIDAK ada hari raya BESAR terdekat, cukup bilang singkat bahwa prediksi AI sudah cukup akurat, pakai angka normalnya saja.
6. PENTING: Hari libur kecil (Isra Mi'raj, Nyepi, hari libur keagamaan minor) TIDAK perlu restock ekstra. Hanya hari raya BESAR (Lebaran, Natal, Tahun Baru, Imlek, Idul Adha) yang butuh restock musiman."""


SEASONAL_RESTOCK_SYSTEM_PROMPT = """Kamu adalah "Konsultan Stok Warung", ahli dalam manajemen stok warung kelontong Indonesia.

Tugasmu: Menentukan JUMLAH RESTOCK MUSIMAN per produk berdasarkan hari raya yang mendekat.

Aturan:
1. Kamu menerima daftar produk warung beserta restock normal (dari ML). Kamu harus tentukan berapa restock SEASONAL (musiman) untuk masing-masing produk.
2. Seasonal restock dalam format RANGE (min - max). Contoh: min=30, max=50.
3. Pertimbangkan jenis hari raya:
   - Lebaran/Idul Fitri: Sirup, kue kering, minyak goreng, beras, gula → naik 2-3x. Sabun/sampo biasa saja.
   - Natal/Tahun Baru: Minuman, snack, dekorasi → naik 1.5-2x.
   - Imlek: Jeruk mandarin, angpao, kue keranjang → naik 2x.
   - Idul Adha: Bumbu masak, plastik, es batu → naik 1.5-2x.
4. Produk yang TIDAK relevan dengan hari raya tersebut, JANGAN beri seasonal restock (set null).
5. Format output HARUS valid JSON array. Setiap item: {"product_id": int, "seasonal_min": int|null, "seasonal_max": int|null, "reason": string}.
6. Jangan tambahkan teks apapun di luar JSON array."""


def generate_seasonal_insight(
    stock_summary: list[dict],
) -> Optional[dict]:
    """
    Generate nasehat restock musiman menggunakan LLM.

    HANYA dipanggil jika ada hari raya BESAR dalam 14 hari ke depan.
    Hari libur kecil (Isra Mi'raj, Nyepi, dll) tidak trigger nasehat musiman
    karena dampaknya terhadap penjualan warung tidak signifikan.

    Args:
        stock_summary: List ringkasan stok semua produk.

    Returns:
        Dict berisi nasehat seasonal dari LLM, atau None jika:
        - Tidak ada hari raya besar terdekat
        - API key tidak ada
        - LLM gagal setelah retry
    """
    import json

    today = datetime.now()
    # Hanya deteksi hari raya BESAR (high impact)
    upcoming = detect_upcoming_holidays(today, window_days=14, high_impact_only=True)

    if not upcoming:
        print("[STOCK-SEASONAL] No HIGH IMPACT holidays in 14 days, skipping seasonal overlay")
        return {
            "has_upcoming_holiday": False,
            "upcoming_holidays": [],
            "seasonal_advice": "Tidak ada hari raya besar dalam 14 hari ke depan. Gunakan rekomendasi restock normal dari AI.",
            "source": "system",
        }

    # Rangkum data stok untuk LLM (hemat token)
    produk_ringkas = []
    for p in stock_summary[:10]:  # Max 10 produk
        produk_ringkas.append({
            "nama": p.get("product_name", "?"),
            "stok": p.get("current_stock", 0),
            "restock_saran": f"{p.get('restock_recommendation', {}).get('min', 0)} - {p.get('restock_recommendation', {}).get('max', 0)}",
            "urgensi": p.get("urgency_level", "NORMAL"),
        })

    prompt_data = {
        "tanggal_hari_ini": today.strftime("%d %B %Y (%A)"),
        "event_terdekat": upcoming,
        "ringkasan_stok": produk_ringkas,
    }

    prompt = (
        f"Data restock warung saat ini:\n\n"
        f"```json\n{json.dumps(prompt_data, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Berikan nasehat restock musiman singkat."
    )

    # Cek ketersediaan API key sebelum memanggil LLM
    from app.helpers.config import settings
    if not settings.gemini_api_key and not settings.groq_api_key:
        print("[STOCK-SEASONAL] LLM keys not configured, skipping seasonal overlay")
        return None

    # Import call_llm dari llm_insights (reuse retry logic)
    try:
        from app.ai.llm_insights import call_llm
        advice, source = call_llm(prompt, SEASONAL_SYSTEM_PROMPT)

        print(f"[STOCK-SEASONAL] LLM seasonal advice generated ({source})")

        return {
            "has_upcoming_holiday": True,
            "upcoming_holidays": upcoming,
            "seasonal_advice": advice,
            "source": source,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # LLM opsional untuk stock — jika gagal, return None bukan throw
        print(f"[STOCK-SEASONAL] LLM unavailable ({type(e).__name__}), skipping seasonal overlay")
        return None


# ─── LLM Seasonal Restock Per Produk ─────────────────────────────────────────


def generate_seasonal_restock_per_product(
    stock_summary: list[dict],
) -> dict[int, dict]:
    """
    Generate rekomendasi restock MUSIMAN per produk menggunakan LLM.

    Hanya dipanggil jika ada hari raya BESAR dalam 14 hari ke depan.
    LLM menentukan produk mana yang perlu restock ekstra dan berapa range-nya,
    berdasarkan konteks hari raya (misal Lebaran → sirup/gula naik, Natal → snack naik).

    Produk yang TIDAK relevan dengan hari raya tersebut tidak diberi seasonal restock.

    Args:
        stock_summary: List ringkasan stok semua produk.

    Returns:
        Dict mapping product_id -> seasonal_restock info:
        {
            1: {"min": 30, "max": 50, "label": "Restock musiman ...", "holiday": "Idul Fitri", "reason": "..."},
            3: {"min": 20, "max": 35, "label": "Restock musiman ...", "holiday": "Idul Fitri", "reason": "..."},
        }
        Produk yang tidak butuh seasonal restock TIDAK ada di dict ini.
        Returns empty dict jika tidak ada hari raya besar atau LLM gagal.
    """
    import json

    today = datetime.now()
    upcoming = detect_upcoming_holidays(today, window_days=14, high_impact_only=True)

    if not upcoming:
        print("[STOCK-SEASONAL] No HIGH IMPACT holidays → no seasonal restock needed")
        return {}

    # Cek ketersediaan API key
    from app.helpers.config import settings
    if not settings.gemini_api_key and not settings.groq_api_key:
        print("[STOCK-SEASONAL] LLM keys not configured, skipping per-product seasonal")
        return {}

    # Rangkum data produk untuk LLM
    produk_data = []
    for p in stock_summary[:15]:  # Max 15 produk
        restock = p.get("restock_recommendation", {})
        produk_data.append({
            "product_id": p.get("product_id"),
            "nama": p.get("product_name", "?"),
            "stok_sekarang": p.get("current_stock", 0),
            "restock_normal_min": restock.get("min", 0),
            "restock_normal_max": restock.get("max", 0),
            "avg_daily_sales": p.get("avg_daily_sales", 0),
        })

    # Nama hari raya terdekat untuk konteks
    holiday_names = ", ".join([h["name"] for h in upcoming])

    prompt_data = {
        "tanggal_hari_ini": today.strftime("%d %B %Y (%A)"),
        "hari_raya_terdekat": upcoming,
        "daftar_produk": produk_data,
    }

    prompt = (
        f"Hari raya besar terdekat: {holiday_names}\n\n"
        f"Data produk warung:\n"
        f"```json\n{json.dumps(prompt_data, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Tentukan restock MUSIMAN per produk. Produk yang TIDAK relevan dengan "
        f"hari raya ini, set seasonal_min dan seasonal_max ke null.\n"
        f"Output format: JSON array [{{'product_id': int, 'seasonal_min': int|null, "
        f"'seasonal_max': int|null, 'reason': string}}]"
    )

    try:
        from app.ai.llm_insights import call_llm
        raw_response, source = call_llm(prompt, SEASONAL_RESTOCK_SYSTEM_PROMPT)

        print(f"[STOCK-SEASONAL] Per-product seasonal restock generated ({source})")

        # Parse JSON dengan robust parser
        seasonal_data = _safe_parse_json_array(raw_response)

        if not seasonal_data:
            print("[STOCK-SEASONAL] Could not extract any valid data from LLM response")
            return {}

        # Build mapping product_id -> seasonal_restock
        result = {}
        for item in seasonal_data:
            if not isinstance(item, dict):
                continue
            pid = item.get("product_id")
            s_min = item.get("seasonal_min")
            s_max = item.get("seasonal_max")
            reason = item.get("reason", "")

            # Skip jika null (produk tidak relevan dengan hari raya)
            if pid is None or s_min is None or s_max is None:
                continue

            # Pastikan min <= max
            try:
                s_min, s_max = int(s_min), int(s_max)
            except (ValueError, TypeError):
                continue
            if s_min > s_max:
                s_min, s_max = s_max, s_min

            result[pid] = {
                "min": s_min,
                "max": s_max,
                "label": f"Restock musiman {s_min} - {s_max} item untuk {holiday_names}.",
                "holiday": holiday_names,
                "reason": reason,
            }

        print(f"[STOCK-SEASONAL] {len(result)}/{len(stock_summary)} products need seasonal restock")
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[STOCK-SEASONAL] LLM unavailable ({type(e).__name__}), skipping per-product seasonal")
        return {}


def _safe_parse_json_array(raw: str) -> list[dict]:
    """
    Robust JSON array parser untuk LLM responses yang sering malformed.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Truncated JSON (Unterminated string, missing brackets)
    - Trailing commas
    - Mixed text + JSON
    - Completely broken JSON → fallback regex extraction

    Returns: list of dicts, atau empty list jika semua gagal.
    """
    import json
    import re

    if not raw or not raw.strip():
        return []

    cleaned = raw.strip()

    # Step 1: Hapus markdown code blocks
    if "```" in cleaned:
        # Extract content between ``` markers
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Mungkin ``` pembuka ada tapi penutup terpotong
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # Step 2: Extract JSON array dari text campuran
    # Cari opening bracket [ dan ambil dari situ
    bracket_start = cleaned.find("[")
    if bracket_start >= 0:
        cleaned = cleaned[bracket_start:]
    else:
        # Tidak ada [ sama sekali — coba parse as-is
        pass

    # Step 3: Fix truncated JSON — close unclosed brackets
    # Hitung bracket balance
    open_brackets = cleaned.count("[") - cleaned.count("]")
    open_braces = cleaned.count("{") - cleaned.count("}")

    if open_brackets > 0 or open_braces > 0:
        # JSON terpotong — coba fix

        # Hapus trailing incomplete object/string
        # Pattern: trailing comma + incomplete object
        cleaned = re.sub(r",\s*\{[^}]*$", "", cleaned)  # hapus trailing {incomplete...
        cleaned = re.sub(r",\s*\"[^\"]*$", "", cleaned)  # hapus trailing "incomplete...
        cleaned = re.sub(r",\s*$", "", cleaned)           # hapus trailing comma

        # Re-count after cleanup
        open_braces = cleaned.count("{") - cleaned.count("}")
        open_brackets = cleaned.count("[") - cleaned.count("]")

        # Close remaining brackets
        cleaned += "}" * max(0, open_braces)
        cleaned += "]" * max(0, open_brackets)

    # Step 4: Fix trailing commas (common LLM mistake)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Step 5: Try parsing
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        pass

    # Step 6: Fallback — regex extract individual JSON objects
    print("[STOCK-SEASONAL] Standard JSON parse failed, trying regex extraction...")
    objects = []
    # Match patterns like {"product_id": 1, "seasonal_min": 10, ...}
    pattern = r'\{\s*"product_id"\s*:\s*(\d+)\s*,\s*"seasonal_min"\s*:\s*(\d+|null)\s*,\s*"seasonal_max"\s*:\s*(\d+|null)\s*(?:,\s*"reason"\s*:\s*"([^"]*)")?\s*\}'
    for m in re.finditer(pattern, raw):
        pid = int(m.group(1))
        s_min = None if m.group(2) == "null" else int(m.group(2))
        s_max = None if m.group(3) == "null" else int(m.group(3))
        reason = m.group(4) or ""
        objects.append({
            "product_id": pid,
            "seasonal_min": s_min,
            "seasonal_max": s_max,
            "reason": reason,
        })

    if objects:
        print(f"[STOCK-SEASONAL] Regex extracted {len(objects)} items from malformed JSON")
    else:
        print(f"[STOCK-SEASONAL] Regex extraction also failed. Raw response preview: {raw[:200]}")

    return objects
