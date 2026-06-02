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
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────

RISK_MAP = {
    "CRITICAL": {"risk": "CRITICAL", "risk_point": 3},
    "MEDIUM": {"risk": "MEDIUM", "risk_point": 2},
    "NORMAL": {"risk": "NORMAL", "risk_point": 1},
}


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


# ─── Model: Prediksi Penjualan Harian ────────────────────────────────────────


class SmartStockEnsemble:
    """Super AI: Ridge (baseline) + Random Forest + HistGradientBoosting.

    Accuracy dihitung menggunakan kombinasi:
    - SMAPE (Symmetric MAPE) — stabil saat y=0
    - R² Score — seberapa baik model menjelaskan variansi data
    - Cross-validation untuk menghindari overfitting
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=0.5)
        self.rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        # HistGBR: algoritma ala LightGBM bawaan Sklearn
        self.hgb = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=8,
            learning_rate=0.03,
            l2_regularization=0.05,
            min_samples_leaf=3,
            random_state=42,
        )
        self.use_trees = False

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y)
        if len(y) > 14:
            self.rf.fit(X_scaled, y)
            self.hgb.fit(X_scaled, y)
            self.use_trees = True
        self.X_scaled = X_scaled  # Cache untuk metrics

    def predict(self, X):
        if hasattr(self, 'X_scaled') and X is self.X_scaled:
            X_scaled = X
        else:
            X_scaled = self.scaler.transform(X)
        p_ridge = self.ridge.predict(X_scaled)
        if self.use_trees:
            p_rf = self.rf.predict(X_scaled)
            p_hgb = self.hgb.predict(X_scaled)
            # Bobot: 50% HistGBR (paling pintar), 35% RF (paling stabil), 15% Ridge (garis aman)
            return np.maximum(0, (p_hgb * 0.50) + (p_rf * 0.35) + (p_ridge * 0.15))
        return np.maximum(0, p_ridge)

    def predict_scaled(self, X_scaled):
        """Predict langsung dari data yang sudah di-scale."""
        p_ridge = self.ridge.predict(X_scaled)
        if self.use_trees:
            p_rf = self.rf.predict(X_scaled)
            p_hgb = self.hgb.predict(X_scaled)
            return np.maximum(0, (p_hgb * 0.50) + (p_rf * 0.35) + (p_ridge * 0.15))
        return np.maximum(0, p_ridge)


def train_sales_model(daily: pd.DataFrame) -> tuple[SmartStockEnsemble, float, float]:
    """
    Melatih model ensemble cerdas (Ridge + RF + HGB).
    Menambahkan fitur weekend, payday, awal bulan, dan filter outlier.

    Accuracy dihitung menggunakan SMAPE + R² (bukan MAPE yang rusak saat y=0).

    Returns: (model, avg_daily_sales, accuracy_percent)
    """
    if daily.empty or daily["sold"].sum() == 0:
        model = SmartStockEnsemble()
        dummy_X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0]])
        model.fit(dummy_X, np.array([0, 0]))
        return model, 0.0, 0.0

    daily = daily.copy()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    # Fitur cerdas: Tanggal gajian (biasanya 25 sampai 2)
    daily["day_of_month"] = daily["date"].dt.day
    daily["is_payday"] = (
        (daily["day_of_month"] >= 25) | (daily["day_of_month"] <= 2)
    ).astype(int)

    # Fitur super cerdas: Awal bulan (dompet masih tebal) dan Tengah bulan (kritis)
    daily["is_start_month"] = (daily["day_of_month"] <= 5).astype(int)
    daily["is_mid_month"] = (
        (daily["day_of_month"] > 10) & (daily["day_of_month"] <= 20)
    ).astype(int)

    # ─── Enhanced Features untuk Akurasi Tinggi ──────────────────────────
    # Week of year (menangkap seasonality mingguan)
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)

    # Rolling average penjualan (smoothing tren)
    daily = daily.sort_values("date")
    daily["rolling_avg_sold"] = (
        daily["sold"].rolling(5, min_periods=1).mean()
    )

    # Log day index (mengurangi efek extrapolasi linear agresif)
    daily["log_day_index"] = np.log1p(daily["day_index"])

    # Outlier handling: IQR-based (lebih robust dari percentile sederhana)
    y_raw = daily["sold"].values
    if len(y_raw) > 10:
        non_zero = y_raw[y_raw > 0]
        if len(non_zero) > 5:
            q1 = np.percentile(non_zero, 25)
            q3 = np.percentile(non_zero, 75)
            iqr = q3 - q1
            upper_bound = q3 + 2.5 * iqr
            y = np.clip(y_raw, 0, max(upper_bound, np.percentile(y_raw, 97)))
        else:
            p95 = np.percentile(y_raw, 95)
            y = np.clip(y_raw, 0, max(p95, 1))
    else:
        y = y_raw

    X = daily[
        [
            "day_of_week",
            "day_index",
            "is_weekend",
            "is_payday",
            "is_start_month",
            "is_mid_month",
            "week_of_year",
            "rolling_avg_sold",
            "log_day_index",
        ]
    ].values

    model = SmartStockEnsemble()
    model.fit(X, y)

    # ─── Akurasi: SMAPE + R² (bukan MAPE yang rusak) ─────────────────────
    n_samples = len(y)
    if n_samples >= 20 and model.use_trees:
        n_folds = min(5, n_samples // 5)
        try:
            # CV pada data yang sudah di-scale
            from sklearn.base import clone, BaseEstimator, RegressorMixin

            class _EnsembleWrapper(BaseEstimator, RegressorMixin):
                """Wrapper untuk cross_val_predict."""
                def __init__(self):
                    self.inner = SmartStockEnsemble()
                def fit(self, X, y):
                    self.inner.fit(X, y)
                    return self
                def predict(self, X):
                    return self.inner.predict(X)

            y_pred = cross_val_predict(_EnsembleWrapper(), X, y, cv=n_folds)
        except Exception:
            y_pred = model.predict_scaled(model.X_scaled)
    else:
        y_pred = model.predict_scaled(model.X_scaled)

    y_pred = np.maximum(0, y_pred)

    # SMAPE: Symmetric MAPE — stabil saat y=0
    denominator = np.abs(y) + np.abs(y_pred)
    mask = denominator > 0
    if mask.sum() > 0:
        smape = np.mean(np.abs(y[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = 0.0

    # R² Score
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = max(0, r2)

    # Gabungan: 60% SMAPE-based + 40% R²-based
    smape_accuracy = max(0, min(100, 100 - smape))
    r2_accuracy = r2 * 100
    accuracy_pct = round((smape_accuracy * 0.6) + (r2_accuracy * 0.4), 2)

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
    Returns list of {date, predicted_sales}.
    """
    predictions = []
    # Mencegah ekstrapolasi linear yang agresif
    if avg_daily_sales > 1.0:
        max_allowed = avg_daily_sales * 1.5
    else:
        max_allowed = min(avg_daily_sales * 2.0, 1.0)

    # Pre-compute rolling avg dari historical data
    if daily_df is not None and not daily_df.empty:
        last_rolling_avg = float(
            daily_df["sold"].rolling(5, min_periods=1).mean().iloc[-1]
        )
    else:
        last_rolling_avg = avg_daily_sales

    for i in range(days_ahead):
        future_date = start_date + timedelta(days=i)
        dow = future_date.weekday()
        day_idx = base_day_index + i
        is_wknd = 1 if dow >= 5 else 0
        dom = future_date.day
        is_payday = 1 if (dom >= 25 or dom <= 2) else 0
        is_start_month = 1 if dom <= 5 else 0
        is_mid_month = 1 if 10 < dom <= 20 else 0
        week_of_year = future_date.isocalendar()[1]
        log_day_idx = np.log1p(day_idx)

        # Prediksi menggunakan Ensemble Model
        predicted = model.predict(
            np.array([[dow, day_idx, is_wknd, is_payday, is_start_month, is_mid_month,
                       week_of_year, last_rolling_avg, log_day_idx]])
        )[0]

        # Cap prediksi agar tetap realistis dan grounded pada actual sales
        predicted = max(0.0, min(predicted, max_allowed))

        # Update rolling avg untuk prediksi berikutnya
        last_rolling_avg = last_rolling_avg * 0.8 + predicted * 0.2

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

        # Parse JSON dari response LLM
        # Bersihkan markdown code block jika ada
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Hapus ```json ... ```
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        seasonal_data = json.loads(cleaned)

        # Build mapping product_id -> seasonal_restock
        result = {}
        for item in seasonal_data:
            pid = item.get("product_id")
            s_min = item.get("seasonal_min")
            s_max = item.get("seasonal_max")
            reason = item.get("reason", "")

            # Skip jika null (produk tidak relevan dengan hari raya)
            if pid is None or s_min is None or s_max is None:
                continue

            # Pastikan min <= max
            s_min, s_max = int(s_min), int(s_max)
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

    except json.JSONDecodeError as e:
        print(f"[STOCK-SEASONAL] Failed to parse LLM response as JSON: {e}")
        return {}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[STOCK-SEASONAL] LLM unavailable ({type(e).__name__}), skipping per-product seasonal")
        return {}

