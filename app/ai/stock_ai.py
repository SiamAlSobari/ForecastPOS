"""Module AI Forecasting untuk Decision Support System Restock Barang.

Menggunakan scikit-learn Linear Regression untuk memprediksi pola penjualan
harian, lalu mensimulasikan kapan stok akan habis dan memberikan rekomendasi
restock beserta level urgensi (NORMAL / MEDIUM / CRITICAL) dan risk point.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# ─── Constants ────────────────────────────────────────────────────────────────

RISK_MAP = {
    "CRITICAL": {"risk": "CRITICAL", "risk_point": 3},
    "MEDIUM":   {"risk": "MEDIUM",   "risk_point": 2},
    "NORMAL":   {"risk": "NORMAL",   "risk_point": 1},
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
                records.append({
                    "date": pd.to_datetime(trx_date),
                    "type": trx_type,
                    "quantity": int(item["quantity"]),
                })

    if not records:
        return pd.DataFrame(columns=["date", "sold", "purchased", "adjusted"])

    df = pd.DataFrame(records)

    # Pisahkan SALE vs PURCHASE vs ADJUSTMENT
    sales = (
        df[df["type"] == "SALE"]
        .groupby("date")["quantity"]
        .sum()
        .rename("sold")
    )
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
    """Ensemble pintar: Ridge (untuk tren stabil) + Random Forest (jika data cukup)"""
    def __init__(self):
        self.ridge = Ridge(alpha=1.0)
        self.rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.use_rf = False

    def fit(self, X, y):
        self.ridge.fit(X, y)
        if len(y) > 10:
            self.rf.fit(X, y)
            self.use_rf = True

    def predict(self, X):
        p_ridge = self.ridge.predict(X)
        if self.use_rf:
            p_rf = self.rf.predict(X)
            return (p_ridge * 0.7) + (p_rf * 0.3)
        return p_ridge


def train_sales_model(daily: pd.DataFrame) -> tuple[SmartStockEnsemble, float]:
    """
    Melatih model ensemble cerdas (Ridge + RF).
    Menambahkan fitur weekend, payday (tanggal gajian), dan filter outlier.
    Returns: (model, avg_daily_sales)
    """
    if daily.empty or daily["sold"].sum() == 0:
        model = SmartStockEnsemble()
        model.fit(np.array([[0, 0, 0, 0], [1, 1, 0, 0]]), np.array([0, 0]))
        return model, 0.0

    daily = daily.copy()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    
    # Fitur cerdas: Tanggal gajian (biasanya 25 sampai 2)
    daily["day_of_month"] = daily["date"].dt.day
    daily["is_payday"] = ((daily["day_of_month"] >= 25) | (daily["day_of_month"] <= 2)).astype(int)

    # Outlier handling: Mencegah spike wholesale/borongan merusak tren
    y_raw = daily["sold"].values
    if len(y_raw) > 10:
        p95 = np.percentile(y_raw, 95)
        y = np.clip(y_raw, 0, max(p95, 1))
    else:
        y = y_raw

    X = daily[["day_of_week", "day_index", "is_weekend", "is_payday"]].values

    model = SmartStockEnsemble()
    model.fit(X, y)

    avg_daily = float(daily["sold"].mean())
    return model, avg_daily


def predict_future_sales(
    model: SmartStockEnsemble,
    start_date: datetime,
    base_day_index: int,
    avg_daily_sales: float,
    days_ahead: int = 30,
) -> list[dict]:
    """
    Memprediksi penjualan harian untuk N hari ke depan.
    Returns list of {date, predicted_sales}.
    """
    predictions = []
    # Mencegah ekstrapolasi linear yang agresif (terutama jika riwayat data sedikit).
    # Batasi prediksi harian maksimal 1.5x dari rata-rata historis.
    # Untuk produk yang sangat slow-moving, kita batasi maksimal 2x rata-ratanya (tapi mentok 1.0).
    if avg_daily_sales > 1.0:
        max_allowed = avg_daily_sales * 1.5
    else:
        max_allowed = min(avg_daily_sales * 2.0, 1.0)

    for i in range(days_ahead):
        future_date = start_date + timedelta(days=i)
        dow = future_date.weekday()
        day_idx = base_day_index + i
        is_wknd = 1 if dow >= 5 else 0
        dom = future_date.day
        is_payday = 1 if (dom >= 25 or dom <= 2) else 0
        
        # Prediksi menggunakan Ensemble Model
        predicted = model.predict(np.array([[dow, day_idx, is_wknd, is_payday]]))[0]
        
        # Cap prediksi agar tetap realistis dan grounded pada actual sales
        predicted = max(0.0, min(predicted, max_allowed))
        
        predictions.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "day_name": future_date.strftime("%A"),
            "predicted_sales": round(predicted, 1),
        })
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


def determine_urgency(days_until_empty: Optional[int], estimated_empty_date: Optional[str]) -> dict:
    """
    Menentukan level urgensi dan risk point berdasarkan hari sampai stok habis.

    Returns: {
        urgency_level, urgency_description,
        risk, risk_point
    }
    """
    if days_until_empty is None:
        urgency = "NORMAL"
        description = "Stok aman untuk periode prediksi ke depan."
    elif days_until_empty <= 2:
        urgency = "CRITICAL"
        description = (
            f"⚠️ DARURAT! Stok akan HABIS dalam {days_until_empty} hari "
            f"(tanggal {estimated_empty_date}). Segera lakukan restock!"
        )
    elif days_until_empty <= 5:
        urgency = "MEDIUM"
        description = (
            f"⚡ PERHATIAN! Stok akan habis dalam {days_until_empty} hari "
            f"(tanggal {estimated_empty_date}). Rencanakan restock segera."
        )
    else:
        urgency = "NORMAL"
        description = (
            f"✅ Stok masih cukup untuk {days_until_empty} hari "
            f"(sampai tanggal {estimated_empty_date})."
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

        timeline.append({
            "date": pred["date"],
            "day_name": pred["day_name"],
            "predicted_sales": pred["predicted_sales"],
            "remaining_stock": stock,
        })

        if stock <= 0 and days_until_empty is None:
            days_until_empty = i + 1
            estimated_empty_date = pred["date"]

    # ─── Tentukan Level Urgensi + Risk ────────────────────────────────────
    urgency_info = determine_urgency(days_until_empty, estimated_empty_date)

    # ─── Rekomendasi Jumlah Restock ───────────────────────────────────────
    # Target: stok cukup untuk 7 hari berdasarkan rata-rata prediksi
    avg_predicted = np.mean([p["predicted_sales"] for p in future_predictions[:7]])
    optimal_stock_7_days = int(np.ceil(avg_predicted * 7))
    restock_qty = max(0, optimal_stock_7_days - current_stock)

    return {
        "current_stock": current_stock,
        "days_until_empty": days_until_empty,
        "estimated_empty_date": estimated_empty_date,
        **urgency_info,
        "restock_recommendation": {
            "recommended_quantity": restock_qty,
            "target_days_coverage": 7,
            "avg_daily_predicted_sales": round(avg_predicted, 1),
            "optimal_stock_for_7_days": optimal_stock_7_days,
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
    model, avg_daily = train_sales_model(daily)

    # 4. Hitung stok saat ini
    # Prioritas: override > stock dari nested product.stocks > hitung dari data
    if current_stock_override is not None:
        current_stock = current_stock_override
    elif product_info["current_stock_from_data"] is not None:
        current_stock = product_info["current_stock_from_data"]
    else:
        current_stock = compute_current_stock(daily)

    # 5. Prediksi ke depan
    last_date = daily["date"].max()
    start_forecast = last_date + timedelta(days=1)
    base_day_index = (last_date - daily["date"].min()).days + 1

    predictions = predict_future_sales(
        model, start_forecast, base_day_index, avg_daily, forecast_days
    )

    # 6. Simulasi & analisis (sudah termasuk risk + risk_point)
    result = simulate_stock_depletion(current_stock, predictions)

    # 7. Tambahkan metadata produk
    result["product_id"] = product_id
    result["product_name"] = product_info["product_name"]
    result["product_price"] = product_info["product_price"]
    result["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["data_range"] = {
        "from": daily["date"].min().strftime("%Y-%m-%d"),
        "to": daily["date"].max().strftime("%Y-%m-%d"),
        "total_days": len(daily),
    }
    total_adjusted = int(daily["adjusted"].sum()) if "adjusted" in daily.columns else 0
    result["historical_stats"] = {
        "avg_daily_sales": round(avg_daily, 1),
        "total_sold": int(daily["sold"].sum()),
        "total_purchased": int(daily["purchased"].sum()),
        "total_adjusted": total_adjusted,
        "max_daily_sales": int(daily["sold"].max()),
        "min_daily_sales": int(daily["sold"].min()),
    }

    return result
