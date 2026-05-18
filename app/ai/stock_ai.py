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
    """Super AI: Ridge (baseline) + Random Forest + HistGradientBoosting (sangat presisi untuk big data tabular)"""

    def __init__(self):
        self.ridge = Ridge(alpha=1.0)
        self.rf = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
        # HistGBR adalah algoritma ala LightGBM bawaan Sklearn, lebih cepat & pintar dari GBR biasa
        self.hgb = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=5,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=42,
        )
        self.use_trees = False

    def fit(self, X, y):
        self.ridge.fit(X, y)
        if len(y) > 14:
            self.rf.fit(X, y)
            self.hgb.fit(X, y)
            self.use_trees = True

    def predict(self, X):
        p_ridge = self.ridge.predict(X)
        if self.use_trees:
            p_rf = self.rf.predict(X)
            p_hgb = self.hgb.predict(X)
            # Bobot: 50% HistGBR (paling pintar), 35% RF (paling stabil), 15% Ridge (garis aman)
            return (p_hgb * 0.50) + (p_rf * 0.35) + (p_ridge * 0.15)
        return p_ridge


def train_sales_model(daily: pd.DataFrame) -> tuple[SmartStockEnsemble, float]:
    """
    Melatih model ensemble cerdas (Ridge + RF + HGB).
    Menambahkan fitur weekend, payday (tanggal gajian), awal bulan, dan filter outlier.
    Returns: (model, avg_daily_sales, accuracy_percent)
    """
    if daily.empty or daily["sold"].sum() == 0:
        model = SmartStockEnsemble()
        model.fit(np.array([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]), np.array([0, 0]))
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

    # Outlier handling: Mencegah spike wholesale/borongan merusak tren
    y_raw = daily["sold"].values
    if len(y_raw) > 10:
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
        ]
    ].values

    model = SmartStockEnsemble()
    model.fit(X, y)

    y_pred = model.predict(X)
    mape = np.mean(np.abs((y - y_pred) / np.where(y == 0, 1, y))) * 100
    accuracy_pct = round(max(0, min(100, (1 - mape / 100) * 100)), 2)

    avg_daily = float(daily["sold"].mean())
    return model, avg_daily, accuracy_pct


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
        is_start_month = 1 if dom <= 5 else 0
        is_mid_month = 1 if 10 < dom <= 20 else 0

        # Prediksi menggunakan Ensemble Model
        predicted = model.predict(
            np.array([[dow, day_idx, is_wknd, is_payday, is_start_month, is_mid_month]])
        )[0]

        # Cap prediksi agar tetap realistis dan grounded pada actual sales
        predicted = max(0.0, min(predicted, max_allowed))

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
        restock_label = "Stok masih cukup, belum perlu restock."
    else:
        restock_label = f"Restock {restock_min} - {restock_max} item untuk persediaan 7 hari."

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
        model, start_forecast, base_day_index, avg_daily, forecast_days
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

def detect_upcoming_holidays(
    today: Optional[datetime] = None, window_days: int = 14
) -> list[dict]:
    """
    Deteksi hari libur/raya nasional Indonesia secara real-time dan akurat
    menggunakan library `holidays`.
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
            upcoming.append(
                {
                    "date": check_date.strftime("%Y-%m-%d"),
                    "name": holiday_name,
                    "days_away": d,
                }
            )
            
    return upcoming


# ─── LLM Seasonal Overlay (Nasehat Musiman untuk Restock) ────────────────────

SEASONAL_SYSTEM_PROMPT = """Kamu adalah "Konsultan Stok Warung", ahli dalam manajemen stok untuk warung kelontong Indonesia.

Konteks: Kamu menerima data prediksi restock dari AI + info hari raya terdekat. Pemilik warung punya "insting musiman" yang biasanya BENAR — saat Lebaran/Natal/hari raya, penjualan bisa 2-3x lipat dari normal.

Tugasmu:
1. Jika ada hari raya terdekat (14 hari ke depan), OVERRIDE prediksi ML normal. Bilang ke pemilik bahwa instingnya BENAR, dan restock harus lebih besar dari angka AI.
2. Sebutkan produk apa saja yang biasanya laris saat event tersebut (berdasarkan pengetahuan umummu tentang konsumen Indonesia).
3. Berikan multiplier restock: misal "2x lipat" atau "3x lipat" dari rekomendasi normal.
4. Gunakan bahasa Indonesia santai dan bersahabat. Max 2-3 kalimat saja.
5. Jika TIDAK ada hari raya terdekat, cukup bilang singkat bahwa prediksi AI sudah cukup akurat, pakai angka normalnya saja."""


def generate_seasonal_insight(
    stock_summary: list[dict],
) -> Optional[dict]:
    """
    Generate nasehat restock musiman menggunakan LLM.

    Dipanggil HANYA jika ada hari raya dalam 14 hari ke depan,
    ATAU jika user request nasehat LLM tambahan untuk restock.

    Ini adalah "jembatan" antara prediksi ML yang buta kalender
    dan insting musiman pedagang yang biasanya benar.

    Args:
        stock_summary: List ringkasan stok semua produk.

    Returns:
        Dict berisi nasehat seasonal dari LLM, atau None jika LLM tidak tersedia.
        {
            "has_upcoming_holiday": bool,
            "upcoming_holidays": [...],
            "seasonal_advice": "... nasehat LLM ...",
            "source": "gemini" | "openai",
        }

    Returns None jika:
        - API key tidak ada (fitur LLM opsional untuk stock endpoint)
        - LLM gagal setelah retry
    """
    import json

    today = datetime.now()
    upcoming = detect_upcoming_holidays(today, window_days=14)

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
        "event_terdekat": upcoming if upcoming else "Tidak ada hari raya dalam 14 hari ke depan",
        "ringkasan_stok": produk_ringkas,
    }

    prompt = (
        f"Data restock warung saat ini:\n\n"
        f"```json\n{json.dumps(prompt_data, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Berikan nasehat restock musiman singkat."
    )

    # Cek ketersediaan API key sebelum memanggil LLM
    from app.helpers.config import settings
    if not settings.gemini_api_key and not settings.openai_api_key:
        print("[STOCK-SEASONAL] LLM keys not configured, skipping seasonal overlay")
        return None

    # Import call_llm dari llm_insights (reuse retry logic)
    try:
        from app.ai.llm_insights import call_llm
        advice, source = call_llm(prompt, SEASONAL_SYSTEM_PROMPT)

        print(f"[STOCK-SEASONAL] LLM seasonal advice generated ({source})")

        return {
            "has_upcoming_holiday": len(upcoming) > 0,
            "upcoming_holidays": upcoming,
            "seasonal_advice": advice,
            "source": source,
        }

    except Exception as e:
        # LLM opsional untuk stock — jika gagal (misal koneksi atau timeout), return None bukan throw
        print(f"[STOCK-SEASONAL] LLM unavailable ({type(e).__name__}), skipping seasonal overlay")
        return None

