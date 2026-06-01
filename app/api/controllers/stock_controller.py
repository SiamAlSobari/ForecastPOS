"""
Controller untuk endpoint Decision Support System Restock Barang.

Fitur:
- Ringkasan restock semua produk (ML-based)
- Seasonal insight: LLM overlay untuk prediksi musiman/hari raya (opsional)
  Aktifkan dengan query param ?include_seasonal=true

Seasonal insight adalah fitur "jembatan" antara:
- ML yang buta kalender (hanya lihat tren data historis)
- Insting musiman pedagang yang biasanya BENAR saat mendekati hari raya

LLM akan meng-override prediksi ML normal dengan nasehat:
"Meski data bilang stok aman, tapi 3 hari lagi Lebaran! Gas restock 2x lipat!"
"""

import json
import os

from fastapi import APIRouter, Query
from app.api.models.predict_model import SummaryRequest
from app.api.services.stock_service import get_all_products_summary

stock_controller = APIRouter()


# ─── Helper: Load data dari file ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
TRX_PATH = os.path.join(DATA_DIR, "trx.json")


def _load_transactions_from_file() -> list[dict]:
    """Membaca data transaksi dari file trx.json."""
    with open(TRX_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("data", raw) if isinstance(raw, dict) else raw


# ─── Endpoints ────────────────────────────────────────────────────────────────

@stock_controller.post("/restock/summary")
def restock_summary(
    body: SummaryRequest = SummaryRequest(),
    include_seasonal: bool = Query(
        default=False,
        description="Sertakan nasehat restock musiman dari LLM? (membutuhkan API key LLM)"
    ),
):
    """
    Ringkasan urgensi restock untuk semua produk.
    Jika data kosong, otomatis pakai dummy data dari trx.json.

    Query Params:
    - include_seasonal=true: Tambahkan nasehat LLM tentang hari raya/musim.
      LLM akan memberikan overlay prediksi musiman yang memvalidasi
      insting pedagang saat mendekati hari raya.

    Returns:
    - products: List produk dengan urgensi restock
    - seasonal_insight: Nasehat musiman dari LLM (jika include_seasonal=true)
    """
    if body.data:
        transactions = [trx.model_dump() for trx in body.data]
    else:
        transactions = _load_transactions_from_file()

    results = get_all_products_summary(
        transactions=transactions,
        forecast_days=body.forecast_days,
        include_seasonal=include_seasonal,
    )

    return {
        "message": "Ringkasan restock semua produk",
        "total_products": len(results["products"]),
        "data": results["products"],
        "seasonal_insight": results["seasonal_insight"],
    }
