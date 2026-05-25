"""
Controller untuk endpoint Busy Hour Prediction System.
Format data input SAMA PERSIS dengan endpoint predict/restock/summary.
"""

import json
import os

from fastapi import APIRouter
from app.api.models.predict_model import SummaryRequest
from app.api.services.busy_hour_service import get_busy_hour_analysis_async

busy_hour_controller = APIRouter()

# ─── Helper: Load data dari file ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
TRX_PATH = os.path.join(DATA_DIR, "trx.json")


def _load_transactions_from_file() -> list[dict]:
    """Membaca data transaksi dari file trx.json."""
    with open(TRX_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("data", raw) if isinstance(raw, dict) else raw


# ─── Endpoints ────────────────────────────────────────────────────────────────

@busy_hour_controller.post("/busy-hours")
async def predict_busy_hours(body: SummaryRequest = SummaryRequest()):
    """
    Prediksi jam sibuk untuk 14 hari ke depan.
    ASYNC: Tidak blocking event loop, optimal untuk cron job.

    Input data format SAMA dengan /api/predict/restock/summary.
    Jika data kosong, otomatis pakai data dari trx.json.

    Returns:
    - Prediksi jam sibuk per hari (hourly breakdown)
    - Prediksi produk & revenue per jam
    - Peak hour rankings & summary
    """
    if body.data:
        transactions = [trx.model_dump() for trx in body.data]
    else:
        transactions = _load_transactions_from_file()

    results = await get_busy_hour_analysis_async(
        transactions=transactions,
        forecast_days=body.forecast_days,
    )
    return {
        "message": "Prediksi jam sibuk berhasil",
        "data": results,
    }
