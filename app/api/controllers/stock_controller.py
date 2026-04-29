"""
Controller untuk endpoint Decision Support System Restock Barang.
"""

import json
import os

from fastapi import APIRouter
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
def restock_summary(body: SummaryRequest = SummaryRequest()):
    """
    Ringkasan urgensi restock untuk semua produk.
    Jika data kosong, otomatis pakai dummy data dari trx.json.
    """
    if body.data:
        transactions = [trx.model_dump() for trx in body.data]
    else:
        transactions = _load_transactions_from_file()

    results = get_all_products_summary(
        transactions=transactions,
        forecast_days=body.forecast_days,
    )
    return {
        "message": "Ringkasan restock semua produk",
        "total_products": len(results),
        "data": results,
    }
