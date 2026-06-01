"""
Controller untuk endpoint Busy Hour Prediction System.
Format data input SAMA PERSIS dengan endpoint predict/restock/summary.
"""

from fastapi import APIRouter
from app.api.models.predict_model import SummaryRequest
from app.api.services.busy_hour_service import get_busy_hour_analysis

busy_hour_controller = APIRouter()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@busy_hour_controller.post("/busy-hours")
def predict_busy_hours(body: SummaryRequest):
    """
    Prediksi jam sibuk untuk 14 hari ke depan.

    Input data format SAMA dengan /api/predict/restock/summary.

    Returns:
    - Prediksi jam sibuk per hari (hourly breakdown)
    - Prediksi produk & revenue per jam
    - Peak hour rankings & summary
    """
    transactions = [trx.model_dump() for trx in body.data]

    results = get_busy_hour_analysis(
        transactions=transactions,
        forecast_days=body.forecast_days,
    )
    return {
        "message": "Prediksi jam sibuk berhasil",
        "data": results,
    }
