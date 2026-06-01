"""
Service layer untuk Busy Hour Prediction System.
Menghubungkan AI engine busy_hour_ai dengan API controller.
"""

from app.ai.busy_hour_ai import analyze_busy_hours, normalize_transactions


def get_busy_hour_analysis(
    transactions: list[dict],
    forecast_days: int = 14,
) -> dict:
    """
    Mengambil analisis prediksi jam sibuk.

    Args:
        transactions: List data transaksi (format sama dengan predict stock).
        forecast_days: Jumlah hari prediksi ke depan (default 14).

    Returns:
        Dictionary berisi analisis lengkap: hourly forecasts, product predictions,
        revenue forecasts, model accuracy, dan summary.
    """
    return analyze_busy_hours(
        transactions=transactions,
        forecast_days=forecast_days,
    )
