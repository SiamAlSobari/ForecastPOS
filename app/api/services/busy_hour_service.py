"""
Service layer untuk Busy Hour Prediction System.
Menghubungkan AI engine busy_hour_ai dengan API controller.
"""

from app.ai.busy_hour_ai import analyze_busy_hours, normalize_transactions
from app.api.cache import busy_hour_cache


def get_busy_hour_analysis(
    transactions: list[dict],
    forecast_days: int = 14,
) -> dict:
    cached = busy_hour_cache.get(transactions, forecast_days)
    if cached is not None:
        return cached

    result = analyze_busy_hours(
        transactions=transactions,
        forecast_days=forecast_days,
    )
    busy_hour_cache.set(result, transactions, forecast_days)
    return result
