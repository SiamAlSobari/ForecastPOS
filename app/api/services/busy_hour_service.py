"""
Service layer untuk Busy Hour Prediction System.
Menghubungkan AI engine busy_hour_ai dengan API controller.

Concurrency support:
- Sync version (get_busy_hour_analysis) untuk backward compatibility
- Async version (get_busy_hour_analysis_async) yang offload CPU-bound
  ML training ke ThreadPoolExecutor agar tidak blocking event loop
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.ai.busy_hour_ai import analyze_busy_hours, normalize_transactions

_executor = ThreadPoolExecutor(max_workers=None)


def get_busy_hour_analysis(
    transactions: list[dict],
    forecast_days: int = 14,
) -> dict:
    """
    Mengambil analisis prediksi jam sibuk (sync version).

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


async def get_busy_hour_analysis_async(
    transactions: list[dict],
    forecast_days: int = 14,
) -> dict:
    """
    Mengambil analisis prediksi jam sibuk (async version).
    CPU-bound ML training dijalankan di thread pool.

    Args:
        transactions: List data transaksi (format sama dengan predict stock).
        forecast_days: Jumlah hari prediksi ke depan (default 14).

    Returns:
        Dictionary berisi analisis lengkap.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        analyze_busy_hours,
        transactions,
        forecast_days,
    )
