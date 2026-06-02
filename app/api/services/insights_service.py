"""
Service layer untuk LLM Insights — Portofolio Bisnis Mingguan.

Merangkum performa bisnis warung 7 HARI KE BELAKANG, lalu mengirimkan
ke LLM untuk mendapatkan analisis portofolio dan nasehat bisnis.

INI BUKAN MODUL PREDIKSI! Ini murni laporan retrospektif.
Prediksi musiman / holiday awareness ada di stock_service.py.

Dirancang untuk dipanggil via Laravel Cronjob setiap 7 hari.
Error dari LLM (LLMConfigError, LLMServiceError) akan di-propagate
ke controller agar bisa ditangani dengan HTTP status code yang tepat.
"""

from app.ai.llm_insights import generate_portfolio_insights
from app.api.cache import insights_cache


def get_portfolio_insights(
    transactions: list[dict],
) -> dict:
    cached = insights_cache.get(transactions)
    if cached is not None:
        return cached

    result = generate_portfolio_insights(transactions)
    insights_cache.set(result, transactions)
    return result
