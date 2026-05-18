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


def get_portfolio_insights(
    transactions: list[dict],
) -> dict:
    """
    Mengambil laporan portofolio mingguan dari LLM.

    Flow:
    1. Kirim data transaksi mentah ke generate_portfolio_insights()
    2. Module akan merangkum 7 hari terakhir + kirim ke LLM

    Ini TIDAK menjalankan prediksi ML (busy hour / stock).
    Data yang dirangkum murni retrospektif (backward-looking).

    Args:
        transactions: List data transaksi dari Laravel.

    Returns:
        Dictionary berisi analisis portofolio, summary data, dan metadata.

    Raises:
        LLMConfigError: Jika tidak ada API key LLM.
        LLMServiceError: Jika semua LLM provider gagal.
    """
    # LLMConfigError dan LLMServiceError akan propagate ke controller
    return generate_portfolio_insights(transactions)
