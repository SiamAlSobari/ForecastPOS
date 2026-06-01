"""
Controller untuk endpoint Portofolio Bisnis Mingguan (LLM Insights).

Endpoint ini dirancang untuk dipanggil oleh Laravel Task Scheduler
setiap 7 hari sekali. Hasilnya berisi ringkasan performa bisnis
7 hari KE BELAKANG (retrospektif) yang disimpan di tabel `ai_insights`.

INI BUKAN PREDIKSI! Untuk prediksi musiman/holiday, lihat stock_controller.

Alur:
    Laravel Cronjob (7 hari) → POST /api/insights/generate
    → Python merangkum data 7 hari + kirim ke LLM → Response JSON
    → Laravel simpan di DB → User buka app → Laravel query DB → instan!

Error Handling:
    - Tidak ada API key → 503 + pesan jelas
    - LLM gagal setelah retry → 502 + pesan jelas
    - Data kosong / error ML → 400 + pesan jelas
"""

import json
import os

from fastapi import APIRouter, HTTPException
from app.api.models.predict_model import SummaryRequest
from app.api.services.insights_service import get_portfolio_insights
from app.ai.llm_insights import LLMConfigError, LLMServiceError

insights_controller = APIRouter()

# ─── Helper: Load data dari file ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
TRX_PATH = os.path.join(DATA_DIR, "trx.json")


def _load_transactions_from_file() -> list[dict]:
    """Membaca data transaksi dari file trx.json."""
    with open(TRX_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("data", raw) if isinstance(raw, dict) else raw


# ─── Endpoints ────────────────────────────────────────────────────────────────

@insights_controller.post("/generate")
def generate_weekly_portfolio(body: SummaryRequest = SummaryRequest()):
    """
    Generate Portofolio Bisnis Mingguan — dipanggil oleh Laravel Cronjob.

    Merangkum performa bisnis 7 HARI KE BELAKANG (retrospektif):
    - Total omset dan transaksi minggu ini
    - Produk terlaris (bintang warung)
    - Produk kurang laku (dead stock)
    - Hari paling ramai vs sepi
    - Nasehat bisnis dari LLM berdasarkan data aktual

    INI BUKAN PREDIKSI! Untuk prediksi musiman, gunakan /api/predict/restock/summary.

    Error Responses:
    - 503: API key LLM tidak dikonfigurasi di .env
    - 502: Semua LLM provider gagal setelah retry (Gemini 2x, OpenAI 2x)
    - 500: Error tidak terduga

    Returns:
    - insight: Analisis portofolio dari LLM (teks).
    - summary: Data ringkas performa minggu lalu.
    - source: Provider yang berhasil ("gemini" atau "openai").
    - generated_at: Waktu insight dibuat.
    - valid_until: Insight berlaku sampai kapan (7 hari dari generated_at).
    """
    try:
        if body.data:
            transactions = [trx.model_dump() for trx in body.data]
        else:
            transactions = _load_transactions_from_file()

        results = get_portfolio_insights(
            transactions=transactions,
        )

        return {
            "message": "Portofolio bisnis mingguan berhasil dibuat",
            "data": results,
        }

    except LLMConfigError as e:
        # Tidak ada API key → 503 Service Unavailable
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM_CONFIG_ERROR",
                "message": str(e),
                "hint": "Tambahkan GEMINI_API_KEY dan/atau OPENAI_API_KEY di file .env",
            },
        )

    except LLMServiceError as e:
        # Semua LLM gagal setelah retry → 502 Bad Gateway
        raise HTTPException(
            status_code=502,
            detail={
                "error": "LLM_SERVICE_ERROR",
                "message": str(e),
                "hint": "Periksa API key, koneksi internet, atau status layanan LLM provider.",
            },
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "DATA_NOT_FOUND",
                "message": "File trx.json tidak ditemukan dan tidak ada data di request body.",
                "hint": "Kirim data transaksi via request body atau pastikan file trx.json tersedia.",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Terjadi error: {type(e).__name__}: {str(e)}",
            },
        )
