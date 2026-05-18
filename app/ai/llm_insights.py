"""Modul LLM Insights — "Portofolio Bisnis Mingguan".

Merangkum performa bisnis warung 7 HARI KE BELAKANG, lalu mengirimkan
ke LLM untuk mendapatkan analisis portofolio dan nasehat bisnis.

INI BUKAN MODUL PREDIKSI! Ini murni laporan retrospektif (backward-looking).
Prediksi musiman / holiday awareness ada di `stock_ai.py`.

Fitur:
- Micro-summarization (hemat token): hanya kirim high-level stats ke LLM.
- Retrospektif: merangkum penjualan, produk terlaris, dead stock, dll.
- Persona: LLM berperan sebagai rekan bisnis warung yang ramah.
- Retry & Fallback: Gemini (2x retry) → OpenAI (2x retry) → throw error.
- Designed untuk dipanggil via Laravel Cronjob setiap 7 hari sekali.

Alur Eksekusi (dari sisi Laravel):
    1. Laravel Task Scheduler (setiap 7 hari) → HTTP POST ke Python API
    2. Python merangkum data historis 7 hari → micro-summarize → kirim ke LLM
    3. Hasil LLM dikembalikan ke Laravel → disimpan di tabel `ai_insights`
    4. User buka app → Laravel query `ai_insights` → instan, 0 token terpakai
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from app.helpers.config import settings

# ─── Custom Exceptions ────────────────────────────────────────────────────────


class LLMConfigError(Exception):
    """Tidak ada API key LLM yang dikonfigurasi di .env."""
    pass


class LLMServiceError(Exception):
    """Semua provider LLM gagal setelah retry."""
    pass


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_RETRIES = 2          # Maksimal retry per provider
RETRY_DELAY_SEC = 2      # Jeda antar retry (seconds)


# ─── Micro-Summarization (Retrospektif 7 Hari) ──────────────────────────────


def build_portfolio_summary(
    transactions: list[dict],
) -> dict:
    """
    Merangkum performa bisnis warung 7 HARI KE BELAKANG.

    Data yang dirangkum (retrospektif):
    - Total omset minggu ini
    - Total transaksi
    - Produk terlaris (bintang warung)
    - Produk yang tidak laku (dead stock)
    - Rata-rata transaksi per hari
    - Perbandingan hari ramai vs sepi

    Args:
        transactions: List data transaksi dari Laravel.

    Returns:
        Dict summary ringan yang siap dikirim ke LLM.
    """
    today = datetime.now()
    week_ago = today - timedelta(days=7)

    # ─── Filter SALE transactions dalam 7 hari terakhir ───────────────────
    sales_records = []
    product_sales = {}  # {product_name: {qty, revenue, product_id}}
    daily_revenue = {}  # {date_str: total_revenue}
    daily_trx_count = {}  # {date_str: count}

    for trx in transactions:
        if trx.get("trx_type") != "SALE":
            continue

        trx_date = pd.to_datetime(trx["trx_date"])
        # Hanya 7 hari ke belakang
        if trx_date < pd.to_datetime(week_ago.strftime("%Y-%m-%d")):
            continue

        date_str = trx_date.strftime("%Y-%m-%d")
        total_amount = float(trx.get("total_amount", "0").replace(",", ""))

        daily_revenue[date_str] = daily_revenue.get(date_str, 0) + total_amount
        daily_trx_count[date_str] = daily_trx_count.get(date_str, 0) + 1

        for item in trx.get("items", []):
            qty = int(item.get("quantity", 0))
            product = item.get("product", {})
            pname = product.get("name", f"Product #{item.get('product_id', '?')}") if isinstance(product, dict) else f"Product #{item.get('product_id', '?')}"
            pprice = float(product.get("price", "0")) if isinstance(product, dict) else 0

            if pname not in product_sales:
                product_sales[pname] = {"qty": 0, "revenue": 0}
            product_sales[pname]["qty"] += qty
            product_sales[pname]["revenue"] += qty * pprice

    # ─── Hitung statistik ─────────────────────────────────────────────────
    total_omset = sum(daily_revenue.values())
    total_trx = sum(daily_trx_count.values())
    avg_trx_per_day = round(total_trx / 7, 1) if total_trx > 0 else 0
    avg_omset_per_day = round(total_omset / 7, 0) if total_omset > 0 else 0

    # Hari paling ramai dan paling sepi
    hari_ramai = max(daily_revenue, key=daily_revenue.get) if daily_revenue else None
    hari_sepi = min(daily_revenue, key=daily_revenue.get) if daily_revenue else None

    # Produk terlaris (top 5)
    sorted_products = sorted(
        product_sales.items(), key=lambda x: x[1]["qty"], reverse=True
    )
    bintang_warung = [
        {"nama": name, "terjual": data["qty"], "omset": round(data["revenue"], 0)}
        for name, data in sorted_products[:5]
    ]

    # Dead stock: produk yang ada di data tapi terjual sangat sedikit
    dead_stock = [
        name for name, data in sorted_products
        if data["qty"] <= 1
    ]

    # ─── Build final summary ──────────────────────────────────────────────
    summary = {
        "tanggal_laporan": today.strftime("%d %B %Y"),
        "periode": f"{week_ago.strftime('%d %B')} - {today.strftime('%d %B %Y')}",
        "total_omset_minggu_ini": round(total_omset, 0),
        "total_transaksi": total_trx,
        "rata_rata_transaksi_per_hari": avg_trx_per_day,
        "rata_rata_omset_per_hari": avg_omset_per_day,
        "bintang_warung": bintang_warung,
    }

    if hari_ramai:
        summary["hari_paling_ramai"] = {
            "tanggal": hari_ramai,
            "omset": round(daily_revenue[hari_ramai], 0),
        }
    if hari_sepi:
        summary["hari_paling_sepi"] = {
            "tanggal": hari_sepi,
            "omset": round(daily_revenue[hari_sepi], 0),
        }
    if dead_stock:
        summary["produk_kurang_laku"] = dead_stock[:5]

    return summary


# ─── LLM Prompt & Call ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah "Asisten Warung", rekan bisnis digital untuk pemilik warung kelontong/kafe kecil di Indonesia.

Konteks: Kamu sedang membuat LAPORAN PORTOFOLIO MINGGUAN. Data yang kamu terima adalah RINGKASAN 7 HARI KE BELAKANG (retrospektif), bukan prediksi ke depan.

Tugasmu:
1. Berikan MAKSIMAL 3-4 poin analisis singkat berdasarkan performa minggu lalu.
2. Gunakan bahasa Indonesia santai tapi sopan (kayak ngobrol sama teman bisnis).
3. Sebutkan nama produk spesifik jika relevan.
4. Jangan gunakan jargon teknis ML/AI.
5. Format respons sebagai daftar bernomor, singkat dan mudah dipahami.
6. Analisis yang berguna: tren omset, produk bintang, produk yang kurang laku (sarankan promo/diskon), hari ramai vs sepi.
7. Jika ada produk kurang laku, sarankan strategi: promo, bundling, atau pertimbangkan tidak restock.
8. Selalu tutup dengan satu kalimat semangat/motivasi.
9. JANGAN prediksi ke depan. Fokus pada evaluasi minggu yang sudah berlalu."""


def build_llm_prompt(summary: dict) -> str:
    """Build the user prompt with the micro-summarized data."""
    return (
        f"Berikut ringkasan performa warung 7 hari terakhir:\n\n"
        f"```json\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Berdasarkan data di atas, berikan analisis portofolio mingguan singkat "
        f"untuk pemilik warung. Fokus pada evaluasi performa minggu lalu, "
        f"bukan prediksi ke depan."
    )


def _call_gemini_once(prompt: str, system_prompt: str) -> str:
    """
    Single attempt memanggil Google Gemini API.
    Raises Exception jika gagal.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=500,
        ),
    )

    if response.text:
        return response.text.strip()

    raise ValueError("Gemini returned empty response")


def _call_openai_once(prompt: str, system_prompt: str) -> str:
    """
    Single attempt memanggil OpenAI API.
    Raises Exception jika gagal.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    if response.choices:
        content = response.choices[0].message.content
        if content:
            return content.strip()

    raise ValueError("OpenAI returned empty response")


def _call_with_retry(
    call_fn, prompt: str, system_prompt: str, provider_name: str
) -> Optional[str]:
    """
    Mencoba memanggil LLM provider dengan retry.
    
    - Retry maksimal MAX_RETRIES kali.
    - Jika semua retry gagal, return None (bukan throw) agar bisa fallback
      ke provider berikutnya.
    - Semua error di-log ke terminal.
    """
    errors = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[LLM] {provider_name} attempt {attempt}/{MAX_RETRIES}...")
            result = call_fn(prompt, system_prompt)
            print(f"[LLM] {provider_name} SUCCESS on attempt {attempt}")
            return result
        except Exception as e:
            error_msg = f"{provider_name} attempt {attempt} failed: {type(e).__name__}: {e}"
            print(f"[LLM] ❌ {error_msg}")
            errors.append(error_msg)
            if attempt < MAX_RETRIES:
                print(f"[LLM] Retrying in {RETRY_DELAY_SEC}s...")
                time.sleep(RETRY_DELAY_SEC)

    # Semua retry gagal
    print(f"[LLM] ⚠️ {provider_name} FAILED after {MAX_RETRIES} attempts")
    return None


def call_llm(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> tuple[str, str]:
    """
    Memanggil LLM dengan retry dan fallback.

    Flow:
    1. Validasi: minimal 1 API key harus ada → jika tidak, throw LLMConfigError
    2. Gemini (2x retry) → jika gagal → OpenAI (2x retry)
    3. Jika SEMUA provider gagal → throw LLMServiceError

    Returns:
        Tuple (response_text, source) dimana source = "gemini" | "openai"

    Raises:
        LLMConfigError: Tidak ada API key yang dikonfigurasi.
        LLMServiceError: Semua LLM provider gagal setelah retry.
    """
    has_gemini = bool(settings.gemini_api_key)
    has_openai = bool(settings.openai_api_key)

    if not has_gemini and not has_openai:
        raise LLMConfigError(
            "Tidak ada API key LLM yang dikonfigurasi. "
            "Set GEMINI_API_KEY dan/atau OPENAI_API_KEY di file .env. "
            "Minimal satu API key harus tersedia untuk fitur AI."
        )

    print(f"[LLM] Available providers: "
          f"{'Gemini ✓' if has_gemini else 'Gemini ✗'} | "
          f"{'OpenAI ✓' if has_openai else 'OpenAI ✗'}")

    # Try Gemini (primary)
    result = None
    source = None

    if has_gemini:
        result = _call_with_retry(
            _call_gemini_once, prompt, system_prompt, "Gemini"
        )
        if result:
            source = "gemini"

    # Fallback to OpenAI
    if not result and has_openai:
        print("[LLM] Falling back to OpenAI...")
        result = _call_with_retry(
            _call_openai_once, prompt, system_prompt, "OpenAI"
        )
        if result:
            source = "openai"

    # Semua gagal
    if not result:
        tried_providers = []
        if has_gemini:
            tried_providers.append(f"Gemini ({MAX_RETRIES}x)")
        if has_openai:
            tried_providers.append(f"OpenAI ({MAX_RETRIES}x)")

        raise LLMServiceError(
            f"Semua LLM provider gagal setelah retry. "
            f"Providers yang dicoba: {', '.join(tried_providers)}. "
            f"Periksa API key, koneksi internet, atau status layanan provider."
        )

    return result, source


# ─── Main Entry Point ────────────────────────────────────────────────────────


def generate_portfolio_insights(
    transactions: list[dict],
) -> dict:
    """
    Entry point utama: menghasilkan laporan portofolio mingguan dari LLM.

    Ini adalah RINGKASAN RETROSPEKTIF (7 hari ke belakang), bukan prediksi.
    Dipanggil via Laravel Cronjob setiap 7 hari.

    Flow:
    1. Rangkum data transaksi 7 hari terakhir → JSON ringan
    2. Kirim ke LLM (Gemini/OpenAI) dengan persona "Asisten Warung"
    3. Return hasil analisis portofolio

    Returns:
        {
            "insight": "... analisis portofolio dari LLM ...",
            "summary": { ... data ringkas ... },
            "source": "gemini" | "openai",
            "generated_at": "...",
            "valid_until": "...",
        }

    Raises:
        LLMConfigError: Tidak ada API key yang dikonfigurasi.
        LLMServiceError: Semua LLM provider gagal setelah retry.
    """
    # 1. Build retrospective summary
    summary = build_portfolio_summary(transactions)
    prompt = build_llm_prompt(summary)

    now = datetime.now()

    print("\n" + "=" * 70)
    print("[PORTFOLIO] Generating weekly business portfolio...")
    print(f"[PORTFOLIO] Summary data size: {len(json.dumps(summary))} chars")
    print("=" * 70)

    # 2. Call LLM with retry & fallback
    insight, source = call_llm(prompt, SYSTEM_PROMPT)

    # 3. Return result
    valid_until = now + timedelta(days=7)

    print(f"[PORTFOLIO] Source: {source}")
    print(f"[PORTFOLIO] Valid until: {valid_until.strftime('%Y-%m-%d')}")
    print(f"[DONE] Weekly portfolio generated!\n")

    return {
        "insight": insight,
        "summary": summary,
        "source": source,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "valid_until": valid_until.strftime("%Y-%m-%d %H:%M:%S"),
    }
