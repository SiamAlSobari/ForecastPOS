"""
Service layer untuk Decision Support System Restock Barang.
Menghubungkan AI engine dengan API controller.

Fitur:
- Analisis restock per produk (ML-based)
- Ringkasan restock semua produk
- CONCURRENT processing: semua produk dianalisis secara paralel menggunakan
  ThreadPoolExecutor agar request dari cron job tidak menumpuk
- Seasonal insight: LLM overlay untuk prediksi musiman/hari raya (opsional)
- Seasonal restock per produk: LLM menentukan range restock musiman
  per produk berdasarkan konteks hari raya (opsional, hanya saat hari raya besar)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.ai.stock_ai import (
    analyze_restock,
    normalize_transactions,
    generate_seasonal_insight,
    generate_seasonal_restock_per_product,
)

# Thread pool yang di-share — reuse antar request untuk efisiensi
# max_workers=None → Python auto-pick (min(32, os.cpu_count() + 4))
_executor = ThreadPoolExecutor(max_workers=None)


def get_restock_analysis(
    transactions: list[dict],
    product_id: int,
    current_stock: int | None = None,
    forecast_days: int = 14,
) -> dict:
    """
    Mengambil analisis restock untuk produk tertentu (sync version).

    Args:
        transactions: List data transaksi dari request body.
        product_id: ID produk.
        current_stock: Override stok saat ini (opsional).
        forecast_days: Jumlah hari prediksi ke depan.

    Returns:
        Dictionary hasil analisis lengkap (termasuk risk & risk_point).
    """
    return analyze_restock(
        transactions=transactions,
        product_id=product_id,
        current_stock_override=current_stock,
        forecast_days=forecast_days,
    )


async def get_restock_analysis_async(
    transactions: list[dict],
    product_id: int,
    current_stock: int | None = None,
    forecast_days: int = 14,
) -> dict:
    """
    Mengambil analisis restock untuk produk tertentu (async version).
    Menggunakan thread pool agar CPU-bound ML training tidak blocking event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        analyze_restock,
        transactions,
        product_id,
        current_stock,
        forecast_days,
    )


def _analyze_single_product(
    transactions: list[dict],
    product_id: int,
    forecast_days: int,
) -> Optional[dict]:
    """
    Analisis satu produk — wrapper untuk ThreadPoolExecutor.
    Returns None jika ada error (skip produk tersebut).
    """
    try:
        analysis = analyze_restock(
            transactions=transactions,
            product_id=product_id,
            forecast_days=forecast_days,
        )
        if "error" in analysis:
            return None

        restock = analysis["restock_recommendation"]
        return {
            "product_id": product_id,
            "product_name": analysis.get("product_name", f"Product #{product_id}"),
            "product_price": analysis.get("product_price", "0.00"),
            "current_stock": analysis["current_stock"],
            "days_until_empty": analysis["days_until_empty"],
            "estimated_empty_date": analysis["estimated_empty_date"],
            "urgency_level": analysis["urgency_level"],
            "urgency_description": analysis["urgency_description"],
            "risk": analysis["risk"],
            "risk_point": analysis["risk_point"],
            "restock_recommendation": {
                "min": restock["min"],
                "max": restock["max"],
                "label": restock["label"],
            },
            "seasonal_restock": None,  # Default: null (diisi jika ada hari raya besar)
            "avg_daily_sales": analysis.get("avg_daily_sales", 0),
            "accuracy_pct": analysis.get("accuracy_pct", 0),
        }
    except Exception as e:
        print(f"[STOCK-SERVICE] Error analyzing product {product_id}: {e}")
        return None


def get_all_products_summary(
    transactions: list[dict],
    forecast_days: int = 14,
    include_seasonal: bool = False,
) -> dict:
    """
    Mengambil ringkasan restock untuk SEMUA produk secara SYNC.
    Produk dianalisis secara PARALEL menggunakan ThreadPoolExecutor.
    Mengurutkan berdasarkan risk_point: CRITICAL(3) -> MEDIUM(2) -> NORMAL(1).

    Args:
        transactions: List data transaksi.
        forecast_days: Jumlah hari prediksi ke depan.
        include_seasonal: Sertakan nasehat musiman dari LLM? (default False).

    Returns:
        {
            "products": [...],
            "seasonal_insight": {...} | null
        }
    """
    # Kumpulkan semua unique product_id
    product_ids = set()
    for trx in transactions:
        for item in trx.get("items", []):
            product_ids.add(item["product_id"])

    sorted_pids = sorted(product_ids)

    # ─── CONCURRENT: Analisis semua produk secara paralel ─────────────
    from concurrent.futures import as_completed

    futures = {}
    for pid in sorted_pids:
        future = _executor.submit(
            _analyze_single_product, transactions, pid, forecast_days
        )
        futures[future] = pid

    results = []
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            results.append(result)

    # Sort by risk_point descending (CRITICAL=3 first, then MEDIUM=2, then NORMAL=1)
    results.sort(key=lambda x: x["risk_point"], reverse=True)

    # Seasonal insight & per-product seasonal restock (opsional, hanya jika diminta)
    seasonal = None
    if include_seasonal:
        seasonal = generate_seasonal_insight(results)

        # Generate per-product seasonal restock ranges via LLM
        # Hanya jika ada hari raya BESAR terdekat
        seasonal_restock_map = generate_seasonal_restock_per_product(results)

        # Inject seasonal_restock ke masing-masing produk
        if seasonal_restock_map:
            for product in results:
                pid = product["product_id"]
                if pid in seasonal_restock_map:
                    product["seasonal_restock"] = seasonal_restock_map[pid]

    return {
        "products": results,
        "seasonal_insight": seasonal,
    }


async def get_all_products_summary_async(
    transactions: list[dict],
    forecast_days: int = 14,
    include_seasonal: bool = False,
) -> dict:
    """
    Mengambil ringkasan restock untuk SEMUA produk secara ASYNC.
    Semua produk dianalisis secara KONKUREN menggunakan asyncio + ThreadPool.
    Optimal untuk cron job yang mengirim banyak request sekaligus.

    Alur:
    1. Kumpulkan semua product_id
    2. Submit SEMUA analisis ke thread pool secara serentak
    3. Gather semua hasil secara async (non-blocking event loop)
    4. Sort by urgency dan inject seasonal restock

    Args:
        transactions: List data transaksi.
        forecast_days: Jumlah hari prediksi ke depan.
        include_seasonal: Sertakan nasehat musiman dari LLM? (default False).

    Returns:
        {
            "products": [...],
            "seasonal_insight": {...} | null
        }
    """
    # Kumpulkan semua unique product_id
    product_ids = set()
    for trx in transactions:
        for item in trx.get("items", []):
            product_ids.add(item["product_id"])

    sorted_pids = sorted(product_ids)
    loop = asyncio.get_event_loop()

    # ─── CONCURRENT: Semua produk dianalisis secara paralel ───────────
    tasks = [
        loop.run_in_executor(
            _executor,
            _analyze_single_product,
            transactions,
            pid,
            forecast_days,
        )
        for pid in sorted_pids
    ]

    # Gather semua results secara async
    raw_results = await asyncio.gather(*tasks)

    results = [r for r in raw_results if r is not None]

    # Sort by risk_point descending
    results.sort(key=lambda x: x["risk_point"], reverse=True)

    # Seasonal insight (opsional) — bisa juga di-offload ke thread pool
    seasonal = None
    if include_seasonal:
        seasonal = await loop.run_in_executor(
            _executor, generate_seasonal_insight, results
        )

        seasonal_restock_map = await loop.run_in_executor(
            _executor, generate_seasonal_restock_per_product, results
        )

        if seasonal_restock_map:
            for product in results:
                pid = product["product_id"]
                if pid in seasonal_restock_map:
                    product["seasonal_restock"] = seasonal_restock_map[pid]

    return {
        "products": results,
        "seasonal_insight": seasonal,
    }
