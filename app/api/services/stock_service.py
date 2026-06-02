"""
Service layer untuk Decision Support System Restock Barang.
Menghubungkan AI engine dengan API controller.

Fitur:
- Analisis restock per produk (ML-based)
- Ringkasan restock semua produk
- Seasonal insight: LLM overlay untuk prediksi musiman/hari raya (opsional)
- Seasonal restock per produk: LLM menentukan range restock musiman
  per produk berdasarkan konteks hari raya (opsional, hanya saat hari raya besar)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from app.ai.stock_ai import (
    analyze_restock,
    normalize_transactions,
    generate_seasonal_insight,
    generate_seasonal_restock_per_product,
)
from app.api.cache import stock_cache


def get_restock_analysis(
    transactions: list[dict],
    product_id: int,
    current_stock: int | None = None,
    forecast_days: int = 14,
) -> dict:
    return analyze_restock(
        transactions=transactions,
        product_id=product_id,
        current_stock_override=current_stock,
        forecast_days=forecast_days,
    )


def get_all_products_summary(
    transactions: list[dict],
    forecast_days: int = 14,
    include_seasonal: bool = False,
) -> dict:
    """
    Mengambil ringkasan restock untuk SEMUA produk yang ada di data.
    Mengurutkan berdasarkan risk_point: CRITICAL(3) -> MEDIUM(2) -> NORMAL(1).

    Jika include_seasonal=True, akan menambahkan:
    1. seasonal_insight: Nasehat umum LLM tentang musiman/hari raya.
    2. seasonal_restock: Range restock musiman per produk (optional field).
       Hanya produk yang relevan dengan hari raya yang mendapat field ini.
       Produk yang tidak relevan → seasonal_restock = null.

    Args:
        transactions: List data transaksi.
        forecast_days: Jumlah hari prediksi ke depan.
        include_seasonal: Sertakan nasehat musiman dari LLM? (default False).

    Returns:
        {
            "products": [
                {
                    ...,
                    "restock_recommendation": {"min": .., "max": .., "label": ..},
                    "seasonal_restock": {"min": .., "max": .., "label": .., "holiday": .., "reason": ..} | null,
                },
                ...
            ],
            "seasonal_insight": {...} | null
        }
    """
    # Cek cache dulu — kalau data sama, skip ML training
    cached = stock_cache.get(transactions, forecast_days, include_seasonal)
    if cached is not None:
        return cached

    # Kumpulkan semua unique product_id
    product_ids = set()
    for trx in transactions:
        for item in trx.get("items", []):
            product_ids.add(item["product_id"])

    # Analisis setiap produk secara paralel
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(
                analyze_restock,
                transactions=transactions,
                product_id=pid,
                forecast_days=forecast_days,
            ): pid
            for pid in sorted(product_ids)
        }

        for future in as_completed(future_map):
            pid = future_map[future]
            analysis = future.result()
            if "error" not in analysis:
                restock = analysis["restock_recommendation"]
                results.append({
                    "product_id": pid,
                    "product_name": analysis.get("product_name", f"Product #{pid}"),
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
                    "seasonal_restock": None,
                    "avg_daily_sales": analysis.get("avg_daily_sales", 0),
                })

    # Sort by risk_point descending (CRITICAL=3 first, then MEDIUM=2, then NORMAL=1)
    results.sort(key=lambda x: x["risk_point"], reverse=True)

    # Seasonal insight & per-product seasonal restock (opsional, hanya jika diminta)
    seasonal = None
    if include_seasonal:
        with ThreadPoolExecutor(max_workers=2) as llm_executor:
            future_insight = llm_executor.submit(generate_seasonal_insight, results)
            future_restock = llm_executor.submit(generate_seasonal_restock_per_product, results)

            seasonal = future_insight.result()
            seasonal_restock_map = future_restock.result()

        # Inject seasonal_restock ke masing-masing produk
        if seasonal_restock_map:
            for product in results:
                pid = product["product_id"]
                if pid in seasonal_restock_map:
                    product["seasonal_restock"] = seasonal_restock_map[pid]

    result = {
        "products": results,
        "seasonal_insight": seasonal,
    }
    stock_cache.set(result, transactions, forecast_days, include_seasonal)
    return result
