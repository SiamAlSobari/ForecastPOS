"""
Service layer untuk Decision Support System Restock Barang.
Menghubungkan AI engine dengan API controller.

Fitur:
- Analisis restock per produk (ML-based)
- Ringkasan restock semua produk
- Seasonal insight: LLM overlay untuk prediksi musiman/hari raya (opsional)
"""

from app.ai.stock_ai import (
    analyze_restock,
    normalize_transactions,
    generate_seasonal_insight,
)


def get_restock_analysis(
    transactions: list[dict],
    product_id: int,
    current_stock: int | None = None,
    forecast_days: int = 14,
) -> dict:
    """
    Mengambil analisis restock untuk produk tertentu.

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


def get_all_products_summary(
    transactions: list[dict],
    forecast_days: int = 14,
    include_seasonal: bool = False,
) -> dict:
    """
    Mengambil ringkasan restock untuk SEMUA produk yang ada di data.
    Mengurutkan berdasarkan risk_point: CRITICAL(3) -> MEDIUM(2) -> NORMAL(1).

    Jika include_seasonal=True, akan menambahkan nasehat LLM tentang
    prediksi musiman/hari raya (membutuhkan API key LLM).

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

    # Analisis setiap produk
    results = []

    for pid in sorted(product_ids):
        analysis = analyze_restock(
            transactions=transactions,
            product_id=pid,
            forecast_days=forecast_days,
        )
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
                "avg_daily_sales": analysis.get("avg_daily_sales", 0),
            })

    # Sort by risk_point descending (CRITICAL=3 first, then MEDIUM=2, then NORMAL=1)
    results.sort(key=lambda x: x["risk_point"], reverse=True)

    # Seasonal insight (opsional, hanya jika diminta)
    seasonal = None
    if include_seasonal:
        seasonal = generate_seasonal_insight(results)

    return {
        "products": results,
        "seasonal_insight": seasonal,
    }
