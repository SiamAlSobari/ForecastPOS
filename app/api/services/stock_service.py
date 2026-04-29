"""
Service layer untuk Decision Support System Restock Barang.
Menghubungkan AI engine dengan API controller.
"""

from app.ai.stock_ai import analyze_restock, normalize_transactions


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
) -> list[dict]:
    """
    Mengambil ringkasan restock untuk SEMUA produk yang ada di data.
    Mengurutkan berdasarkan risk_point: CRITICAL(3) -> MEDIUM(2) -> NORMAL(1).
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
                "recommended_restock_qty": analysis["restock_recommendation"]["recommended_quantity"],
                "avg_daily_sales": analysis["historical_stats"]["avg_daily_sales"],
            })

    # Sort by risk_point descending (CRITICAL=3 first, then MEDIUM=2, then NORMAL=1)
    results.sort(key=lambda x: x["risk_point"], reverse=True)
    return results
