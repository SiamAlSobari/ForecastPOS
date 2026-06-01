"""
Pydantic models untuk request/response Decision Support System Restock.
Disesuaikan dengan format response API yang menyertakan nested product + stocks.
"""

from pydantic import BaseModel, Field


class StockEntry(BaseModel):
    """Stok produk saat ini."""
    model_config = {"extra": "allow"}

    id: str
    product_id: int
    stock_on_hand: int
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None


class Product(BaseModel):
    """Detail produk yang terdapat di dalam item transaksi."""
    model_config = {"extra": "allow"}

    id: int
    name: str
    price: str
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    image_url: str | None = None
    category_id: int | None = None
    deleted_at: str | None = None
    is_active: bool = True
    user_id: int | None = None
    stocks: list[StockEntry] = []


class TransactionItem(BaseModel):
    """Item dalam sebuah transaksi."""
    model_config = {"extra": "allow"}

    id: int | None = None
    transaction_id: int | None = None
    product_id: int
    quantity: int
    unit_price: str
    line_price: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    product: Product | None = None


class Transaction(BaseModel):
    """Satu transaksi (SALE, PURCHASE, atau ADJUSTMENT).

    Tipe transaksi:
    - SALE:       Penjualan barang (mengurangi stok)
    - PURCHASE:   Pembelian/restock barang (menambah stok)
    - ADJUSTMENT: Koreksi stok manual / stock opname (menambah stok,
                  terjadi ketika user melakukan update stock product)
    """
    model_config = {"extra": "allow"}

    id: int
    user_id: int
    trx_type: str  # "SALE", "PURCHASE", atau "ADJUSTMENT"
    trx_date: str
    payment_method: str
    paid_at: str | None = None
    total_amount: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None
    items: list[TransactionItem]


class RestockRequest(BaseModel):
    """Body untuk analisis restock satu produk."""
    data: list[Transaction]
    product_id: int
    current_stock: int | None = None
    forecast_days: int = Field(default=14, ge=1, le=90)


class SummaryRequest(BaseModel):
    """Body untuk ringkasan restock semua produk."""
    data: list[Transaction] = []
    forecast_days: int = Field(default=14, ge=1, le=90)
