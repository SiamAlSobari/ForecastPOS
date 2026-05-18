"""Generate natural transaction data with varied urgency levels."""
import json
from datetime import datetime, timedelta

PRODUCTS = {
    1: {"name": "Sabun cuci piring", "price": "15000.00", "cat": 1, "stock": 2},
    2: {"name": "Mi instant", "price": "3000.00", "cat": 2, "stock": 8},
    3: {"name": "Air mineral 600ml", "price": "3500.00", "cat": 3, "stock": 20},
    4: {"name": "Beras 5 kg", "price": "70000.00", "cat": 4, "stock": 15},
    5: {"name": "Minyak goreng 1L", "price": "18000.00", "cat": 5, "stock": 60},
    6: {"name": "Gula pasir 1kg", "price": "14000.00", "cat": 6, "stock": 14},
}

STOCK_IDS = {
    1: "019dc8a5-47d3-716f-b18f-444d7cfc593b",
    2: "019dc8a5-47f4-70ba-8bbf-468182f4300c",
    3: "019dc8a5-47fe-7320-82be-3081bb593d3e",
    4: "019dc8a5-4808-7244-9b5d-101c90e6319e",
    5: "019dc8a5-4812-7111-aaaa-111111111111",
    6: "019dc8a5-481c-7222-bbbb-222222222222",
}

PROD_CREATED = "2026-05-02T08:00:00.000000Z"

def mk_product(pid):
    p = PRODUCTS[pid]
    return {
        "id": pid, "name": p["name"], "price": p["price"],
        "description": f"{p['name']} adalah produk contoh.",
        "created_at": PROD_CREATED, "updated_at": PROD_CREATED,
        "image_url": f"https://placehold.co/400x400?text={p['name'].replace(' ', '+')}",
        "category_id": p["cat"], "deleted_at": None, "is_active": True, "user_id": 1,
        "stocks": [{"id": STOCK_IDS[pid], "product_id": pid,
                     "stock_on_hand": p["stock"],
                     "created_at": PROD_CREATED,
                     "updated_at": "2026-05-18T07:24:03.000000Z",
                     "deleted_at": None}]
    }

def ts(dt): return dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
def paid(dt): return dt.strftime("%Y-%m-%d %H:%M:%S")
def ymd(dt): return dt.strftime("%Y-%m-%d")

base = datetime(2026, 5, 2, 8, 0, 0)

# (day, h, m, s, type, [(pid, qty)])
# Types:
#   SALE      → penjualan (stock berkurang)
#   PURCHASE  → pembelian/restock (stock bertambah)
#   ADJUSTMENT→ koreksi stock manual / stock opname (stock bertambah/berkurang)
# Design:
#   P1 Sabun: stock=2, sells ~2/day → CRITICAL (habis 1 hari)
#   P2 Mi:    stock=8, sells ~5/day → CRITICAL (habis 1-2 hari)
#   P3 Air:   stock=50, sells ~4/day → NORMAL (habis ~12 hari)
#   P4 Beras: stock=15, sells ~1/day → NORMAL (habis ~15 hari)
#   P5 Minyak: stock=60, sells ~2/day → NORMAL (habis ~30 hari)
#   P6 Gula:  stock=45, sells ~3/day → NORMAL (habis ~15 hari)
RAW = [
    # May 2 - Restock awal besar
    (0, 8,0,0, "PURCHASE", [(1,20),(2,60),(3,40),(4,10),(5,30),(6,25)]),
    (0, 10,15,30, "SALE", [(1,2),(2,5),(3,3),(6,2)]),
    (0, 14,30,10, "SALE", [(2,4),(3,2),(5,1),(4,1)]),
    (0, 17,45,22, "SALE", [(1,1),(6,3),(5,2)]),
    # Apr 11
    (1, 9,5,18, "SALE", [(2,6),(3,4),(1,2)]),
    (1, 12,20,45, "SALE", [(4,1),(5,2),(6,2),(2,3)]),
    (1, 16,40,8, "SALE", [(1,2),(3,3),(6,3)]),
    # Apr 12 (weekend - busier)
    (2, 8,50,0, "SALE", [(2,7),(3,5),(1,3),(5,3)]),
    (2, 11,30,25, "SALE", [(4,1),(6,4),(2,5)]),
    (2, 15,15,40, "SALE", [(1,2),(3,4),(5,2),(6,2)]),
    (2, 18,5,12, "SALE", [(2,4),(3,2)]),
    # Apr 12 - Adjustment: stock opname koreksi (menemukan selisih stok)
    (2, 19,30,0, "ADJUSTMENT", [(3,5)]),  # Air mineral ternyata ada 5 lebih dari catatan
    # Apr 13 (weekend peak)
    (3, 9,10,33, "SALE", [(2,8),(3,6),(1,3),(6,3)]),
    (3, 13,25,15, "SALE", [(4,2),(5,3),(2,4)]),
    (3, 17,0,50, "SALE", [(1,2),(3,3),(6,2),(5,1)]),
    # Apr 14 - Restock mi & sabun
    (4, 7,30,0, "PURCHASE", [(1,15),(2,50),(3,20),(6,20)]),
    (4, 10,20,42, "SALE", [(2,5),(3,3),(1,1),(5,2)]),
    (4, 14,55,18, "SALE", [(4,1),(6,3),(2,3)]),
    (4, 18,10,30, "SALE", [(1,2),(3,2),(5,1)]),
    # Apr 15
    (5, 9,15,22, "SALE", [(2,4),(3,4),(6,2)]),
    (5, 13,40,8, "SALE", [(1,2),(5,2),(4,1)]),
    (5, 17,25,55, "SALE", [(2,6),(3,3),(6,3)]),
    # Apr 16
    (6, 10,5,15, "SALE", [(2,5),(1,1),(3,2)]),
    (6, 14,30,40, "SALE", [(4,1),(5,1),(6,2),(2,3)]),
    (6, 18,50,22, "SALE", [(1,2),(3,3)]),
    # Apr 17 - Restock beras & minyak
    (7, 8,0,0, "PURCHASE", [(4,10),(5,30),(3,15)]),
    (7, 11,10,30, "SALE", [(2,4),(3,4),(1,2),(6,2)]),
    (7, 15,35,18, "SALE", [(5,2),(4,1),(2,5)]),
    (7, 18,20,45, "SALE", [(1,1),(3,2),(6,3)]),
    # Apr 17 - Adjustment: barang rusak/expired dihapus dari stok
    (7, 19,0,0, "ADJUSTMENT", [(2,3),(6,2)]),  # Mi & Gula expired, koreksi tambah
    # Apr 18
    (8, 9,45,10, "SALE", [(2,6),(1,2),(3,3)]),
    (8, 13,15,38, "SALE", [(5,2),(4,1),(6,2)]),
    (8, 17,50,22, "SALE", [(2,3),(3,2),(1,1)]),
    # Apr 19 (weekend)
    (9, 8,30,0, "SALE", [(2,7),(3,5),(1,3),(5,3)]),
    (9, 12,10,25, "SALE", [(4,1),(6,4),(2,4)]),
    (9, 16,45,50, "SALE", [(1,2),(3,3),(5,2),(6,2)]),
    # Apr 20 (weekend peak)
    (10, 9,20,33, "SALE", [(2,9),(3,6),(1,3)]),
    (10, 13,50,15, "SALE", [(4,2),(5,3),(6,3),(2,5)]),
    (10, 18,5,40, "SALE", [(1,2),(3,4),(5,1)]),
    # Apr 21 - Restock
    (11, 7,45,0, "PURCHASE", [(1,15),(2,55),(6,25)]),
    (11, 10,30,18, "SALE", [(2,5),(3,3),(1,2)]),
    (11, 15,20,42, "SALE", [(4,1),(5,2),(6,3)]),
    (11, 18,40,10, "SALE", [(2,4),(3,2),(1,1)]),
    # Apr 21 - Adjustment: stock opname berkala
    (11, 19,15,0, "ADJUSTMENT", [(5,4),(4,2)]),  # Minyak & Beras ditemukan sisa lebih
    # Apr 22
    (12, 9,8,15, "SALE", [(2,5),(3,4),(6,2),(5,2)]),
    (12, 13,42,30, "SALE", [(1,2),(4,1),(2,3)]),
    (12, 17,55,8, "SALE", [(3,3),(6,3),(5,1)]),
    # Apr 23
    (13, 10,5,22, "SALE", [(2,6),(3,3),(1,2)]),
    (13, 14,30,50, "SALE", [(4,1),(5,2),(6,2),(2,3)]),
    (13, 18,15,38, "SALE", [(1,1),(3,2)]),
    # Apr 24
    (14, 9,20,10, "SALE", [(2,5),(3,4),(1,2),(6,3)]),
    (14, 15,10,35, "SALE", [(4,1),(5,2),(2,4)]),
    (14, 18,40,18, "SALE", [(3,3),(1,1),(6,2)]),
    # Apr 25 - Small restock
    (15, 8,50,0, "PURCHASE", [(4,8),(5,25)]),
    (15, 10,15,30, "SALE", [(2,5),(3,3),(6,2)]),
    (15, 14,25,12, "SALE", [(1,2),(5,2),(2,4),(4,1)]),
    (15, 18,5,45, "SALE", [(3,4),(6,3),(1,1)]),
    # May 18 (today) - Busy day
    (16, 8,30,22, "SALE", [(2,6),(3,3),(5,2)]),
    (16, 11,15,40, "SALE", [(1,2),(4,1),(6,2),(2,3)]),
    (16, 14,50,8, "SALE", [(3,3),(2,5),(5,1)]),
    (16, 17,30,55, "SALE", [(1,1),(3,2),(6,2)]),
]

transactions = []
item_id = 1

for idx, (day, h, m, s, ttype, items_raw) in enumerate(RAW, start=1):
    dt = base.replace(hour=h, minute=m, second=s) + timedelta(days=day)
    total = sum(int(float(PRODUCTS[p]["price"]) * q) for p, q in items_raw)
    trx_items = []
    for pid, qty in items_raw:
        pr = PRODUCTS[pid]["price"]
        trx_items.append({
            "id": item_id, "transaction_id": idx, "product_id": pid,
            "quantity": qty, "unit_price": pr,
            "line_price": f"{int(float(pr)*qty)}.00",
            "created_at": ts(dt), "updated_at": ts(dt),
            "product": mk_product(pid),
        })
        item_id += 1
    transactions.append({
        "id": idx, "user_id": 1, "trx_type": ttype, "trx_date": ymd(dt),
        "payment_method": "CASH", "paid_at": paid(dt),
        "total_amount": f"{total}.00",
        "created_at": ts(dt), "updated_at": ts(dt), "deleted_at": None,
        "items": trx_items,
    })

out = {"message": "Daftar riwayat transaksi berhasil diambil", "data": transactions}
with open("c:/Koding/ml/kasir/data/trx.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"OK: {len(transactions)} transaksi, {item_id-1} items")
# Quick check expected urgency
for pid, p in PRODUCTS.items():
    sales = sum(q for _,_,_,_,t,items in RAW if t=="SALE" for pp,q in items if pp==pid)
    purch = sum(q for _,_,_,_,t,items in RAW if t=="PURCHASE" for pp,q in items if pp==pid)
    adj = sum(q for _,_,_,_,t,items in RAW if t=="ADJUSTMENT" for pp,q in items if pp==pid)
    avg = sales/17
    stock = p["stock"]
    days = stock/avg if avg > 0 else 999
    print(f"  P{pid} {p['name']:20s} | stock={stock:3d} | sold={sales:3d} | purchased={purch:3d} | adjusted={adj:3d} | avg/day={avg:.1f} | ~{days:.0f} days")
