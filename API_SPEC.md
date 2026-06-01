# ML Kasir API - API Specification

Dokumentasi ini menjelaskan endpoint, request, dan response dari Python AI Engine (Decision Support System) untuk aplikasi Kasir.

**Base URL:** `http://localhost:8080` (atau URL server deployment Anda)

---

## 📋 Struktur Data Global (Shared Model)

Ketiga endpoint di bawah ini menerima format request body yang sama, yaitu berupa kumpulan data transaksi historis dari aplikasi kasir utama (Laravel/Frontend).

### `SummaryRequest` (Request Body)
Semua endpoint menggunakan format JSON body berikut:

```json
{
  "forecast_days": 14,
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "trx_type": "SALE", // "SALE", "PURCHASE", atau "ADJUSTMENT"
      "trx_date": "2026-05-18",
      "payment_method": "CASH",
      "paid_at": "2026-05-18 14:30:00",
      "total_amount": "50000.00",
      "items": [
        {
          "product_id": 101,
          "quantity": 2,
          "unit_price": "25000.00",
          "product": {
            "id": 101,
            "name": "Sabun Cuci",
            "price": "25000.00",
            "stocks": [
              {
                "id": "stock-1",
                "product_id": 101,
                "stock_on_hand": 50
              }
            ]
          }
        }
      ]
    }
  ]
}
```

*Keterangan:*
- `forecast_days` (opsional): Jumlah hari ke depan untuk prediksi (default: 14, min: 1, max: 90).
- `data` (opsional): Array of Transactions. Jika kosong, backend otomatis memakai data *dummy* dari `trx.json` (berguna untuk testing).

---

## 1. 📦 Restock Summary API

Endpoint ini memberikan ringkasan prediksi kapan stok tiap produk akan habis beserta rekomendasi jumlah *restock*-nya. Fitur ini menggunakan Machine Learning untuk prediksi dasar dan Large Language Model (LLM) untuk *overlay* prediksi hari raya/musiman.

**Endpoint:**
`POST /api/predict/restock/summary`

**Query Parameters:**
- `include_seasonal` (boolean, opsional): Jika `true`, sistem akan mengaktifkan pengecekan kalender hari raya dan meminta nasehat musiman dari LLM. Default `false`.

**Response (200 OK):**
```json
{
  "message": "Ringkasan restock semua produk",
  "total_products": 10,
  "data": [
    {
      "product_id": 101,
      "product_name": "Sabun Cuci",
      "product_price": "25000.00",
      "analysis_date": "2026-05-19 08:00:00",
      "avg_daily_sales": 3.4,
      "current_stock": 50,
      "days_until_empty": 4,
      "estimated_empty_date": "2026-05-23",
      "urgency_level": "MEDIUM",
      "urgency_description": "⚡ PERHATIAN! Stok diestimasi akan habis dalam 4 hari (sekitar tanggal 2026-05-23). Pertimbangkan untuk restock.",
      "risk": "MEDIUM",
      "risk_point": 2,
      "restock_recommendation": {
        "min": 15,
        "max": 25,
        "label": "Saran restock: 15 - 25 item untuk persediaan 7 hari ke depan.",
        "target_days_coverage": 7
      },
      "stock_timeline": [
        {
          "date": "2026-05-19",
          "day_name": "Tuesday",
          "predicted_sales": 3.5,
          "remaining_stock": 46
        }
        // ... (data per hari lainnya)
      ]
    }
  ],
  "seasonal_insight": {
    "has_upcoming_holiday": true,
    "upcoming_holidays": [
      {
        "date": "2026-05-27",
        "name": "Eid al-Adha",
        "days_away": 8,
        "impact": "HIGH"
      }
    ],
    "seasonal_advice": "Menjelang Idul Adha minggu depan, disarankan restock lebih besar terutama untuk bumbu masakan dan alat pemanggang. Insting Anda benar!",
    "source": "gemini-primary"
  }
}
```

---

## 2. ⏳ Busy Hours Prediction API

Endpoint ini memprediksi jam-jam sibuk, jumlah transaksi, pendapatan (*revenue*), serta probabilitas produk apa saja yang laku terjual di jam tertentu pada hari-hari mendatang. Sangat berguna untuk menjadwalkan shift karyawan dan persiapan stok harian.

**Endpoint:**
`POST /api/predict/busy-hours`

**Response (200 OK):**
```json
{
  "message": "Prediksi jam sibuk berhasil",
  "data": {
    "analysis_date": "2026-05-19 08:00:00",
    "forecast_days": 14,
    "busiest_day": "2026-05-23 (Saturday)",
    "quietest_day": "2026-05-20 (Wednesday)",
    "total_peak_hours": 8,
    "top_peak_hours": [
      {
        "date": "2026-05-23",
        "day_name": "Saturday",
        "hour": "19:00",
        "level": "PEAK",
        "label": "Sangat Sibuk 🔥",
        "estimated_transactions": "10 - 15"
      }
    ],
    "daily_forecasts": [
      {
        "date": "2026-05-19",
        "day_name": "Tuesday",
        "day_of_week": 1,
        "is_weekend": false,
        "estimated_transactions": {
          "min": 40,
          "max": 60,
          "label": "40 - 60 transaksi"
        },
        "estimated_revenue": {
          "min": 500000,
          "max": 750000,
          "label": "Rp 500.000 - Rp 750.000"
        },
        "peak_hour": "19:00",
        "peak_hour_label": "Ramai 📈",
        "busy_hours_count": 2,
        "hourly_breakdown": [
          {
            "hour": "07:00",
            "estimated_transactions": {
              "min": 1,
              "max": 3,
              "label": "1 - 3 transaksi"
            },
            "estimated_revenue": {
              "min": 15000,
              "max": 30000,
              "label": "Rp 15.000 - Rp 30.000"
            },
            "busy_level": "LOW",
            "busy_label": "Sepi Santai 🍃",
            "what_to_prepare": null,
            "predicted_products": [
              {
                "product_id": 101,
                "product_name": "Sabun Cuci",
                "probability": 0.85,
                "estimated_qty": 2.5,
                "estimated_revenue": 62500
              }
            ]
          }
        ]
      }
    ]
  }
}
```

---

## 3. 📈 Weekly Portfolio Insights API

Endpoint ini bersifat **retrospektif** (menganalisis masa lalu, BUKAN memprediksi masa depan). Endpoint ini dibuat khusus untuk cronjob mingguan, yang bertugas merangkum performa transaksi 7 hari ke belakang lalu dianalisis oleh LLM (Gemini/Groq) untuk memberikan nasehat strategi bisnis (produk terlaris, *dead stock*, dll).

**Endpoint:**
`POST /api/insights/generate`

**Response (200 OK):**
```json
{
  "message": "Portofolio bisnis mingguan berhasil dibuat",
  "data": {
    "insight": "1. **Omset Mingguan Stabil**: Minggu ini warung mencetak Rp 2.500.000, hari Sabtu jadi juara paling ramai!\n2. **Produk Bintang**: Mi Instan dan Telur terus laku keras, pertahankan stok ini.\n3. **Promo Dibutuhkan**: Shampo sachet kurang bergerak, pertimbangkan untuk di-bundle atau beri diskon ya! Terus semangat jualannya!",
    "summary": {
      "tanggal_laporan": "19 May 2026",
      "periode": "12 May - 19 May 2026",
      "total_omset_minggu_ini": 2500000,
      "total_transaksi": 150,
      "rata_rata_transaksi_per_hari": 21.4,
      "rata_rata_omset_per_hari": 357143,
      "bintang_warung": [
        {
          "nama": "Mi Instan",
          "terjual": 80,
          "omset": 240000
        }
      ],
      "hari_paling_ramai": {
        "tanggal": "2026-05-16",
        "omset": 500000
      },
      "produk_kurang_laku": ["Shampo Sachet"]
    },
    "source": "gemini-primary",
    "generated_at": "2026-05-19 08:00:00",
    "valid_until": "2026-05-26 08:00:00"
  }
}
```

**Kemungkinan HTTP Errors (khusus Insights API):**
- **400 Bad Request:** Data payload kosong dan file `trx.json` tidak ditemukan.
- **502 Bad Gateway:** Semua percobaan LLM (Gemini & Groq) gagal karena masalah koneksi/limit.
- **503 Service Unavailable:** API Key tidak di-setup di `.env` backend Python.
