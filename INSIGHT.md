# Dokumentasi AI Insights (Portofolio Bisnis Mingguan)

Fitur **AI Insights Portofolio** adalah endpoint yang dirancang untuk menghasilkan laporan retrospektif (kinerja masa lalu) selama 7 hari ke belakang. Laporan ini tidak melakukan prediksi, melainkan merangkum performa bisnis warung dan memberikan nasehat/analisis menggunakan LLM.

## Alur Kerja (Workflow)
1. **Laravel Task Scheduler** menjalankan cronjob setiap 7 hari.
2. Cronjob memanggil endpoint `POST /api/insights/generate` pada backend Python.
3. Python memfilter data transaksi khusus tipe `SALE` 7 hari terakhir, kemudian melakukan **Micro-Summarization** (merangkum data menjadi JSON ringan agar hemat token).
4. Data ringkasan dikirim ke LLM (Gemini / Groq) dengan persona "Asisten Warung".
5. Respons dari Python dikembalikan ke Laravel dan disimpan di database pada tabel `ai_insights`.
6. Ketika pengguna membuka aplikasi, aplikasi mengambil data instan dari database tanpa memanggil API LLM berulang kali (0 token tambahan).

## Endpoint API
**`POST /api/insights/generate`**

### Request (Payload)
Secara default, data transaksi dikirim melalui request body (JSON) menggunakan model `SummaryRequest`. Jika request body tidak berisi `data`, sistem akan otomatis membaca dari file lokal `trx.json`.

```json
{
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "trx_type": "SALE", // Hanya trx_type SALE yang akan dihitung
      "trx_date": "2026-05-18T10:00:00",
      "payment_method": "CASH",
      "total_amount": "50000",
      "items": [
        {
          "product_id": 101,
          "quantity": 2,
          "unit_price": "25000",
          "product": {
            "id": 101,
            "name": "Indomie Goreng",
            "price": "25000"
          }
        }
      ]
    }
  ]
}
```

### Response (Berhasil - 200 OK)
Jika berhasil, API akan merespons dengan hasil insight LLM beserta data agregasi yang sudah di-summary oleh sistem.

```json
{
  "message": "Portofolio bisnis mingguan berhasil dibuat",
  "data": {
    "insight": "1. Mantap bosku! Omset minggu ini mencapai Rp 1.500.000...\n2. Produk Indomie Goreng jadi bintang warung kita...\n3. Tapi hati-hati, produk Taro kurang laku, mungkin bisa coba dipromo...",
    "summary": {
      "tanggal_laporan": "19 May 2026",
      "periode": "12 May - 19 May 2026",
      "total_omset_minggu_ini": 1500000,
      "total_transaksi": 45,
      "rata_rata_transaksi_per_hari": 6.4,
      "rata_rata_omset_per_hari": 214286,
      "bintang_warung": [
        {"nama": "Indomie Goreng", "terjual": 50, "omset": 150000}
      ],
      "hari_paling_ramai": {"tanggal": "2026-05-15", "omset": 350000},
      "hari_paling_sepi": {"tanggal": "2026-05-18", "omset": 50000},
      "produk_kurang_laku": ["Taro Snack"]
    },
    "source": "gemini-primary", 
    "generated_at": "2026-05-19 13:00:00",
    "valid_until": "2026-05-26 13:00:00"
  }
}
```

### Response Error Handling
- **`503 Service Unavailable`**: API Key LLM (`GEMINI_API_KEY` / `GROQ_API_KEY`) belum disetting di file `.env`.
- **`502 Bad Gateway`**: Semua provider LLM gagal dipanggil (limit quota habis, layanan mati) setelah mencoba seluruh skema retry.
- **`400 Bad Request`**: File `trx.json` tidak ditemukan dan tidak ada data yang dikirim di request body.
- **`500 Internal Server Error`**: Error server/logika sistem tidak terduga.

## Skema Fallback & Keandalan LLM (Retry Mechanism)
Sistem menggunakan pendekatan berlapis untuk memastikan portofolio terus-menerus bisa di-generate meskipun salah satu AI mengalami *Rate Limit* (429) atau sedang *down*. Setiap model akan di-*retry* hingga 2 kali sebelum berpindah (fallback) ke model selanjutnya:
1. **Gemini Primary (`gemini-2.0-flash`)**
2. **Gemini Lite (`gemini-2.5-flash-lite`)**
3. **Groq Fallback (`llama-3.3-70b-versatile`)**
