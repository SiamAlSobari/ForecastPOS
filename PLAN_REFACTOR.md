# Blueprint Refactor & Optimalisasi AI (Kasir / Warung)

Dokumen ini berisi rencana komprehensif untuk merombak sistem AI (Busy Hour & Stock) agar lebih manusiawi, relevan dengan dunia nyata, dan lebih pintar secara bisnis.

---

## 1. Pembersihan Response API (API Payload Optimization)
**Masalah Saat Ini:** Response API mengembalikan terlalu banyak metadata teknis ML (seperti `accuracy_percent`, `training_samples`, `data_range`, `historical_stats`). Ini membebani payload, tidak berguna bagi frontend (aplikasi mobile), dan bisa membingungkan jika ter-expose.
**Solusi & Aturan Baru:**
- **Frontend Only:** API HANYA boleh mengembalikan data yang benar-benar akan dirender atau dibaca oleh pemilik warung (misal: status, range transaksi, range harga, nama barang).
- **Log via Terminal:** Semua data teknis (akurasi model, total data training, metrik MAPE) **wajib dihilangkan dari return JSON** dan diganti menjadi `print()` atau `logging.info()` di backend Python. Biarkan developer yang melihatnya di terminal.

---

## 2. Perombakan Busy Hour AI (Jam Sibuk)
**Masalah Saat Ini:** Prediksi transaksi berbentuk angka desimal tunggal (`10.5` transaksi).
**Ide & Perbaikan Lanjutan:**
- **Format Rentang (Range):** Ubah hasil output menjadi batas bawah dan batas atas.
  - *Sebelum:* `predicted_transactions: 10.5`
  - *Sesudah:* `estimated_transactions: { min: 9, max: 12, label: "9 - 12 transaksi" }`
  - Berlaku juga untuk estimasi revenue (misal: Rp 150.000 - Rp 200.000).
- **"What to Prepare" (Aksi Persiapan):** Tambahkan balikan data yang memberitahu *apa yang harus disiapkan* di jam tersebut.
  - Daripada sekadar bilang "Jam 12 Sangat Sibuk", AI bisa menambahkan konteks: *"Jam 12 Sangat Sibuk. Siapkan lebih banyak Kopi dan Es Teh karena probabilitas lakunya 80%."*
- **Kategorisasi Label yang Merakyat:** 
  - `PEAK` → "Sangat Sibuk 🔥"
  - `HIGH` → "Ramai 📈"
  - `MEDIUM` → "Biasa Sedang ☕"
  - `LOW` → "Sepi Santai 🍃"

---

## 3. Perombakan Stock AI & Masalah "Insting Musiman"
**Diskusi Masalah:** Saat mendekati hari raya (Lebaran, Natal, dsb), pemilik warung biasanya menggunakan insting mereka untuk restock besar-besaran. Di kondisi ini, model Machine Learning (ML) tradisional yang kita pakai (Ridge, Random Forest) sering kali **diabaikan atau dipandang sebelah mata** karena prediksi angkanya terlalu kecil (ML hanya melihat tren harian historis pendek, tidak tahu kalender Islam/Nasional).

**Apakah LLM bisa jadi solusi?**
Jawabannya: **YA, tapi bertindak sebagai jembatan (overlay), bukan pengganti kalkulator.** 
Berikut adalah perbandingan Solusi Jangka Pendek (Pakai LLM) vs Solusi Jangka Panjang (ML Murni) dari kacamata AI:

### A. Solusi Jangka Pendek (LLM + World Knowledge)
Model ML kita *buta huruf* soal kalender, tapi **LLM itu tahu segalanya tentang dunia nyata**. 
- **Cara Kerja:** Kita mengirimkan data ML + **Tanggal Hari Ini & Info Mendekati Hari Raya** ke LLM.
- **Hasilnya:** LLM akan melihat bahwa ML memprediksi penjualan normal, tapi karena LLM tahu ini mau Lebaran, LLM akan menimpa (override) atau menambahkan nasehat.
- **Output LLM:** *"Meski data harian bilang stok aman, tapi 3 hari lagi Lebaran bos! Insting lu bener, gas restock Sirup Marjan, Biskuit, dan Beras 2x lipat dari biasanya. Jangan dengerin angka normal!"*
- **Kelebihan:** Sangat memvalidasi insting pedagang. Pemilik warung merasa AI ini "mengerti" mereka dan tidak sotoy. AI LLM justru mendukung insting manusia (Human-in-the-Loop).

### B. Solusi Jangka Panjang (Feature Engineering di ML)
Sebagai AI, saya juga menyarankan agar model `stock_ai.py` kita diajari kalender.
- **Cara Kerja:** Kita harus *inject* library kalender libur Indonesia (misal modul Python `holidays`). 
- **Fitur Baru:** Tambahkan fitur seperti `days_until_lebaran`, `is_holiday`, atau `is_ramadan` ke dalam matriks data (X). 
- **Hasilnya:** Dengan data ini, saat musim libur tiba di tahun berikutnya, model Random Forest kita akan otomatis paham: *"Oh fitur `is_holiday` nilainya 1, berarti penjualannya harus dikali lipat."* Ini membuat angka mutlak/range dari AI menjadi akurat kembali di masa depan.

**Kesimpulan untuk Stock AI Saat Ini:**
1. **Rekomendasi Range:** Ubah keluaran angka mutlak jadi *range* (Misal: 40 - 60 item).
2. **Implementasikan LLM Seasonal Overlay:** Gunakan LLM langsung di endpoint `/api/predict/restock/summary` (via param `?include_seasonal=true`) untuk menimpa prediksi ML jika terdeteksi hari libur nasional dalam 14 hari ke depan.

---

## 4. Integrasi LLM: Portofolio Bisnis Mingguan (`llm_insights.py`)
**INI BUKAN PREDIKSI.** Ini adalah murni laporan retrospektif (evaluasi 7 hari ke belakang).

**Ide Implementasi & Prompt Engineering:**
1. **Micro-Summarization di Backend (Hemat Token):**
   - Python merangkum data penjualan 7 hari terakhir. Contoh JSON yang dikirim ke LLM (sangat ringan):
     ```json
     {
       "tanggal_laporan": "18 Mei 2026",
       "total_omset_minggu_ini": 1500000,
       "bintang_warung": [{"nama": "Indomie", "terjual": 50}],
       "produk_kurang_laku": ["Sabun Mandi X"]
     }
     ```
2. **Karakter LLM (Persona):**
   - Set *System Prompt* LLM menjadi karakter yang bersahabat dan membumi.
   - *Prompt:* "Kamu adalah rekan bisnis untuk warung kelontong. Evaluasi performa minggu lalu berdasarkan data ini. Jangan prediksi ke depan."
3. **Eksekusi Teknis:**
   - Endpoint: `/api/insights/generate`
   - Dipanggil oleh **Laravel Cronjob setiap 7 hari** agar tidak boros token API. Hasilnya disimpan di tabel `ai_insights`.

---

## 📋 Langkah Eksekusi Berurutan (To-Do List)

1. ✅ **Clean Up API Responses:**
   - Buka `app/ai/busy_hour_ai.py` & `app/ai/stock_ai.py`.
   - Pindahkan data teknis seperti `accuracy_percent` ke `print()` console backend.
2. ✅ **Refactor Busy Hour AI:**
   - Modifikasi `pred_trx` dan `pred_rev` menjadi format batas bawah dan atas (range).
3. ✅ **Refactor Stock AI (Range & Seasonal Overlay):**
   - Ubah `restock_recommendation` menjadi bentuk range.
   - Tambahkan deteksi hari libur nasional (`detect_upcoming_holidays`).
   - Tambahkan LLM Seasonal Overlay untuk memvalidasi "insting musiman" jelang hari raya.
4. ✅ **Bangun Modul LLM Portofolio (`llm_insights.py`):**
   - Buat modul untuk merangkum performa retrospektif 7 hari terakhir (omset, produk terlaris, hari ramai).
   - Retry logic: Gemini (2x retry) → OpenAI (2x retry) → throw error.
   - Config via `app/helpers/config.py` (bukan `os.getenv` langsung).
   - Error handling: `LLMConfigError` (no API key) & `LLMServiceError` (all failed).
5. **(Opsional) Upgrade Model ML (Solusi Jangka Panjang):**
   - Tambahkan integrasi modul `holidays` Python ke dataset training di `stock_ai.py` untuk mengajari AI tentang libur nasional dan anomali musiman.

---

## 5. Dampak pada Sisi Laravel (Backend Utama & Database)
Perubahan drastis di Python (AI Engine) pasti menuntut penyesuaian di Laravel sebagai jembatan utama ke aplikasi mobile. Berikut adalah gambaran penyesuaian yang mungkin diperlukan:

### A. Penyesuaian Response Format (Controller Laravel)
- Jika Laravel menyimpan hasil prediksi ke database (bukan sekadar passthrough/bypass), maka tipe data yang sebelumnya tunggal (misal `float/integer`) harus diubah untuk menangani objek range.
- **Lebih direkomendasikan:** Jadikan Laravel hanya sebagai *passthrough* untuk hasil prediksi ini, atau simpan hasil JSON mentahnya dalam format kolom `JSON/TEXT` di database Laravel agar schema tidak terlalu sering dibongkar.

### B. Penambahan Tabel Baru untuk Caching LLM (`ai_insights`)
Karena *request* ke LLM (OpenAI/Gemini) menggunakan API berbayar dan memakan waktu (latency beberapa detik), **JANGAN eksekusi LLM setiap kali user membuka aplikasi**.
- **Solusi Database:** Buat tabel baru di Laravel, misal `ai_insights`.
  - `id` (PK)
  - `user_id` (FK ke users, untuk multi-tenant)
  - `insight_type` (Enum: 'weekly_summary', 'stock_warning')
  - `content` (TEXT — nasehat dari LLM)
  - `summary_data` (JSON — data ringkas yang dikirim ke LLM)
  - `source` (VARCHAR — 'gemini' atau 'openai')
  - `valid_until` (DATETIME — insight berlaku sampai kapan, 7 hari dari created_at)
  - `created_at` / `updated_at`
- **Alur Kerja (Cronjob/Task Scheduling — 7 Hari Sekali):** 
  ```
  // app/Console/Kernel.php
  $schedule->call(function () {
      // 1. Ambil data transaksi 30 hari terakhir dari DB
      $transactions = Transaction::with('items.product.stocks')
          ->where('trx_date', '>=', now()->subDays(30))
          ->get()->toArray();

      // 2. Kirim ke Python API
      $response = Http::timeout(120)
          ->post('http://localhost:8080/api/insights/generate', [
              'data' => $transactions,
              'forecast_days' => 14,
          ]);

      // 3. Handle response
      if ($response->successful()) {
          $data = $response->json('data');
          AiInsight::create([
              'user_id'      => $userId,
              'insight_type' => 'weekly_summary',
              'content'      => $data['insight'],
              'summary_data' => json_encode($data['summary']),
              'source'       => $data['source'],
              'valid_until'  => $data['valid_until'],
          ]);
      } else {
          Log::error('AI Insight generation failed', [
              'status' => $response->status(),
              'body'   => $response->json(),
          ]);
      }
  })->weekly()->mondays()->at('06:00');
  ```
  Saat pemilik warung buka aplikasi, Laravel cukup query:
  ```sql
  SELECT * FROM ai_insights
  WHERE user_id = ? AND valid_until >= NOW()
  ORDER BY created_at DESC LIMIT 1
  ```
  Hasilnya instan (0 detik) dan token LLM hanya terpakai 1x per minggu!

### C. Pembuatan Route Baru (Laravel)
- `GET /api/ai/insights` → Mengambil nasehat LLM dari tabel `ai_insights` (baca DB, bukan call Python).
- `GET /api/ai/busy-hours` → (Disesuaikan) untuk mengirimkan format range ke frontend.
- `GET /api/ai/stocks` → (Disesuaikan) untuk mengirimkan format range dan status "Dead Stock" ke frontend.

### D. Error Handling dari Python API
Python API mengembalikan status code yang jelas agar Laravel bisa handle:
| Status | Error Code | Artinya |
|--------|------------|---------|
| 200 | - | Sukses, simpan ke DB |
| 503 | `LLM_CONFIG_ERROR` | API key belum diset di .env Python |
| 502 | `LLM_SERVICE_ERROR` | Semua LLM gagal setelah retry |
| 400 | `DATA_NOT_FOUND` | Tidak ada data transaksi |
| 500 | `INTERNAL_ERROR` | Error tidak terduga |
