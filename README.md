# Fake News Detection API

API deteksi berita palsu menggunakan **TF-IDF Vectorizer** dan **Logistic Regression** yang di-deploy sebagai REST API dengan Flask.

> **Base URL**: `http://localhost:5000`  
> **Auth**: None (open API)

## Overview

Proyek ini adalah pipeline **training + inference** untuk mendeteksi berita palsu menggunakan **TF-IDF Vectorizer** dan **Logistic Regression**, di-deploy sebagai REST API dengan Flask.

Jika model `.pkl` belum tersedia, aplikasi akan secara otomatis menjalankan pipeline dari awal: **ingest data → preprocessing → train → evaluate → save model**, lalu memulai server.

## Custom Dataset

Anda dapat melatih model dengan dataset sendiri. Pipeline akan otomatis mendeteksi dan melatih ulang model saat aplikasi dijalankan.

### Struktur File

Letakkan dua file CSV di folder `dataset/`:

| File | Label |
|------|-------|
| `Fake.csv` | Berita palsu (label `0`) |
| `True.csv` | Berita asli (label `1`) |

### Format Kolom

Kedua file harus memiliki kolom-kolom berikut:

| Column | Type | Description |
|--------|------|-------------|
| `title` | string | Judul berita |
| `text` | string | Isi/konten berita |
| `subject` | string | Kategori topik |
| `date` | string | Tanggal publikasi |

> **Catatan**: Pipeline menggunakan kolom `text` sebagai fitur utama untuk training. Kolom `date` otomatis di-drop saat preprocessing.

### Contoh Baris CSV

```csv
title,text,subject,date
"Scientists confirm the earth is flat","A new study claims the earth is flat...",politics,January 1 2020
"Stock market hits new high","The stock market reached an all-time high today...",business,January 2 2020
```

## Local Development

### Clone dan buka repository
```bash
git clone git@github.com:IfanPrasetyoAji/fake-news-detection.git
cd fake-news-detection
```

### Menjalankan dengan Python
```bash
pip install -r requirements.txt
python app.py
```

### Menjalankan dengan Docker
```bash
docker compose up
```

Setelah server berjalan, buka **http://localhost:5000** untuk mengakses web interface.

## API Endpoints

### `GET /health`

Health check endpoint untuk memastikan API dan model berjalan.

**Response `200`**:
```json
{
  "status": "ok",
  "model": "fake_news_model.pkl"
}
```

---

### `POST /predict`

Mengirim teks berita untuk diprediksi.

**Request body** (JSON):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Teks berita yang akan dianalisis |

**Response `200`**:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"FAKE"` atau `"REAL"` |
| `label` | int | `0` (FAKE) atau `1` (REAL) |
| `confidence` | float | Skor kepercayaan model (0.0 - 1.0) |

**Error Responses**:

- `400` — `{"error": "No text provided"}` (saat field `text` kosong atau tidak dikirim)
- `500` — `{"error": "Prediction failed"}` (saat terjadi error internal)

---

## Examples

### Prediksi (curl)
```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm the earth is flat"}'
```
```json
{
  "prediction": "FAKE",
  "label": 0,
  "confidence": 0.9312
}
```

### Health Check (curl)
```bash
curl -s http://localhost:5000/health
```
```json
{
  "status": "ok",
  "model": "fake_news_model.pkl"
}
```

### Prediksi (Python)
```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "Scientists confirm the earth is flat"}
)
data = response.json()

print(f"Prediction: {data['prediction']}")
print(f"Confidence: {data['confidence']:.2%}")
```

### Prediksi (JavaScript)
```javascript
const response = await fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Scientists confirm the earth is flat" })
});
const data = await response.json();
console.log(`Prediction: ${data.prediction}`);
```

## Tech Stack

- **Framework**: Flask
- **ML Pipeline**: TF-IDF Vectorizer + Logistic Regression (scikit-learn)
- **Text Preprocessing**: NLTK (stopword removal, Porter Stemmer)
- **Deployment**: Docker, Gunicorn
- **CORS**: Enabled (all origins)
