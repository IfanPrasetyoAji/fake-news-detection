# Fake News Detection API

API deteksi berita palsu menggunakan **TF-IDF Vectorizer** dan **Logistic Regression** yang di-deploy sebagai REST API dengan Flask.

## Overview

Proyek ini menerima teks berita melalui endpoint REST dan mengembalikan prediksi apakah berita tersebut **FAKE** atau **REAL**, lengkap dengan confidence score. Model ML telah di-pre-train dan disimpan dalam format `.pkl`.

## Usage

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

Setelah server berjalan, buka **http://localhost:5000** untuk mengakses antarmuka web.

### Contoh Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm the earth is flat"}'
```

### Contoh Response

```json
{
  "prediction": "FAKE",
  "label": 0,
  "confidence": 0.9312
}
```
