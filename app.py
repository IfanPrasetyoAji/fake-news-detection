"""
app.py — Fake News Detection API
=================================
Flask web service yang meng-expose model Machine Learning sebagai REST API.
Model yang digunakan adalah sklearn Pipeline (TF-IDF + Logistic Regression)
yang sudah di-train sebelumnya dan disimpan sebagai file .pkl.

Endpoints:
    GET  /         → Halaman frontend (index.html)
    GET  /health   → Mengecek status API dan model
    POST /predict  → Menerima teks berita, mengembalikan prediksi FAKE/REAL

Cara menjalankan:
    python app.py

Cara test via curl:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Scientists confirm the earth is flat"}'
"""

import os
import re
import nltk
import joblib
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from pipeline.data_ingestion import * 
from pipeline.preprocessing import * 
from pipeline.model_training import *

# ── App Initialization ────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Mengizinkan request dari origin manapun (termasuk frontend lokal)


# ── NLTK Setup ────────────────────────────────────────────────────────────────

# Download stopwords otomatis jika belum tersedia di environment
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# ── Model Loading / Training ──────────────────────────────────────────────────

model_path = "model/model.pkl"
if not os.path.exists(model_path):
    print("Model tidak ditemukan. Melatih model baru...")

    try:
        raw_data = ingest_data("dataset")
        preprocess_and_split_pipeline(raw_data, model_dir="model")
        train_and_evaluate(model_dir="model")

        print(f"\n[SUCCESS] Automated Pipeline selesai! Cek folder 'model' untuk melihat hasilnya.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Pipeline terhenti karena file tidak ditemukan: {str(e)}")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Pipeline terhenti akibat kendala tak terduga: {str(e)}")
        sys.exit(1)

model_pipeline = joblib.load(model_path)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Menampilkan halaman frontend dari folder templates/."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint untuk memastikan API dan model berjalan normal.

    Returns:
        JSON: Status API dan nama file model yang digunakan.

    Example response:
        {
            "status": "ok",
            "model": "fake_news_model.pkl"
        }
    """
    return jsonify({
        "status": "ok",
        "model": "fake_news_model.pkl"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Menerima teks berita dan mengembalikan hasil prediksi FAKE atau REAL.

    Request body (JSON):
        {
            "text": "isi berita yang ingin dicek"
        }

    Returns:
        JSON: Hasil prediksi, label numerik, dan confidence score.

    Example response (success):
        {
            "prediction": "FAKE",
            "label": 0,
            "confidence": 0.9312
        }

    Example response (error):
        {
            "error": "Field 'text' is required"
        }

    Status codes:
        200 — Prediksi berhasil
        400 — Request tidak valid (field kosong / missing)
        500 — Server error tak terduga
    """
    try:
        data = request.get_json()

        # Validasi: body harus ada dan mengandung field "text"
        if not data or "text" not in data:
            return jsonify({"error": "Field 'text' is required"}), 400

        text = data["text"]

        # Validasi: teks tidak boleh kosong
        if not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400

        # Preprocessing → predict (pipeline handle TF-IDF + model sekaligus)
        cleaned_text = preprocess_text(text)
        prediction_label = model_pipeline.predict([cleaned_text])[0]
        confidence = model_pipeline.predict_proba([cleaned_text]).max()

        # Konversi label numerik ke string (0 = Fake, 1 = Real)
        result = "REAL" if prediction_label == 1 else "FAKE"

        return jsonify({
            "prediction": result,
            "label": int(prediction_label),
            "confidence": round(float(confidence), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # host="0.0.0.0" agar bisa diakses dari luar container (penting untuk Docker)
    # debug=False untuk production/submission
    app.run(host="0.0.0.0", port=5000, debug=False)
