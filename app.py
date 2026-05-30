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

import re
import nltk
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ── App Initialization ────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Mengizinkan request dari origin manapun (termasuk frontend lokal)


# ── NLTK Setup ────────────────────────────────────────────────────────────────

# Download stopwords otomatis jika belum tersedia di environment
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# ── Model Loading ─────────────────────────────────────────────────────────────

# Load sklearn Pipeline (TF-IDF Vectorizer + Logistic Regression dalam 1 file)
# Pipeline ini di-generate oleh src/model_training.py
model_pipeline = joblib.load("model/fake_news_model.pkl")


# ── Helper Functions ──────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Membersihkan dan memproses teks input sebelum dimasukkan ke model.

    Langkah preprocessing (harus identik dengan src/preprocessing.py):
        1. Lowercase — menyamakan kapitalisasi
        2. Remove non-alphabetic — buang angka, tanda baca, simbol
        3. Tokenize — pisah jadi list kata
        4. Remove stopwords — buang kata umum (the, is, at, ...)
        5. Stemming — potong kata ke bentuk dasarnya (running → run)

    Args:
        text (str): Teks berita mentah dari user.

    Returns:
        str: Teks yang sudah bersih dan siap di-vectorize.
    """
    if not isinstance(text, str):
        return ""

    # Step 1 & 2: Lowercase + hapus karakter non-huruf
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 3: Tokenize
    words = text.split()

    # Step 4: Hapus stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 5: Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)


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
