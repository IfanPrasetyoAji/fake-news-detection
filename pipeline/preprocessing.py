import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# Pastikan stopwords terunduh otomatis saat pipeline berjalan
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Membersihkan teks menggunakan Stopwords dan Stemming (Sesuai kodemu)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    words = text.split()  # Tokenize

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords

    # Add stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

def preprocess_and_split_pipeline(df, model_dir="../model"):
    """Menjalankan preprocessing teks dan menyimpan hasil split ke folder model."""
    print("\n--- Step 2: Preprocessing Text & Splitting Data ---")

    print("Menjalankan stemming dan stopword removal (proses ini mungkin memakan waktu)...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Hapus baris yang teksnya kosong setelah diproses
    df = df[df['processed_text'] != ""].reset_index(drop=True)

    # Split menjadi data Train (80%) dan Test (20%)
    df_train, df_test = train_test_split(
        df[['processed_text', 'label']], test_size=0.2, random_state=42, stratify=df['label']
    )

    # Simpan langsung ke dalam 1 folder yaitu 'model/split'
    split_dir = os.path.join(model_dir, "split");
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, "train.csv")
    test_path = os.path.join(split_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Selesai! File berhasil disimpan di folder '{model_dir}':")
    print(f"- {train_path} ({len(df_train)} baris)")
    print(f"- {test_path} ({len(df_test)} baris)")
