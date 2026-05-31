import os
import pandas as pd

def ingest_data(dataset_dir="dataset"):
    """Membaca Fake.csv dan True.csv dari folder model, memberi label, dan mengacak data."""
    fake_path = os.path.join(dataset_dir, "Fake.csv")
    true_path = os.path.join(dataset_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"Pastikan Fake.csv dan True.csv ada di folder '{dataset_dir}'")

    print("--- Step 1: Ingesting & Shuffling Data ---")
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)

    fake['label'] = 0
    real['label'] = 1

    # Menggabungkan dan mengacak data sesuai kodemu
    df = pd.concat([fake, real], ignore_index=True).sample(frac=1, random_state=42)

    # Menghapus kolom date sesuai kodemu
    if 'date' in df.columns:
        df.drop(columns='date', inplace=True)

    print("Distribusi Kelas:")
    print(df['label'].value_counts())
    return df
