import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(model_dir="../model"):
    """Membaca data train/test dari folder model, melatih model, lalu menyimpannya."""
    print("\n--- Step 3: Model Training & Evaluation ---")

    train_path = os.path.join(model_dir + "/split", "train.csv")
    test_path = os.path.join(model_dir + "/split", "test.csv")

    # Load data
    train_df = pd.read_csv(train_path).dropna()
    test_df = pd.read_csv(test_path).dropna()

    X_train, y_train = train_df['processed_text'], train_df['label']
    X_test, y_test = test_df['processed_text'], test_df['label']

    # Membuat Scikit-Learn Pipeline agar Vectorizer dan Model menyatu saat di-eksport
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    print("Melatih model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluasi hasil akhir
    y_pred = model_pipeline.predict(X_test)
    print(f"\nAkurasi Model Akhir: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))

    # Menyimpan file model .pkl langsung ke folder model
    model_output_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model_pipeline, model_output_path)
    print(f"Model .pkl sukses disimpan di: {model_output_path}")
