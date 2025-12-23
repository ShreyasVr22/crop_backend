import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / ".." / "models" / "crop_model.pkl"
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "Crop_recommendation.csv"


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # show 5 sample predictions
    sample = X_test.sample(5, random_state=1)
    preds = model.predict(sample)
    print("Sample predictions:")
    out = sample.copy()
    out["predicted"] = preds
    print(out)


if __name__ == '__main__':
    main()
