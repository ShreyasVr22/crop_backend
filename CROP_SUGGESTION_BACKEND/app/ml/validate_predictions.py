import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / ".." / "models" / "crop_model.pkl"
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "Crop_recommendation.csv"


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]

    # train/test split consistent with training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # pick 5 sample rows (same random_state used earlier)
    sample = X_test.sample(5, random_state=1)
    preds = model.predict(sample)

    results = []
    for idx, row in sample.iterrows():
        pred = preds[list(sample.index).index(idx)]
        # distribution of features for predicted class in training set
        class_rows = X_train[y_train == pred]
        if class_rows.empty:
            plaus = False
            details = {f: None for f in feature_cols}
        else:
            z = (row - class_rows.mean()) / class_rows.std()
            within_2sigma = (z.abs() <= 2)
            plaus = within_2sigma.mean() >= 0.6  # >=60% features within 2σ
            details = {f: float(z[f]) for f in feature_cols}

        results.append({
            "index": int(idx),
            "predicted": pred,
            "plausible": bool(plaus),
            "z_scores": details,
        })

    # print concise report
    for r in results:
        print(f"Row {r['index']}: predicted={r['predicted']}, plausible={r['plausible']}")
        z = r['z_scores']
        zs = ", ".join([f"{k}:{z[k]:+.2f}" for k in z])
        print(f"  z-scores: {zs}")


if __name__ == '__main__':
    main()
