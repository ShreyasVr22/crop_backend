import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"


def resolve_data_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        # look for CSV inside directory
        candidates = list(p.glob('*.csv'))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        return candidates[0]
    if p.is_file():
        return p
    raise FileNotFoundError(f"Data path not found: {p}")


def train_model(data_path: Path, model_out: Path):
    """Train RandomForest model on crop recommendation dataset"""
    print("🌾 Loading dataset...", flush=True)
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    if 'label' not in df.columns:
        raise KeyError('Expected column "label" not found in dataset')
    print(f"Crops available: {sorted(df['label'].unique())}")

    # Features: N, P, K, temperature, humidity, ph, rainfall
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]

    print(f"\nFeatures: {feature_cols}")
    print(f"Number of crops: {len(y.unique())}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: Scale + RandomForest
    print("\n🚀 Training RandomForest model...")
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    print(f"\n✅ Training Accuracy: {train_score:.4f}")
    print(f"✅ Testing Accuracy: {test_score:.4f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"\n💾 Model saved to {model_out}")
    print("✨ Backend is ready to use!")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train crop recommendation model")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to CSV file or directory containing CSV (default: project's data/Crop_recommendation.csv)")
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "models" / "crop_model.pkl"),
                        help="Output path for the saved model")
    args = parser.parse_args(argv)

    try:
        data_file = resolve_data_path(args.data_path)
    except Exception as e:
        print(f"Error resolving data path: {e}", file=sys.stderr)
        sys.exit(1)

    model_out = Path(args.out)
    train_model(data_file, model_out)


if __name__ == "__main__":
    main()