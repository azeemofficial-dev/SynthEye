"""
Train a misinformation/news binary classifier (fake vs real) using TF-IDF + Logistic Regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:
    joblib = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    TfidfVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    classification_report = None
    f1_score = None
    train_test_split = None


FAKE_STRINGS = {
    "fake",
    "false",
    "f",
    "rumor",
    "misleading",
    "fabricated",
    "0",
}
REAL_STRINGS = {
    "real",
    "true",
    "t",
    "legit",
    "reliable",
    "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train misinformation detector from a CSV with text and label columns."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/misinfo/news.csv"),
        help="Path to input CSV.",
    )
    parser.add_argument("--text_col", type=str, default="text", help="Text column name.")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name.")
    parser.add_argument(
        "--numeric_labels",
        choices=["0_fake_1_real", "0_real_1_fake"],
        default="0_fake_1_real",
        help="How to interpret numeric labels when present.",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Validation split ratio for holdout set."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_features",
        type=int,
        default=50000,
        help="Max TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/misinfo"),
        help="Output directory for artifacts.",
    )
    return parser.parse_args()


def normalize_label(value, numeric_labels: str) -> str | None:
    if pd.isna(value):
        return None

    raw = str(value).strip().lower()
    if not raw:
        return None

    if raw in {"0", "1"}:
        if numeric_labels == "0_fake_1_real":
            return "fake" if raw == "0" else "real"
        return "real" if raw == "0" else "fake"

    if raw in FAKE_STRINGS:
        return "fake"
    if raw in REAL_STRINGS:
        return "real"
    return None


def load_and_clean_data(
    csv_path: Path,
    text_col: str,
    label_col: str,
    numeric_labels: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in [text_col, label_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {csv_path}: {missing_cols}. Available: {list(df.columns)}"
        )

    cleaned = df[[text_col, label_col]].copy()
    cleaned.columns = ["text", "label"]
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].apply(
        lambda x: normalize_label(x, numeric_labels=numeric_labels)
    )

    cleaned = cleaned[cleaned["text"].str.len() > 0]
    cleaned = cleaned[cleaned["label"].isin(["fake", "real"])]
    cleaned = cleaned.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return cleaned


def main() -> int:
    args = parse_args()

    if joblib is None or pd is None:
        raise SystemExit(
            "Missing dependencies (joblib and/or pandas). Install requirements.txt before running."
        )
    if (
        TfidfVectorizer is None
        or LogisticRegression is None
        or accuracy_score is None
        or classification_report is None
        or f1_score is None
        or train_test_split is None
    ):
        raise SystemExit("scikit-learn is not installed. Install dependencies from requirements.txt.")

    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("--test_size must be between 0 and 1.")

    cleaned = load_and_clean_data(
        csv_path=args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        numeric_labels=args.numeric_labels,
    )
    if cleaned.empty:
        raise ValueError("No valid rows after cleaning. Check text/label columns and label values.")
    if cleaned["label"].nunique() < 2:
        raise ValueError("Need both classes (fake and real) to train.")

    print(f"[INFO] Cleaned samples: {len(cleaned)}")
    print("[INFO] Label distribution:")
    print(cleaned["label"].value_counts())

    x_train, x_val, y_train, y_val = train_test_split(
        cleaned["text"],
        cleaned["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=cleaned["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=2,
        max_df=0.95,
    )
    classifier = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=args.seed,
    )

    print("[INFO] Training TF-IDF vectorizer...")
    x_train_vec = vectorizer.fit_transform(x_train)
    print("[INFO] Training LogisticRegression classifier...")
    classifier.fit(x_train_vec, y_train)

    x_val_vec = vectorizer.transform(x_val)
    y_pred = classifier.predict(x_val_vec)

    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, pos_label="fake")
    report = classification_report(y_val, y_pred, digits=4)

    print("[INFO] Validation report:")
    print(report)
    print(f"[INFO] Validation accuracy: {val_accuracy:.4f}")
    print(f"[INFO] Validation F1 (fake class): {val_f1:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = args.output_dir / "vectorizer.joblib"
    classifier_path = args.output_dir / "classifier.joblib"
    metrics_path = args.output_dir / "metrics.json"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)

    metrics = {
        "samples_total": int(len(cleaned)),
        "samples_train": int(len(x_train)),
        "samples_validation": int(len(x_val)),
        "label_distribution": cleaned["label"].value_counts().to_dict(),
        "validation_accuracy": float(val_accuracy),
        "validation_f1_fake": float(val_f1),
        "numeric_labels": args.numeric_labels,
        "classes": [str(c) for c in classifier.classes_],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[INFO] Saved vectorizer -> {vectorizer_path}")
    print(f"[INFO] Saved classifier -> {classifier_path}")
    print(f"[INFO] Saved metrics    -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
