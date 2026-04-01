"""
Run misinformation/news inference using saved TF-IDF vectorizer + LogisticRegression classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:
    joblib = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fake/real for text input.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Input text to classify.")
    group.add_argument("--file", type=Path, help="Path to a .txt file containing input text.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/misinfo"),
        help="Directory containing vectorizer.joblib and classifier.joblib.",
    )
    return parser.parse_args()


def read_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text.strip()
    if not args.file.exists():
        raise FileNotFoundError(f"Text file not found: {args.file}")
    text = args.file.read_text(encoding="utf-8").strip()
    return text


def main() -> int:
    args = parse_args()

    if joblib is None:
        raise SystemExit("joblib is not installed. Install dependencies from requirements.txt.")
    text = read_input_text(args)
    if not text:
        raise ValueError("Input text is empty.")

    vectorizer_path = args.model_dir / "vectorizer.joblib"
    classifier_path = args.model_dir / "classifier.joblib"
    if not vectorizer_path.exists() or not classifier_path.exists():
        raise FileNotFoundError(
            "Model artifacts missing. Expected:\n"
            f"  {vectorizer_path}\n"
            f"  {classifier_path}\n"
            "Run train_misinfo.py first."
        )

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    x_vec = vectorizer.transform([text])
    prediction = str(classifier.predict(x_vec)[0])

    proba_map = {}
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(x_vec)[0]
        for label, prob in zip(classifier.classes_, probs):
            proba_map[str(label)] = float(prob)

    confidence = max(proba_map.values()) if proba_map else 0.0
    fake_prob = proba_map.get("fake", 0.0)
    real_prob = proba_map.get("real", 0.0)

    print("=" * 52)
    print("MISINFORMATION TEXT RESULT")
    print("=" * 52)
    print(f"Prediction       : {prediction.upper()}")
    print(f"Confidence       : {confidence * 100:.2f}%")
    print(f"Fake probability : {fake_prob:.4f}")
    print(f"Real probability : {real_prob:.4f}")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
