"""
Train a deepfake image classifier (fake vs real) using MobileNetV2 transfer learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train deepfake detector on data/deepfake/images/{fake,real}."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/deepfake/images"),
        help="Directory containing fake/ and real/ folders.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Square image size (default: 128 means 128x128).",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for Adam."
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Validation split ratio between 0 and 1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/deepfake"),
        help="Output directory for model and artifacts.",
    )
    return parser.parse_args()


def count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_dataset(data_dir: Path) -> None:
    fake_dir = data_dir / "fake"
    real_dir = data_dir / "real"
    if not fake_dir.exists() or not real_dir.exists():
        raise FileNotFoundError(
            f"Expected folders not found. Required:\n  {fake_dir}\n  {real_dir}"
        )

    fake_count = count_images(fake_dir)
    real_count = count_images(real_dir)
    if fake_count == 0 or real_count == 0:
        raise ValueError(
            "Both classes need at least one image.\n"
            f"Found fake={fake_count}, real={real_count}."
        )
    print(f"[INFO] Dataset counts -> fake: {fake_count}, real: {real_count}")


def build_datasets(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    if validation_split <= 0 or validation_split >= 1:
        raise ValueError("--validation_split must be between 0 and 1.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["fake", "real"],
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["fake", "real"],
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def build_model(img_size: int, learning_rate: float) -> tf.keras.Model:
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="real_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="deepfake_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics.get("accuracy", []), label="train_accuracy")
    plt.plot(epochs, metrics.get("val_accuracy", []), label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics.get("loss", []), label="train_loss")
    plt.plot(epochs, metrics.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()

    if tf is None:
        raise SystemExit(
            "TensorFlow is not installed. Install dependencies from requirements.txt.\n"
            "For deepfake training, use Python 3.10 or 3.11."
        )
    if plt is None:
        raise SystemExit("matplotlib is not installed. Install dependencies from requirements.txt.")
    tf.random.set_seed(args.seed)

    validate_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    model = build_model(img_size=args.img_size, learning_rate=args.learning_rate)
    model_path = args.output_dir / "deepfake_detector.keras"
    curves_path = args.output_dir / "training_curves.png"
    metadata_path = args.output_dir / "metadata.json"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if not model_path.exists():
        model.save(model_path)

    val_loss, val_acc, val_auc = model.evaluate(val_ds, verbose=0)
    save_training_curves(history, curves_path)

    metadata = {
        "model_path": str(model_path),
        "img_size": args.img_size,
        "class_names": {"0": "fake", "1": "real"},
        "threshold": 0.5,
        "validation_metrics": {
            "loss": float(val_loss),
            "accuracy": float(val_acc),
            "auc": float(val_auc),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[INFO] Saved model     -> {model_path}")
    print(f"[INFO] Saved curves    -> {curves_path}")
    print(f"[INFO] Saved metadata  -> {metadata_path}")
    print(
        "[INFO] Validation metrics -> "
        f"loss={val_loss:.4f} accuracy={val_acc:.4f} auc={val_auc:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
