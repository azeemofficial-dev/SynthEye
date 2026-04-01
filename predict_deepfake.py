"""
Run deepfake inference for single images or videos.

Image mode:
  python predict_deepfake.py --image path/to/image.jpg

Video mode:
  python predict_deepfake.py --video path/to/video.mp4 --sample_fps 1 --max_frames 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake inference for image or video.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to image file.")
    group.add_argument("--video", type=Path, help="Path to video file.")

    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/deepfake/deepfake_detector.keras"),
        help="Path to .keras model file.",
    )
    parser.add_argument("--img_size", type=int, default=128, help="Square input image size.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on real probability.",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="Frames-per-second to sample from video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=120,
        help="Maximum sampled frames for video inference.",
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for robust mean aggregation in video mode (0 to <0.5).",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"[INFO] Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)


def score_to_label(score: float, threshold: float) -> tuple[str, float]:
    label = "real" if score >= threshold else "fake"
    confidence = score if label == "real" else 1.0 - score
    return label, confidence


def preprocess_image_file(image_path: Path, img_size: int) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(image, dtype=np.float32)
    return arr[None, ...]


def predict_image(model, image_path: Path, img_size: int, threshold: float) -> None:
    batch = preprocess_image_file(image_path, img_size)
    score = float(model.predict(batch, verbose=0).reshape(-1)[0])
    label, confidence = score_to_label(score, threshold)

    print("=" * 52)
    print("DEEPFAKE IMAGE RESULT")
    print("=" * 52)
    print(f"Input       : {image_path}")
    print(f"Prediction  : {label.upper()}")
    print(f"Confidence  : {confidence * 100:.2f}%")
    print(f"Real score  : {score:.4f} (0=fake, 1=real)")
    print("=" * 52)


def trimmed_mean(scores: np.ndarray, trim_ratio: float) -> float:
    if len(scores) == 0:
        raise ValueError("No scores provided.")
    sorted_scores = np.sort(scores)
    trim_count = int(len(sorted_scores) * trim_ratio)
    if trim_count * 2 >= len(sorted_scores):
        trim_count = 0
    core = sorted_scores[trim_count : len(sorted_scores) - trim_count]
    return float(core.mean())


def sample_video_frames(video_path: Path, img_size: int, sample_fps: float, max_frames: int):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "opencv-python is required for video inference. Install dependencies from requirements.txt."
        ) from exc

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if sample_fps <= 0:
        raise ValueError("--sample_fps must be > 0")
    if max_frames <= 0:
        raise ValueError("--max_frames must be > 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    stride = max(int(round(fps / sample_fps)), 1)

    sampled_frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while len(sampled_frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(
                frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA
            )
            sampled_frames.append(frame_resized.astype(np.float32))
            sampled_indices.append(frame_idx)
        frame_idx += 1

    cap.release()

    if not sampled_frames:
        raise ValueError("No frames sampled from video. Check file and sampling params.")

    batch = np.stack(sampled_frames, axis=0)
    return batch, sampled_indices, fps, total_frames


def predict_video(
    model,
    video_path: Path,
    img_size: int,
    threshold: float,
    sample_fps: float,
    max_frames: int,
    trim_ratio: float,
) -> None:
    if trim_ratio < 0 or trim_ratio >= 0.5:
        raise ValueError("--trim_ratio must be in [0, 0.5).")

    batch, frame_indices, source_fps, total_frames = sample_video_frames(
        video_path=video_path,
        img_size=img_size,
        sample_fps=sample_fps,
        max_frames=max_frames,
    )

    frame_scores = model.predict(batch, verbose=0).reshape(-1).astype(float)
    video_score = trimmed_mean(frame_scores, trim_ratio)
    label, confidence = score_to_label(video_score, threshold)

    best_real_idx = int(np.argmax(frame_scores))
    best_fake_idx = int(np.argmin(frame_scores))

    print("=" * 60)
    print("DEEPFAKE VIDEO RESULT")
    print("=" * 60)
    print(f"Input                 : {video_path}")
    print(f"Prediction            : {label.upper()}")
    print(f"Confidence            : {confidence * 100:.2f}%")
    print(f"Aggregated real score : {video_score:.4f} (0=fake, 1=real)")
    print(f"Frames sampled        : {len(frame_scores)} / max {max_frames}")
    print(f"Video fps (reported)  : {source_fps:.2f}")
    if total_frames > 0:
        print(f"Total frames (reported): {total_frames}")
    print(
        f"Most fake-like frame  : #{frame_indices[best_fake_idx]} "
        f"(score={frame_scores[best_fake_idx]:.4f})"
    )
    print(
        f"Most real-like frame  : #{frame_indices[best_real_idx]} "
        f"(score={frame_scores[best_real_idx]:.4f})"
    )
    print("=" * 60)


def main() -> int:
    args = parse_args()

    if np is None:
        raise SystemExit("numpy is not installed. Install dependencies from requirements.txt.")
    if Image is None:
        raise SystemExit("pillow is not installed. Install dependencies from requirements.txt.")
    model = load_model(args.model_path)

    if args.image:
        predict_image(
            model=model,
            image_path=args.image,
            img_size=args.img_size,
            threshold=args.threshold,
        )
    else:
        predict_video(
            model=model,
            video_path=args.video,
            img_size=args.img_size,
            threshold=args.threshold,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            trim_ratio=args.trim_ratio,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
