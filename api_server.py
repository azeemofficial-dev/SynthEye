"""
SynthEye FastAPI backend:
- Deepfake image/video inference
- Misinformation text inference
- Serves syntHeye.html at "/"
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from collections import defaultdict, deque
from contextlib import suppress
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from starlette.middleware.sessions import SessionMiddleware


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEEPFAKE_MODEL_PATH = ROOT / "models" / "deepfake" / "deepfake_detector.keras"
DEEPFAKE_METADATA_PATH = ROOT / "models" / "deepfake" / "metadata.json"
MISINFO_MODEL_DIR = ROOT / "models" / "misinfo"
MISINFO_VECTORIZER_PATH = MISINFO_MODEL_DIR / "vectorizer.joblib"
MISINFO_CLASSIFIER_PATH = MISINFO_MODEL_DIR / "classifier.joblib"
MISINFO_METRICS_PATH = MISINFO_MODEL_DIR / "metrics.json"
SIGNUP_PAGE_PATH = ROOT / "signup.html"
LOGIN_PAGE_PATH = ROOT / "login.html"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
TEXT_EXTS = {".txt"}

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PBKDF2_ITERATIONS = 390_000
APP_ENV = os.getenv("SYNTHEYE_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV in {"prod", "production"}
SESSION_TTL_SECONDS = int(os.getenv("SYNTHEYE_SESSION_TTL_SECONDS", "86400"))
REQUIRE_AUTH = os.getenv("SYNTHEYE_REQUIRE_AUTH", "1").strip().lower() not in {"0", "false", "no"}
SESSION_HTTPS_ONLY = os.getenv("SYNTHEYE_SESSION_HTTPS_ONLY", "0").strip().lower() in {"1", "true", "yes"}
SESSION_SECRET = os.getenv("SYNTHEYE_SESSION_SECRET")
LOCAL_SQLITE_PATH = DATA_DIR / "syntheye.db"
DEFAULT_DATABASE_URL = f"sqlite:///{LOCAL_SQLITE_PATH.as_posix()}"
MAX_UPLOAD_MB = max(1, int(os.getenv("SYNTHEYE_MAX_UPLOAD_MB", "50")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_TEXT_CHARS = int(os.getenv("SYNTHEYE_MAX_TEXT_CHARS", "20000"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("SYNTHEYE_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_AUTH = int(os.getenv("SYNTHEYE_RATE_LIMIT_AUTH_PER_WINDOW", "20"))
RATE_LIMIT_ANALYZE = int(os.getenv("SYNTHEYE_RATE_LIMIT_ANALYZE_PER_WINDOW", "60"))
RATE_LIMIT_GLOBAL = int(os.getenv("SYNTHEYE_RATE_LIMIT_GLOBAL_PER_WINDOW", "240"))
TRUST_PROXY_HEADERS = os.getenv("SYNTHEYE_TRUST_PROXY_HEADERS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
LOG_LEVEL = os.getenv("SYNTHEYE_LOG_LEVEL", "INFO").strip().upper()
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = Path(os.getenv("SYNTHEYE_LOG_FILE", str(LOG_DIR / "syntheye.log")))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "SYNTHEYE_ALLOW_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]
if not SESSION_SECRET:
    SESSION_SECRET = "dev-only-change-this-before-deploy"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("syntheye")
    if logger.handlers:
        return logger
    with suppress(OSError):
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    with suppress(OSError):
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = setup_logging()
security_warnings: list[str] = []


def validate_allowed_origins(origins: list[str]) -> list[str]:
    if not origins:
        return ["No CORS origins configured."]
    issues: list[str] = []
    for origin in origins:
        if origin == "*" and IS_PRODUCTION:
            issues.append("Wildcard CORS origin '*' is not allowed in production.")
        if IS_PRODUCTION and ("localhost" in origin or "127.0.0.1" in origin):
            issues.append(f"Localhost origin not allowed in production: {origin}")
        if IS_PRODUCTION and not origin.startswith("https://"):
            issues.append(f"Production CORS origin must use HTTPS: {origin}")
    return issues


def is_weak_session_secret(secret: str) -> bool:
    if secret in {"dev-only-change-this-before-deploy", "change-this-in-production", "replace-me"}:
        return True
    return len(secret) < 32


def validate_security_settings() -> None:
    issues: list[str] = []
    if is_weak_session_secret(SESSION_SECRET):
        issues.append("SYNTHEYE_SESSION_SECRET is weak. Use a random secret >= 32 chars.")
    if IS_PRODUCTION and not SESSION_HTTPS_ONLY:
        issues.append("SYNTHEYE_SESSION_HTTPS_ONLY must be 1 in production.")
    if IS_PRODUCTION and DATABASE_KIND != "mysql":
        issues.append("Production mode requires MySQL. Set SYNTHEYE_DATABASE_URL to mysql+pymysql://...")
    issues.extend(validate_allowed_origins(ALLOWED_ORIGINS))

    if IS_PRODUCTION and issues:
        raise RuntimeError("Invalid production security config: " + " | ".join(issues))
    security_warnings.clear()
    if issues:
        security_warnings.extend(issues)
        for issue in issues:
            logger.warning(issue)


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        if limit <= 0:
            return True
        with self._lock:
            bucket = self._events[key]
            cutoff = now - window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


rate_limiter = InMemoryRateLimiter()
metrics_lock = threading.Lock()
metrics = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "requests_total": 0,
    "requests_error_total": 0,
    "rate_limited_total": 0,
    "analyze_requests_total": 0,
    "upload_rejected_total": 0,
    "analyze_duration_ms_total": 0,
    "analyze_duration_samples": 0,
}


def record_metric(key: str, delta: int = 1) -> None:
    with metrics_lock:
        metrics[key] = int(metrics.get(key, 0)) + delta


def record_analyze_duration(duration_ms: int) -> None:
    with metrics_lock:
        metrics["analyze_duration_ms_total"] = int(metrics.get("analyze_duration_ms_total", 0)) + duration_ms
        metrics["analyze_duration_samples"] = int(metrics.get("analyze_duration_samples", 0)) + 1


def build_database_url() -> str:
    direct_url = os.getenv("SYNTHEYE_DATABASE_URL", "").strip()
    if direct_url:
        return direct_url

    db_host = os.getenv("SYNTHEYE_DB_HOST", "").strip()
    db_user = os.getenv("SYNTHEYE_DB_USER", "").strip()
    db_password = os.getenv("SYNTHEYE_DB_PASSWORD", "")
    db_name = os.getenv("SYNTHEYE_DB_NAME", "").strip()
    db_port = os.getenv("SYNTHEYE_DB_PORT", "3306").strip()

    if db_host and db_user and db_name:
        return (
            "mysql+pymysql://"
            f"{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
        )
    return DEFAULT_DATABASE_URL


DATABASE_URL = build_database_url()
DATABASE_KIND = "mysql" if DATABASE_URL.startswith("mysql+") else "sqlite"
with suppress(Exception):
    DATABASE_SAFE_URL = make_url(DATABASE_URL).render_as_string(hide_password=True)
if "DATABASE_SAFE_URL" not in globals():
    DATABASE_SAFE_URL = DATABASE_URL
DATABASE_CONNECT_ARGS: dict[str, Any] = {"check_same_thread": False} if DATABASE_KIND == "sqlite" else {}

db_engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=DATABASE_CONNECT_ARGS,
)
DBSessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(String(40), nullable=False)
    last_login_at = Column(String(40), nullable=True)


class AnalysisLogModel(Base):
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    task = Column(String(64), nullable=False, index=True)
    filename = Column(String(512), nullable=True)
    prediction = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String(16), nullable=False, default="success")
    error_detail = Column(Text, nullable=True)
    request_meta = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(String(40), nullable=False)


deepfake_model: Any | None = None
misinfo_vectorizer = None
misinfo_classifier = None
tf_module = None
cv2_module = None
deepfake_model_lock = threading.Lock()
misinfo_artifacts_lock = threading.Lock()
db_init_lock = threading.Lock()
db_initialized = False

app = FastAPI(title="SynthEye API", version="1.0.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=SESSION_TTL_SECONDS,
    same_site="lax",
    https_only=SESSION_HTTPS_ONLY,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    record_metric("requests_total")
    enforce_rate_limit(request, bucket="global", limit=RATE_LIMIT_GLOBAL)
    request_id = secrets.token_hex(8)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except HTTPException:
        record_metric("requests_error_total")
        raise
    except Exception:
        record_metric("requests_error_total")
        logger.exception("Unhandled error request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse({"detail": "Internal server error."}, status_code=500, headers={"X-Request-ID": request_id})

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if response.status_code >= 400:
        record_metric("requests_error_total")
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


class TextRequest(BaseModel):
    text: str


class SignupRequest(BaseModel):
    full_name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_database() -> None:
    Base.metadata.create_all(bind=db_engine)


def ensure_database() -> None:
    global db_initialized
    if db_initialized:
        return
    with db_init_lock:
        if db_initialized:
            return
        try:
            init_database()
        except SQLAlchemyError as exc:
            raise RuntimeError(f"Database initialization failed for {DATABASE_SAFE_URL}: {exc}") from exc
        db_initialized = True


def db_session() -> Session:
    ensure_database()
    return DBSessionLocal()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_password_strength(password: str) -> list[str]:
    issues: list[str] = []
    if len(password) < 8:
        issues.append("Password must be at least 8 characters.")
    if not any(c.islower() for c in password):
        issues.append("Password must include a lowercase letter.")
    if not any(c.isupper() for c in password):
        issues.append("Password must include an uppercase letter.")
    if not any(c.isdigit() for c in password):
        issues.append("Password must include a digit.")
    return issues


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, rounds_str, salt_b64, digest_b64 = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        rounds = int(rounds_str)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(actual, expected)


def get_user_by_email(email: str) -> UserModel | None:
    clean_email = normalize_email(email)
    with db_session() as session:
        stmt = select(UserModel).where(UserModel.email == clean_email)
        return session.execute(stmt).scalar_one_or_none()


def get_user_by_id(user_id: int) -> UserModel | None:
    with db_session() as session:
        return session.get(UserModel, user_id)


def create_user(full_name: str, email: str, password: str) -> UserModel:
    clean_name = full_name.strip()
    clean_email = normalize_email(email)
    if len(clean_name) < 2:
        raise HTTPException(status_code=400, detail="Full name must be at least 2 characters.")
    if not EMAIL_REGEX.match(clean_email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    password_issues = validate_password_strength(password)
    if password_issues:
        raise HTTPException(status_code=400, detail=" ".join(password_issues))

    if get_user_by_email(clean_email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    password_hash = hash_password(password)
    new_user = UserModel(
        full_name=clean_name,
        email=clean_email,
        password_hash=password_hash,
        created_at=utc_now_iso(),
        last_login_at=None,
    )
    with db_session() as session:
        session.add(new_user)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
        session.refresh(new_user)
    return new_user


def authenticate_user(email: str, password: str) -> UserModel:
    user = get_user_by_email(email)
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return user


def mark_login(user_id: int) -> None:
    with db_session() as session:
        user = session.get(UserModel, user_id)
        if user is None:
            return
        user.last_login_at = utc_now_iso()
        session.commit()


def serialize_user(user: UserModel) -> dict:
    return {
        "id": int(user.id),
        "full_name": str(user.full_name),
        "email": str(user.email),
        "created_at": str(user.created_at),
        "last_login_at": user.last_login_at,
    }


def get_session_user_id(request: Request) -> int | None:
    raw_user_id = request.session.get("user_id")
    if raw_user_id is None:
        return None
    try:
        return int(raw_user_id)
    except (TypeError, ValueError):
        request.session.clear()
        return None


def get_session_user(request: Request) -> dict | None:
    user_id = get_session_user_id(request)
    if user_id is None:
        return None
    row = get_user_by_id(user_id)
    if row is None:
        request.session.clear()
        return None
    return serialize_user(row)


def require_session_user(request: Request) -> dict:
    user = get_session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in at /login.")
    return user


def get_client_identifier(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request, bucket: str, limit: int) -> None:
    if limit <= 0:
        return
    client = get_client_identifier(request)
    key = f"{bucket}:{client}"
    if not rate_limiter.allow(key, limit=limit, window_seconds=RATE_LIMIT_WINDOW_SECONDS):
        record_metric("rate_limited_total")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry shortly.")


async def read_upload_with_limit(file: UploadFile, limit_bytes: int) -> bytes:
    content = bytearray()
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > limit_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.",
            )
    return bytes(content)


def serialize_analysis_log(log_item: AnalysisLogModel) -> dict:
    result_payload: dict[str, Any] | None = None
    if log_item.result_json:
        with suppress(json.JSONDecodeError):
            result_payload = json.loads(log_item.result_json)
    request_meta: dict[str, Any] | None = None
    if log_item.request_meta:
        with suppress(json.JSONDecodeError):
            request_meta = json.loads(log_item.request_meta)
    return {
        "id": int(log_item.id),
        "user_id": log_item.user_id,
        "task": log_item.task,
        "filename": log_item.filename,
        "prediction": log_item.prediction,
        "confidence": log_item.confidence,
        "status": log_item.status,
        "error_detail": log_item.error_detail,
        "request_meta": request_meta,
        "result": result_payload,
        "duration_ms": log_item.duration_ms,
        "created_at": log_item.created_at,
    }


def get_recent_analysis_logs(user_id: int | None, limit: int = 20) -> tuple[list[dict], int]:
    safe_limit = max(1, min(limit, 100))
    with db_session() as session:
        stmt = select(AnalysisLogModel)
        count_stmt = select(func.count(AnalysisLogModel.id))
        if user_id is not None:
            stmt = stmt.where(AnalysisLogModel.user_id == user_id)
            count_stmt = count_stmt.where(AnalysisLogModel.user_id == user_id)
        total_count = int(session.execute(count_stmt).scalar_one())
        rows = session.execute(stmt.order_by(AnalysisLogModel.id.desc()).limit(safe_limit)).scalars().all()
    return [serialize_analysis_log(row) for row in rows], total_count


def log_analysis_event(
    request: Request | None,
    *,
    task: str,
    filename: str | None = None,
    result: dict[str, Any] | None = None,
    error_detail: str | None = None,
    request_meta: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> None:
    user_id = get_session_user_id(request) if request else None
    prediction = result.get("prediction") if result else None
    confidence = result.get("confidence") if result else None
    confidence_value = float(confidence) if isinstance(confidence, (float, int)) else None
    safe_result_json = None
    safe_meta_json = None

    if result is not None:
        with suppress(TypeError, ValueError):
            safe_result_json = json.dumps(result, ensure_ascii=False)
    if request_meta is not None:
        with suppress(TypeError, ValueError):
            safe_meta_json = json.dumps(request_meta, ensure_ascii=False)

    row = AnalysisLogModel(
        user_id=user_id,
        task=task,
        filename=filename,
        prediction=str(prediction) if prediction is not None else None,
        confidence=confidence_value,
        status="error" if error_detail else "success",
        error_detail=error_detail,
        request_meta=safe_meta_json,
        result_json=safe_result_json,
        duration_ms=duration_ms,
        created_at=utc_now_iso(),
    )
    with suppress(SQLAlchemyError):
        with db_session() as session:
            session.add(row)
            session.commit()


def get_database_stats() -> dict:
    try:
        with db_session() as session:
            users_count = int(session.execute(select(func.count(UserModel.id))).scalar_one())
            logs_count = int(session.execute(select(func.count(AnalysisLogModel.id))).scalar_one())
        return {
            "kind": DATABASE_KIND,
            "url": DATABASE_SAFE_URL,
            "ready": True,
            "users_count": users_count,
            "analysis_logs_count": logs_count,
        }
    except Exception as exc:
        return {
            "kind": DATABASE_KIND,
            "url": DATABASE_SAFE_URL,
            "ready": False,
            "error": str(exc),
        }


@app.on_event("startup")
def startup() -> None:
    validate_security_settings()
    ensure_database()
    logger.info(
        "SynthEye startup env=%s db=%s auth_required=%s https_only=%s",
        APP_ENV,
        DATABASE_SAFE_URL,
        REQUIRE_AUTH,
        SESSION_HTTPS_ONLY,
    )


def get_tensorflow():
    global tf_module
    if tf_module is None:
        try:
            import tensorflow as tf  # type: ignore
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="TensorFlow is not installed. Deepfake endpoints require TensorFlow.",
            ) from exc
        tf_module = tf
    return tf_module


def get_cv2():
    global cv2_module
    if cv2_module is None:
        try:
            import cv2  # type: ignore
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="opencv-python is not installed. Video endpoint requires OpenCV.",
            ) from exc
        cv2_module = cv2
    return cv2_module


def get_deepfake_model():
    global deepfake_model
    if deepfake_model is None:
        with deepfake_model_lock:
            if deepfake_model is None:
                if not DEEPFAKE_MODEL_PATH.exists():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Deepfake model not found at {DEEPFAKE_MODEL_PATH}",
                    )
                tf = get_tensorflow()
                deepfake_model = tf.keras.models.load_model(DEEPFAKE_MODEL_PATH)
    return deepfake_model


def get_misinfo_artifacts():
    global misinfo_vectorizer, misinfo_classifier
    if misinfo_vectorizer is None or misinfo_classifier is None:
        with misinfo_artifacts_lock:
            if misinfo_vectorizer is None or misinfo_classifier is None:
                if not MISINFO_VECTORIZER_PATH.exists() or not MISINFO_CLASSIFIER_PATH.exists():
                    raise HTTPException(
                        status_code=500,
                        detail="Misinformation artifacts are missing. Train train_misinfo.py first.",
                    )
                misinfo_vectorizer = joblib.load(MISINFO_VECTORIZER_PATH)
                misinfo_classifier = joblib.load(MISINFO_CLASSIFIER_PATH)
    return misinfo_vectorizer, misinfo_classifier


def warmup_deepfake_model() -> dict:
    started = time.perf_counter()
    _ = get_deepfake_model()
    elapsed = time.perf_counter() - started
    return {
        "status": "ready",
        "loaded": True,
        "load_seconds": round(elapsed, 3),
        "model_path": str(DEEPFAKE_MODEL_PATH),
    }


def score_to_label(score: float, threshold: float = 0.5) -> tuple[str, float]:
    label = "real" if score >= threshold else "fake"
    confidence = score if label == "real" else (1.0 - score)
    return label, confidence


def predict_image_bytes(image_bytes: bytes, filename: str) -> dict:
    model = get_deepfake_model()
    try:
        image = Image.open(np_to_filelike(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    image = image.resize((128, 128))
    arr = np.asarray(image, dtype=np.float32)[None, ...]
    score = float(model.predict(arr, verbose=0).reshape(-1)[0])
    label, confidence = score_to_label(score)
    return {
        "task": "deepfake_image",
        "filename": filename,
        "prediction": label,
        "confidence": round(confidence, 4),
        "real_score": round(score, 4),
    }


def np_to_filelike(binary: bytes):
    # Tiny helper to let PIL open bytes without touching disk.
    import io

    return io.BytesIO(binary)


def trimmed_mean(scores: np.ndarray, trim_ratio: float = 0.1) -> float:
    ordered = np.sort(scores)
    k = int(len(ordered) * trim_ratio)
    if k * 2 >= len(ordered):
        k = 0
    core = ordered[k : len(ordered) - k]
    return float(core.mean())


def predict_video_file(video_path: Path, filename: str) -> dict:
    cv2 = get_cv2()
    model = get_deepfake_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    stride = max(int(round(fps / 1.0)), 1)  # sample_fps=1

    frames: list[np.ndarray] = []
    indices: list[int] = []
    idx = 0
    max_frames = 120

    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            frames.append(frame_small.astype(np.float32))
            indices.append(idx)
        idx += 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if not frames:
        raise HTTPException(status_code=400, detail="No frames sampled from uploaded video.")

    batch = np.stack(frames, axis=0)
    scores = model.predict(batch, verbose=0).reshape(-1).astype(float)
    video_score = trimmed_mean(scores, trim_ratio=0.1)
    label, confidence = score_to_label(video_score)

    return {
        "task": "deepfake_video",
        "filename": filename,
        "prediction": label,
        "confidence": round(confidence, 4),
        "real_score": round(video_score, 4),
        "frames_sampled": int(len(scores)),
        "total_frames": total_frames,
        "fps": round(float(fps), 2),
        "most_fake_like_frame": int(indices[int(np.argmin(scores))]),
        "most_real_like_frame": int(indices[int(np.argmax(scores))]),
    }


def predict_text(text: str) -> dict:
    vectorizer, classifier = get_misinfo_artifacts()
    clean = text.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Text is empty.")

    x = vectorizer.transform([clean])
    prediction = str(classifier.predict(x)[0])
    probs = classifier.predict_proba(x)[0]
    classes = [str(c) for c in classifier.classes_]
    proba_map = {label: float(prob) for label, prob in zip(classes, probs)}

    return {
        "task": "misinfo_text",
        "prediction": prediction,
        "confidence": round(max(proba_map.values()), 4),
        "fake_probability": round(proba_map.get("fake", 0.0), 4),
        "real_probability": round(proba_map.get("real", 0.0), 4),
    }


def detect_file_kind(suffix: str, content_type: str | None) -> str | None:
    content = (content_type or "").lower()
    if suffix in IMAGE_EXTS or content.startswith("image/"):
        return "image"
    if suffix in VIDEO_EXTS or content.startswith("video/"):
        return "video"
    if suffix in TEXT_EXTS or content.startswith("text/"):
        return "text"
    return None


def has_route(path: str) -> bool:
    for route in app.router.routes:
        if getattr(route, "path", None) == path:
            return True
    return False


def get_model_stats() -> dict:
    deepfake_meta = read_json_file(DEEPFAKE_METADATA_PATH) or {}
    deepfake_val = deepfake_meta.get("validation_metrics", {})
    deepfake_accuracy = deepfake_val.get("accuracy")
    deepfake_auc = deepfake_val.get("auc")

    misinfo_meta = read_json_file(MISINFO_METRICS_PATH) or {}
    misinfo_accuracy = misinfo_meta.get("validation_accuracy")
    misinfo_f1_fake = misinfo_meta.get("validation_f1_fake")
    misinfo_samples = misinfo_meta.get("samples_total")

    deepfake_available = DEEPFAKE_MODEL_PATH.exists()
    misinfo_available = MISINFO_VECTORIZER_PATH.exists() and MISINFO_CLASSIFIER_PATH.exists()
    active_models = int(deepfake_available) + int(misinfo_available)

    return {
        "deepfake": {
            "available": deepfake_available,
            "validation_accuracy": deepfake_accuracy,
            "validation_auc": deepfake_auc,
        },
        "misinfo": {
            "available": misinfo_available,
            "validation_accuracy": misinfo_accuracy,
            "validation_f1_fake": misinfo_f1_fake,
            "samples_total": misinfo_samples,
        },
        "active_models": active_models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def serve_index():
    html_path = ROOT / "syntHeye.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="syntHeye.html not found.")
    return FileResponse(html_path)


@app.get("/syntHeye.html")
def serve_syntheye_html():
    html_path = ROOT / "syntHeye.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="syntHeye.html not found.")
    return FileResponse(html_path)


@app.get("/index.html")
def serve_landing_html():
    html_path = ROOT / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(html_path)


@app.get("/signup")
def serve_signup(request: Request):
    if get_session_user(request):
        return RedirectResponse(url="/", status_code=303)
    if not SIGNUP_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="signup.html not found.")
    return FileResponse(SIGNUP_PAGE_PATH)


@app.get("/signup.html")
def serve_signup_html(request: Request):
    return serve_signup(request)


@app.get("/login")
def serve_login(request: Request):
    if get_session_user(request):
        return RedirectResponse(url="/", status_code=303)
    if not LOGIN_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="login.html not found.")
    return FileResponse(LOGIN_PAGE_PATH)


@app.get("/login.html")
def serve_login_html(request: Request):
    return serve_login(request)


@app.post("/api/auth/signup")
def api_signup(payload: SignupRequest, request: Request):
    enforce_rate_limit(request, bucket="auth", limit=RATE_LIMIT_AUTH)
    user_row = create_user(payload.full_name, payload.email, payload.password)
    request.session["user_id"] = int(user_row.id)
    return {
        "status": "ok",
        "message": "Signup successful.",
        "user": serialize_user(user_row),
    }


@app.post("/api/auth/login")
def api_login(payload: LoginRequest, request: Request):
    enforce_rate_limit(request, bucket="auth", limit=RATE_LIMIT_AUTH)
    user_row = authenticate_user(payload.email, payload.password)
    mark_login(int(user_row.id))
    request.session["user_id"] = int(user_row.id)
    user_public = get_session_user(request)
    return {"status": "ok", "message": "Login successful.", "user": user_public}


@app.post("/api/auth/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"status": "ok", "message": "Logged out."}


@app.get("/api/me")
def api_me(request: Request):
    user = get_session_user(request)
    return {
        "authenticated": user is not None,
        "user": user,
        "require_auth": REQUIRE_AUTH,
    }


@app.get("/api/health")
def health():
    db_stats = get_database_stats()
    return {
        "status": "ok",
        "env": APP_ENV,
        "app_root": str(ROOT),
        "require_auth": REQUIRE_AUTH,
        "session_https_only": SESSION_HTTPS_ONLY,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_text_chars": MAX_TEXT_CHARS,
        "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        "database": db_stats,
        "security_warnings": security_warnings,
        "routes": {
            "signup": has_route("/signup"),
            "login": has_route("/login"),
            "me": has_route("/api/me"),
        },
        "pages": {
            "signup_html": SIGNUP_PAGE_PATH.exists(),
            "login_html": LOGIN_PAGE_PATH.exists(),
            "syntheye_html": (ROOT / "syntHeye.html").exists(),
        },
        "deepfake_model_present": DEEPFAKE_MODEL_PATH.exists(),
        "deepfake_model_loaded": deepfake_model is not None,
        "misinfo_artifacts_present": MISINFO_VECTORIZER_PATH.exists()
        and MISINFO_CLASSIFIER_PATH.exists(),
    }


@app.get("/api/db/stats")
def api_db_stats():
    return get_database_stats()


@app.get("/api/metrics")
def api_metrics():
    with metrics_lock:
        snapshot = dict(metrics)
    samples = int(snapshot.get("analyze_duration_samples", 0))
    total_ms = int(snapshot.get("analyze_duration_ms_total", 0))
    avg_ms = round(total_ms / samples, 2) if samples > 0 else 0.0
    snapshot["analyze_duration_avg_ms"] = avg_ms
    snapshot["rate_limit_window_seconds"] = RATE_LIMIT_WINDOW_SECONDS
    return snapshot


@app.get("/api/model/stats")
def model_stats():
    return get_model_stats()


@app.post("/api/warmup/deepfake")
def warmup_deepfake():
    return warmup_deepfake_model()


@app.get("/api/history")
def api_history(request: Request, limit: int = 20):
    enforce_rate_limit(request, bucket="history", limit=RATE_LIMIT_ANALYZE)
    if REQUIRE_AUTH:
        user = require_session_user(request)
        user_id = int(user["id"])
    else:
        session_user = get_session_user(request)
        user_id = int(session_user["id"]) if session_user else None
    items, total_count = get_recent_analysis_logs(user_id=user_id, limit=limit)
    return {
        "items": items,
        "count": len(items),
        "total_count": total_count,
        "limit": max(1, min(limit, 100)),
    }


@app.post("/api/analyze/text")
def analyze_text(payload: TextRequest, request: Request):
    enforce_rate_limit(request, bucket="analyze", limit=RATE_LIMIT_ANALYZE)
    started = time.perf_counter()
    record_metric("analyze_requests_total")
    if REQUIRE_AUTH:
        require_session_user(request)
    if len(payload.text) > MAX_TEXT_CHARS:
        detail = f"Text too long. Max allowed length is {MAX_TEXT_CHARS} characters."
        log_analysis_event(
            request,
            task="misinfo_text",
            error_detail=detail,
            request_meta={"input_type": "text", "text_length": len(payload.text)},
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(status_code=413, detail=detail)
    try:
        result = predict_text(payload.text)
    except HTTPException as exc:
        log_analysis_event(
            request,
            task="misinfo_text",
            error_detail=str(exc.detail),
            request_meta={"input_type": "text", "text_length": len(payload.text)},
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise
    record_analyze_duration(int((time.perf_counter() - started) * 1000))
    log_analysis_event(
        request,
        task=result.get("task", "misinfo_text"),
        result=result,
        request_meta={"input_type": "text", "text_length": len(payload.text)},
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return result


@app.post("/api/analyze/file")
async def analyze_file(request: Request, file: UploadFile = File(...)):
    enforce_rate_limit(request, bucket="analyze", limit=RATE_LIMIT_ANALYZE)
    started = time.perf_counter()
    record_metric("analyze_requests_total")
    if REQUIRE_AUTH:
        require_session_user(request)
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    kind = detect_file_kind(suffix, file.content_type)
    content_length = request.headers.get("content-length")
    with suppress(ValueError, TypeError):
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            record_metric("upload_rejected_total")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.",
            )
    binary = await read_upload_with_limit(file, MAX_UPLOAD_BYTES)
    if not binary:
        detail = "Uploaded file is empty."
        log_analysis_event(
            request,
            task=f"file_{kind or 'unknown'}",
            filename=filename,
            error_detail=detail,
            request_meta={"input_type": kind or "unknown", "content_type": file.content_type, "size_bytes": 0},
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(status_code=400, detail=detail)

    request_meta = {
        "input_type": kind or "unknown",
        "content_type": file.content_type,
        "size_bytes": len(binary),
    }

    try:
        if kind == "image":
            result = predict_image_bytes(binary, filename=filename)
        elif kind == "video":
            tmp_suffix = suffix if suffix in VIDEO_EXTS else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
                tmp.write(binary)
                tmp_path = Path(tmp.name)
            try:
                result = predict_video_file(tmp_path, filename=filename)
            finally:
                tmp_path.unlink(missing_ok=True)
        elif kind == "text":
            result = predict_text(binary.decode("utf-8", errors="ignore"))
        else:
            detail = (
                f"Unsupported file type: suffix='{suffix}' content_type='{file.content_type}'. "
                "Use image/video/txt."
            )
            raise HTTPException(status_code=400, detail=detail)
    except HTTPException as exc:
        if exc.status_code == 413:
            record_metric("upload_rejected_total")
        log_analysis_event(
            request,
            task=f"file_{kind or 'unknown'}",
            filename=filename,
            error_detail=str(exc.detail),
            request_meta=request_meta,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise

    record_analyze_duration(int((time.perf_counter() - started) * 1000))
    log_analysis_event(
        request,
        task=result.get("task", f"file_{kind or 'unknown'}"),
        filename=filename,
        result=result,
        request_meta=request_meta,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return result
