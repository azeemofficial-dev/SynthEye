# SynthEye Dual-Model CLI

This project provides two CPU-friendly detection pipelines:

1. Deepfake vision detection
- Image classification (`fake` vs `real`)
- Video detection by sampling frames and aggregating per-frame scores

2. Misinformation/news text detection
- Binary text classification (`fake` vs `real`)
- TF-IDF + Logistic Regression baseline

It also includes a FastAPI backend, session-based authentication (signup/login), and a wired frontend (`syntHeye.html`) for file/text analysis.

## Files

```text
SynthEye/
|-- train_deepfake.py
|-- predict_deepfake.py
|-- train_misinfo.py
|-- predict_misinfo.py
|-- prepare_data.py
|-- api_server.py
|-- signup.html
|-- login.html
|-- train.py               # compatibility wrapper
|-- predict.py             # compatibility wrapper
|-- requirements.txt
|-- syntHeye.html
|-- .env.example
|-- Dockerfile
```

## Web API + UI

Start API server:

```bash
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Windows (recommended in this project):

```powershell
.\.venv311\Scripts\python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Safer Windows launcher (kills stale process on same port and starts from correct folder):

```powershell
.\start_syntheye.ps1
```

Verify backend instance and auth routes:

```powershell
.\check_backend.ps1
```

Open:
- `http://127.0.0.1:8000/` for the UI
- `http://127.0.0.1:8000/index.html` for the connected landing page
- `http://127.0.0.1:8000/docs` for Swagger
- `http://127.0.0.1:8000/signup` and `http://127.0.0.1:8000/login` for auth pages

Main endpoints:
- `GET /api/health`
- `GET /api/db/stats` (database status + row counts)
- `GET /api/metrics` (request/error/rate-limit counters and latency aggregates)
- `GET /api/model/stats` (model validation metrics used by UI meters)
- `POST /api/warmup/deepfake` (pre-load deepfake model to avoid first-request delay)
- `POST /api/analyze/file` (image/video/txt upload)
- `POST /api/analyze/text` (JSON body with `text`)
- `GET /api/history?limit=20` (recent analysis history for signed-in user)
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/me`

Auth pages:
- `GET /signup`
- `GET /login`
- Static-server fallback pages: `/signup.html` and `/login.html` (for `127.0.0.1:5500` style local servers)

Connected page map:
- `index.html` (landing + live API integration)
- `syntHeye.html` (main dashboard)
- `signup.html` and `login.html` (auth)
- All pages auto-detect `:5500` vs `:8000` and use correct navigation/API base.

## Installation

Recommended Python:
- Python 3.10 or 3.11 for full deepfake + misinformation pipeline support

```bash
pip install -r requirements.txt
```

If you are on Python 3.12+ (for example Python 3.14), TensorFlow will not install, so deepfake scripts will not run until you switch to Python 3.10/3.11.

## Authentication and Session Config

Environment variables:
- `SYNTHEYE_ENV` (`development` or `production`)
- `SYNTHEYE_REQUIRE_AUTH` (`1` or `0`, default `1`)
- `SYNTHEYE_SESSION_SECRET` (set a long random value in production)
- `SYNTHEYE_SESSION_TTL_SECONDS` (default `86400`)
- `SYNTHEYE_SESSION_HTTPS_ONLY` (`1` in HTTPS production, default `0` for local)
- `SYNTHEYE_ALLOW_ORIGINS` (comma-separated CORS origins, includes `:8000` and `:5500` by default)
- `SYNTHEYE_MAX_UPLOAD_MB` (max upload size, default `50`)
- `SYNTHEYE_MAX_TEXT_CHARS` (max text payload length, default `20000`)
- `SYNTHEYE_RATE_LIMIT_*` controls (`GLOBAL`, `AUTH`, `ANALYZE`)
- `SYNTHEYE_LOG_LEVEL` / `SYNTHEYE_LOG_FILE`

Quick flow:
1. Open `/signup` and create an account.
2. You are logged in automatically.
3. Use `/` for analysis dashboard.
4. Use nav button to log out.

## Database (MySQL Workbench Ready)

The backend persists:
- user accounts (`users`)
- analysis runs (`analysis_logs`)

Default (no config):
- SQLite local file: `data/syntheye.db`

MySQL option (for MySQL Workbench):
1. Create database in Workbench:

```sql
CREATE DATABASE syntheye CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Or run [`database_setup.sql`](database_setup.sql) in MySQL Workbench.

2. Set one of these config styles before starting API:

Option A:
- `SYNTHEYE_DATABASE_URL=mysql+pymysql://root:<password>@127.0.0.1:3306/syntheye`

Option B:
- `SYNTHEYE_DB_HOST=127.0.0.1`
- `SYNTHEYE_DB_PORT=3306`
- `SYNTHEYE_DB_USER=root`
- `SYNTHEYE_DB_PASSWORD=<password>`
- `SYNTHEYE_DB_NAME=syntheye`

Tables are auto-created on startup.

## Production Deploy (HTTPS + MySQL + Reverse Proxy)

This repository includes production files:
- `docker-compose.prod.yml`
- `Caddyfile`
- `.env.production.example`
- `ops/generate_prod_env.ps1`
- `ops/deploy_preflight.ps1`
- `ops/backup_mysql.sh`
- `ops/backup_mysql.ps1`
- `ops/monitoring_plan.md`

### 1) Prepare production env

Windows quick-start (auto-generate strong secrets):

```powershell
powershell -ExecutionPolicy Bypass -File .\ops\generate_prod_env.ps1 -Domain YOUR_DOMAIN -LetsEncryptEmail you@example.com
```

Manual copy/edit:

```bash
cp .env.production.example .env.production
```

Set real values:
- strong `SYNTHEYE_SESSION_SECRET` (64+ random chars)
- your public domain in `SYNTH_DOMAIN`
- strict CORS in `SYNTHEYE_ALLOW_ORIGINS` (HTTPS origin only)
- MySQL passwords

### 2) Run preflight checks

```powershell
powershell -ExecutionPolicy Bypass -File .\ops\deploy_preflight.ps1 -EnvFile .env.production
```

This validates:
- required env values
- model artifacts
- CORS/session production safety
- Docker Compose config (when Docker is installed)

### 3) Start stack

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
```

Stack services:
- `app` (FastAPI, no reload)
- `mysql` (persistent DB)
- `caddy` (HTTPS reverse proxy)

### 4) Validate after deploy

```bash
curl -s https://YOUR_DOMAIN/api/health
curl -s https://YOUR_DOMAIN/api/db/stats
curl -s https://YOUR_DOMAIN/api/metrics
```

### 5) Backups and monitoring

- Linux backup: `bash ops/backup_mysql.sh`
- Windows backup: `powershell -ExecutionPolicy Bypass -File .\ops\backup_mysql.ps1`
- Monitoring checklist: `ops/monitoring_plan.md`

Make sure trained artifacts exist before deploy:
- `models/deepfake/deepfake_detector.keras`
- `models/misinfo/vectorizer.joblib`
- `models/misinfo/classifier.joblib`

## Standard Dataset Contract

### Deepfake expected layout

```text
data/deepfake/images/
|-- fake/
|-- real/
```

### Misinformation expected layout

```text
data/misinfo/news.csv
```

Required CSV columns:
- `text`
- `label` (must resolve to `fake` or `real`)

## Prepare Kaggle Data

Use `prepare_data.py` to normalize downloaded Kaggle datasets.

### Deepfake prep

```bash
python prepare_data.py --task deepfake --source kaggle_extract/deepfake --target data/deepfake/images
```

Notes:
- The script auto-detects class by folder/file path keywords (for example `fake`, `real`, `deepfake`, `authentic`).
- Output is copied into `target/fake` and `target/real`.

### Misinformation prep

```bash
python prepare_data.py --task misinfo --source kaggle_extract/news --target data/misinfo
```

Optional controls:

```bash
python prepare_data.py --task misinfo --source kaggle_extract/news.csv --target data/misinfo/news.csv --text_col text --label_col label --numeric_labels 0_fake_1_real
```

`--numeric_labels` choices:
- `0_fake_1_real` (default)
- `0_real_1_fake`

## Training

### Deepfake model

```bash
python train_deepfake.py --data_dir data/deepfake/images --epochs 15 --batch_size 16 --output_dir models/deepfake
```

Artifacts:
- `models/deepfake/deepfake_detector.keras`
- `models/deepfake/training_curves.png`
- `models/deepfake/metadata.json`

### Misinformation model

```bash
python train_misinfo.py --csv data/misinfo/news.csv --text_col text --label_col label --output_dir models/misinfo
```

Artifacts:
- `models/misinfo/vectorizer.joblib`
- `models/misinfo/classifier.joblib`
- `models/misinfo/metrics.json`

## Inference

### Deepfake image

```bash
python predict_deepfake.py --image sample.jpg --model_path models/deepfake/deepfake_detector.keras
```

### Deepfake video (frame aggregation)

```bash
python predict_deepfake.py --video sample.mp4 --model_path models/deepfake/deepfake_detector.keras --sample_fps 1 --max_frames 120
```

### Misinformation text

```bash
python predict_misinfo.py --text "Breaking: shocking claim with no source" --model_dir models/misinfo
```

or

```bash
python predict_misinfo.py --file input.txt --model_dir models/misinfo
```

## Compatibility Wrappers

You can still use old entrypoints:

```bash
python train.py --task deepfake --data_dir data/deepfake/images
python train.py --task misinfo --csv data/misinfo/news.csv

python predict.py --task deepfake --image sample.jpg
python predict.py --task misinfo --text "news text"
```

If no `--task` is provided, wrappers infer task from your flags.

## Example Outputs

Deepfake image:

```text
DEEPFAKE IMAGE RESULT
Prediction  : FAKE
Confidence  : 93.12%
Real score  : 0.0688
```

Misinformation text:

```text
MISINFORMATION TEXT RESULT
Prediction       : REAL
Confidence       : 87.40%
Fake probability : 0.1260
Real probability : 0.8740
```

## Troubleshooting

1. `Model not found`
- Train the corresponding model first.
- Check `--model_path` or `--model_dir`.

2. `Found 0 files` or class folder issues in deepfake training
- Ensure exact folder names: `fake` and `real`.
- Ensure image extensions are valid (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).

3. Low misinformation accuracy
- Increase dataset size and quality.
- Verify correct numeric label mapping in `prepare_data.py`.
- Check for heavy class imbalance.

4. Kaggle CSV column mismatch
- Provide explicit `--text_col` and `--label_col`.

5. Video issues
- Install `opencv-python` from `requirements.txt`.
- Try lower `--sample_fps` or lower `--max_frames` for memory limits.

6. `Unexpected end of JSON input` in browser UI
- This usually happens when opening `syntHeye.html` from a static server without the API running.
- Start backend first: `uvicorn api_server:app --reload --host 127.0.0.1 --port 8000`
- Open `http://127.0.0.1:8000/` (recommended). If using `:5500`, keep backend running on `:8000`.

7. `Authentication required` on analyze endpoints
- Create an account at `/signup` or sign in at `/login`.
- If you want open access in local testing, set `SYNTHEYE_REQUIRE_AUTH=0` before starting the server.

8. MySQL connection issues
- Ensure MySQL service is running and reachable on the configured host/port.
- Confirm DB credentials and database name are correct.
- Install dependencies from `requirements.txt` (`sqlalchemy` and `pymysql` are required).

9. Production startup fails with security config error
- In production, app enforces:
  - `SYNTHEYE_SESSION_HTTPS_ONLY=1`
  - strong `SYNTHEYE_SESSION_SECRET`
  - MySQL database (not SQLite)
  - HTTPS-only CORS origins
- Fix env values in `.env.production`.

10. `413` errors on upload/text
- Increase `SYNTHEYE_MAX_UPLOAD_MB` for larger files.
- Increase `SYNTHEYE_MAX_TEXT_CHARS` for longer text payloads.

11. `429 Rate limit exceeded`
- Adjust `SYNTHEYE_RATE_LIMIT_GLOBAL_PER_WINDOW`, `SYNTHEYE_RATE_LIMIT_AUTH_PER_WINDOW`, or `SYNTHEYE_RATE_LIMIT_ANALYZE_PER_WINDOW`.
