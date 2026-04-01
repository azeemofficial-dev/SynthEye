# SynthEye Dual-Model CLI

This project provides two CPU-friendly detection pipelines:

#SynthEye Team Distribution

This repository package contains a complete copy of the SynthEye project and a role-based split for team submissions on GitHub.

## Contents

1. `full_project_copy/` - exact snapshot of the original project.
2. `team_split/` - contributor-wise distribution of files.

## Directory Structure

```text
AI 2/
|-- full_project_copy/
|-- team_split/
|   |-- frontend/
|   |   |-- Bhumika/
|   |   |-- Azeem/
|   |-- backend/
|       |-- Surya/
|       |-- Akshita/
|       |-- Jiya/
|-- README.md
```

## Team and Responsibilities

| Member | Area | Responsibility |
|---|---|---|
| Bhumika | Frontend | Equal frontend split (UI pages and styling) |
| Azeem | Frontend | Equal frontend split + shared remaining files |
| Surya | Backend | Deepfake pipeline + deepfake models + shared remaining files |
| Akshita | Backend | Misinformation/news pipeline + misinfo models |
| Jiya | Backend | Remaining training and testing scripts |

## File Ownership

### Frontend - Bhumika

- `index.html`
- `login.html`
- `styles.css`
- `login/index.html`

### Frontend - Azeem

- `syntHeye.html`
- `signup.html`
- `script.js`
- `signup/index.html`
- all remaining unassigned project files (shared with Surya)

### Backend - Surya

- `train_deepfake.py`
- `predict_deepfake.py`
- `models/deepfake/deepfake_detector.keras`
- `models/deepfake/metadata.json`
- `models/deepfake/training_curves.png`
- all remaining unassigned project files (shared with Azeem)

### Backend - Akshita

- `train_misinfo.py`
- `predict_misinfo.py`
- `models/misinfo/classifier.joblib`
- `models/misinfo/vectorizer.joblib`
- `models/misinfo/metrics.json`

Note: a dedicated `news.csv` dataset file was not present in the project snapshot at split time.

### Backend - Jiya

- `train.py`
- `predict.py`
- `prepare_data.py`
- `check_backend.ps1`

## GitHub Upload Guidance

1. Upload each member folder from `team_split/` as that member's contribution.
2. Keep the file ownership mapping above in each member repository README or description.
3. If needed, keep `full_project_copy/` as a reference archive in a separate repository.

## Security Notes

- `.env.production` contains deployment secrets and should not be committed.
- Use `.env.example` and `.env.production.example` for public repositories.
