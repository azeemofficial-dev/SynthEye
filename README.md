# SynthEye
Deepfake Videos and Pictures and Misinformation Detection System

Team Members:
* Azeem 
* Bhumika 
* Jiya 
* Akshita 
* Surya 

## Team Structure

- Frontend:
  - Bhumika
  - Azeem
- Backend:
  - Surya
  - Akshita
  - Jiya

## Folder Layout

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

## Work Distribution

### Bhumika (Frontend - equal split)
Assigned files:
- `index.html`
- `login.html`
- `styles.css`
- `login/index.html`

### Azeem (Frontend - equal split + shared remaining)
Frontend assigned files:
- `syntHeye.html`
- `signup.html`
- `script.js`
- `signup/index.html`

Plus all **remaining** project files (shared with Surya).

### Surya (Backend - Deepfake + Models + shared remaining)
Deepfake and model files:
- `train_deepfake.py`
- `predict_deepfake.py`
- `models/deepfake/deepfake_detector.keras`
- `models/deepfake/metadata.json`
- `models/deepfake/training_curves.png`

Plus all **remaining** project files (shared with Azeem).

### Akshita (Backend - Misinfo and News)
Assigned files:
- `train_misinfo.py`
- `predict_misinfo.py`
- `models/misinfo/classifier.joblib`
- `models/misinfo/vectorizer.joblib`
- `models/misinfo/metrics.json`

Note: no dedicated `news.csv` file was found in the current project snapshot; misinfo/model files were distributed as requested.

### Jiya (Backend - Remaining training/testing)
Assigned files:
- `train.py`
- `predict.py`
- `prepare_data.py`
- `check_backend.ps1`

## Shared Remaining Files Rule Applied

As requested, all project files not explicitly assigned above were copied to:
- `team_split/frontend/Azeem/`
- `team_split/backend/Surya/`

## Security Note for GitHub Uploads

- `.env.production` may contain secrets and was intentionally excluded from role-based split folders.
- Use `.env.production.example` for public repositories.
