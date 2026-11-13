# Deployment and operations

## Local API

```bash
uvicorn app.main:app --reload
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

## Docker Compose

After training creates `models/fooddemand.joblib` and `reports/latest_metrics.json`:

```bash
docker compose up --build
```

- API: http://127.0.0.1:8000/docs
- Dashboard: http://127.0.0.1:8501

The API image runs as an unprivileged user. The model and report are mounted read-only in Compose.
Production deployment would additionally require HTTPS, authentication if exposed publicly,
structured logs, metrics, request limits, artifact registry and drift monitoring.
