.PHONY: install install-all data demo eda train portfolio test lint ci api dashboard docker-up docker-down mlflow

install:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[dev,portfolio]"

data:
	bash scripts/download_data.sh

demo:
	python scripts/generate_demo_data.py --output-dir data/demo
	python -m fooddemand.train --data-dir data/demo --backtest-folds 2 --validation-days 10

eda:
	python -m fooddemand.eda --data-dir data/raw

train:
	python -m fooddemand.train --data-dir data/raw --backtest-folds 3 --validation-days 16

portfolio:
	bash scripts/run_portfolio.sh

test:
	pytest -q

lint:
	ruff check .

ci: lint test
	python scripts/generate_demo_data.py --output-dir /tmp/fooddemand-demo --days 90
	python -m fooddemand.train --data-dir /tmp/fooddemand-demo --backtest-folds 2 --validation-days 10 --model-out /tmp/fooddemand.joblib --report-out /tmp/metrics.json --model-card-out /tmp/MODEL_CARD.md

api:
	uvicorn app.main:app --reload

dashboard:
	streamlit run dashboard/app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

mlflow:
	mlflow server --host 127.0.0.1 --port 5000
