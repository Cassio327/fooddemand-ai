.PHONY: install test lint train api docker-build

install:
	python -m pip install -e ".[dev,data]"

test:
	pytest -q

lint:
	ruff check .

train:
	python -m fooddemand.train --data data/raw/train.csv --max-rows 500000

api:
	uvicorn app.main:app --reload

docker-build:
	docker build -t fooddemand-ai .
