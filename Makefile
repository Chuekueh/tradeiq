.PHONY: install test lint type-check run run-frontend run-docker ingest evaluate clean

install:
	pip install -e ".[dev,frontend]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

type-check:
	mypy src/

run:
	uvicorn src.api.app:create_app --factory --reload --port 8000

run-frontend:
	streamlit run src/frontend/app.py

run-docker:
	docker compose up --build

fetch-filings:
	python scripts/fetch_sec_filings.py

generate-data:
	python scripts/generate_synthetic_data.py

ingest:
	python scripts/ingest_data.py --source data/raw/

evaluate:
	python scripts/run_evaluation.py --dataset data/evaluation/golden_qa.json

clean:
	rm -rf data/chroma_db/ data/processed/ __pycache__ .pytest_cache .mypy_cache
