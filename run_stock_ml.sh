#!/usr/bin/env bash
# Launches the Stock_ML Stock_ML web service (FastAPI + static SPA) on http://localhost:${PORT:-8000}
# Optional: STOCKML_DB=/path/to/copy.duckdb to use a DB copy; key via ANTHROPIC_API_KEY or .env
cd "$(dirname "$0")/src"
poetry run uvicorn stock_ml.backend.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
