# DriftGuardAI

DriftGuardAI is a modular Python starter project for data drift monitoring. The scaffold follows clean architecture principles so domain logic stays separate from framework code, data access, and utility concerns.

## Project Structure

```text
.
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── routes.py
│   └── schemas.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── entities.py
│   ├── exceptions.py
│   ├── interfaces.py
│   └── use_cases.py
├── data/
│   ├── __init__.py
│   ├── ingestion.py
│   └── repositories.py
├── datasets/
│   └── .gitkeep
├── drift/
│   ├── __init__.py
│   └── detectors.py
├── utils/
│   ├── __init__.py
│   ├── dataset_validation.py
│   └── logging.py
├── config.yaml
├── main.py
└── requirements.txt
```

## Architecture

- `core/`: domain models, ports, configuration, and use cases.
- `data/`: infrastructure adapters for ingestion and dataset access.
- `drift/`: drift detection implementations.
- `api/`: HTTP delivery layer using FastAPI.
- `utils/`: shared helpers for cross-cutting concerns like schema validation and logging.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
streamlit run dashboard.py
```

The API will be available at `http://127.0.0.1:8000`, with the health endpoint at `http://127.0.0.1:8000/api/v1/health`.
The Streamlit dashboard will be available at `http://127.0.0.1:8501`.

## Notes

- Update `config.yaml` with your preferred thresholds, dataset paths, and runtime settings.
- Add your reference and production datasets under `datasets/` or point the config to your storage layer.
- Extend `data/repositories.py` with database, object storage, or streaming adapters as the project grows.
