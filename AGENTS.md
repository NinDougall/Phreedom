# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a single-file Streamlit Python app (`app.py`) called **N-Deavour Alignment** — a personal financial agent interface for expense tracking, tax modeling, and bookkeeping. All data is mock-backed and lives in Streamlit session state (no database).

### Running the app

```bash
python3 -m streamlit run app.py --server.port 8501 --server.headless true
```

The app serves on port **8501**. Add `--server.enableCORS false --server.enableXsrfProtection false` if testing from a different origin.

### Dependencies

Install with `pip install -r requirements.txt`. The only runtime dependencies are: `streamlit`, `pandas`, `pypdf`, `altair`, and `openai` (the `openai` package is listed but not actively used — the app uses mock responses).

### Notes

- There are **no automated tests** or linting configuration in this repository. No test runner, no `pytest`, no `flake8`/`ruff`/`mypy` config.
- The devcontainer specifies Python 3.11, but the app runs fine on Python 3.12.
- No external services, databases, or API keys are required. The `OPENAI_API_KEY` env var is optional and not used by the current code.
- The app is a single `app.py` file (~823 lines). All state resets on page reload.
