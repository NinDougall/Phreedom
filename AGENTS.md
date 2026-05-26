# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**N-Deavour Alignment** is a Streamlit-based personal financial agent (Phreedom) for
expense tracking, tax modelling, and bookkeeping.  It has been refactored from a
single-file monolith into a **decoupled Multi-Agent Orchestrator-Worker system**.

```
workspace/
├── app.py                    Streamlit entry point — UI only, no business logic
├── storage_bridge.py         Local-first persistence layer (vault, ledger, manifest)
└── agents/
    ├── __init__.py           Public re-exports
    ├── orchestrator.py       Central manager — delegates jobs, checkpoints state
    ├── ingestion_worker.py   File parsing, SHA-256 dedup, raw-JSON conversion
    ├── tax_engine.py         Tax logic, metric validation, context building
    └── advisor_ui.py         Phreedom chat persona, N-Deavour tone, OpenAI routing
```

All data is disk-persisted in `.ndeavour_profile/` (gitignored).  Session state is
an in-memory cache; every mutation is flushed to disk immediately.

### Running the app

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py --server.port 8501 --server.headless true
```

The app serves on port **8501**.  Add `--server.enableCORS false --server.enableXsrfProtection false`
if testing from a different origin.

### Agent architecture

#### OrchestratorAgent (`agents/orchestrator.py`)
- Single entry point for all business logic called by `app.py`.
- **`handle_upload(file_name, content, current_ledger)`** → runs IngestionWorker (Node A)
  then TaxEngine (Node B), saves ledger, checkpoints manifest.
- **`handle_chat(prompt, conversation, profile, ledger)`** → runs TaxEngine context
  (Node B) then AdvisorUI (Node C), saves chat history, checkpoints manifest.
- **`handle_reparse(file_name, content, current_ledger)`** → re-parses a vaulted file
  without re-vaulting it.
- **Checkpoint protocol**: after every node, a timestamped entry is appended to
  `agent_run_log` inside `profile_manifest.json` — the manifest is always a complete
  audit trail.

#### IngestionWorker (`agents/ingestion_worker.py`)
- Stateless: given the same bytes, always produces the same output.
- Parses CSV (column inference, debit/credit splits) and PDF (regex date/amount scan).
- Returns `IngestionResult` with `was_new`, `message`, `ParsedDocument`, and registry entry.
- Never writes to ledger — that responsibility belongs to the Orchestrator.

#### TaxEngine (`agents/tax_engine.py`)
- Computes `FinancialSummary` (income, expenses, profit, tax reserve, top expenses).
- `validate_metrics()` performs arithmetic self-audit; raises `TaxValidationError` on
  any inconsistency exceeding floating-point epsilon.
- `build_context()` assembles the financial context string injected into chat prompts.
- `record_snapshot()` persists the computed summary to the manifest.

#### AdvisorUI (`agents/advisor_ui.py`)
- Maintains the **Phreedom** persona and N-Deavour minimalist tone system.
- Routes prompts to `gpt-4o-mini` (or `OPENAI_MODEL` env var) when an API key is
  present; otherwise falls back to the deterministic keyword responder.
- `_enforce_tone()` strips filler phrases ("certainly", "great question", etc.).
- `persist_conversation()` saves chat history to disk via the bridge.

### Dependencies

```
streamlit
pandas
pypdf
altair
openai
```

Install with `pip install -r requirements.txt`.  All packages have no pinned versions;
the latest stable release of each is used.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | No | Enables LLM-backed responses (gpt-4o-mini by default) |
| `OPENAI_MODEL` | No | Override the OpenAI model name |

When no API key is present, the app uses the deterministic fallback in `AdvisorUI`.

### Notes

- **No automated tests** — no pytest, flake8, ruff, or mypy configuration.
- The devcontainer specifies Python 3.11; the app also runs on Python 3.12.
- `.ndeavour_profile/` is gitignored — all persistent data lives outside the repo.
- The `agent_run_log` inside `profile_manifest.json` records every agent node
  completion for auditing.
