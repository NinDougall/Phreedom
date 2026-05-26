"""N-Deavour Services — Multi-Agent Orchestrator-Worker package.

Public re-exports for convenient top-level imports:

    from agents import OrchestratorAgent
    from agents import IngestionWorker, TaxEngine, AdvisorUI

Worker graph
────────────
    ┌──────────────────┐
    │  OrchestratorAgent│
    │  (orchestrator)  │
    └───────┬──────────┘
            │
    ┌───────▼──────────────────────────────────┐
    │  IngestionWorker  →  TaxEngine  →  AdvisorUI
    │  (parse/vault)       (tax math)   (chat/tone)
    └──────────────────────────────────────────┘

State is committed to the local profile manifest after every node transition.
"""

from agents.ingestion_worker import IngestionWorker, ParsedDocument
from agents.tax_engine import TaxEngine
from agents.advisor_ui import AdvisorUI
from agents.orchestrator import OrchestratorAgent

__all__ = [
    "OrchestratorAgent",
    "IngestionWorker",
    "TaxEngine",
    "AdvisorUI",
    "ParsedDocument",
]
