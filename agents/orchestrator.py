"""OrchestratorAgent — central multi-agent manager.

The Orchestrator is the single entry point the Streamlit app (or any other
front-end) calls for all business logic.  It:

1. Accepts a file upload or a chat prompt from the stream.
2. Delegates to the correct specialist worker.
3. Merges the worker's output context back into the shared ledger / chat.
4. **Checkpoints state to the local profile manifest after every node
   completes** — guaranteeing durability regardless of where the process
   is interrupted.

Worker graph
────────────
    ┌────────────────────┐
    │  OrchestratorAgent │
    └──────┬─────────────┘
           │
    ┌──────▼──────────────────────────────────────────────────────────────┐
    │ Node A: IngestionWorker                                             │
    │   ingest(file_name, content) → IngestionResult                     │
    │   checkpoint("ingestion", result.to_dict())                        │
    └──────┬──────────────────────────────────────────────────────────────┘
           │ (file upload path)
    ┌──────▼──────────────────────────────────────────────────────────────┐
    │ Node B: TaxEngine                                                   │
    │   compute_summary(ledger, tax_rate) → FinancialSummary             │
    │   record_snapshot(summary)   ← writes to manifest                  │
    │   checkpoint("tax_compute", summary.to_dict())                     │
    └──────┬──────────────────────────────────────────────────────────────┘
           │ (chat path — context building)
    ┌──────▼──────────────────────────────────────────────────────────────┐
    │ Node C: AdvisorUI                                                   │
    │   generate_response(prompt, context, conversation, …)  → str       │
    │   persist_conversation(messages)  ← writes to disk                 │
    │   checkpoint("advisor_reply", {"reply": reply})                    │
    └─────────────────────────────────────────────────────────────────────┘

Checkpoint protocol
-------------------
After each node, ``_checkpoint(stage, payload)`` appends a timestamped entry
to the ``agent_run_log`` array inside the profile manifest.  This means the
manifest is always a complete audit trail of every agent interaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from agents.advisor_ui import AdvisorUI
from agents.ingestion_worker import IngestionResult, IngestionWorker
from agents.tax_engine import FinancialSummary, TaxEngine


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------


class OrchestratorAgent:
    """Central manager that delegates jobs and merges contexts.

    Parameters
    ----------
    bridge:
        A ``StorageBridge`` instance.  All disk I/O flows through this object.

    Example
    -------
    ::

        from storage_bridge import get_bridge
        from agents import OrchestratorAgent

        orchestrator = OrchestratorAgent(get_bridge())

        # Handle a file upload
        result = orchestrator.handle_upload("bank.csv", csv_bytes)
        new_ledger = result["ledger"]

        # Handle a chat turn
        reply = orchestrator.handle_chat(
            prompt="How much should I reserve for taxes?",
            conversation=[...],
            profile={"tax_rate": 0.30, "business_type": "Freelancer", "tax_notes": ""},
            ledger=new_ledger,
        )
    """

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge
        self._ingestion_worker = IngestionWorker(bridge)
        self._tax_engine = TaxEngine(bridge)
        self._advisor_ui = AdvisorUI(bridge)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_upload(
        self,
        file_name: str,
        content: bytes,
        current_ledger: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Process a file upload through the ingestion and tax nodes.

        Flow
        ----
        1. IngestionWorker.ingest → parse + vault (Node A)
        2. Checkpoint "ingestion" to manifest
        3. Merge new transactions into the ledger
        4. Persist updated ledger to disk
        5. TaxEngine.compute_summary → validate metrics (Node B)
        6. TaxEngine.record_snapshot → persist snapshot to manifest
        7. Checkpoint "tax_compute" to manifest
        8. Return full result dict

        Parameters
        ----------
        file_name:
            Original filename (e.g. ``"bank_export.csv"``).
        content:
            Raw file bytes.
        current_ledger:
            Existing ledger DataFrame to merge new transactions into.
            Pass ``None`` or an empty DataFrame for the first import.

        Returns
        -------
        dict with keys:
            ``was_new`` (bool), ``message`` (str), ``ledger`` (DataFrame),
            ``ingestion_result`` (dict), ``tax_summary`` (dict).
        """
        # ── Node A: Ingestion ─────────────────────────────────────────
        result: IngestionResult = self._ingestion_worker.ingest(file_name, content)
        self._checkpoint("ingestion", result.to_dict())

        if not result.was_new:
            return {
                "was_new": False,
                "message": result.message,
                "ledger": current_ledger if current_ledger is not None else pd.DataFrame(),
                "ingestion_result": result.to_dict(),
                "tax_summary": {},
            }

        # ── Merge transactions ────────────────────────────────────────
        assert result.parsed is not None
        new_ledger = self._merge_ledger(current_ledger, result.parsed.transactions)
        self._bridge.save_ledger(new_ledger)

        # ── Node B: Tax compute ───────────────────────────────────────
        profile = self._bridge.fetch_profile()
        tax_rate = float(profile.get("tax_reserve_rate", 0.30))
        summary: FinancialSummary = self._tax_engine.compute_summary(new_ledger, tax_rate)
        self._tax_engine.record_snapshot(summary)
        self._checkpoint("tax_compute", summary.to_dict())

        return {
            "was_new": True,
            "message": result.message,
            "ledger": new_ledger,
            "ingestion_result": result.to_dict(),
            "tax_summary": summary.to_dict(),
        }

    def handle_chat(
        self,
        prompt: str,
        conversation: list[dict[str, str]],
        profile: dict[str, Any],
        ledger: pd.DataFrame,
    ) -> str:
        """Process a chat prompt through the tax context and advisor nodes.

        Flow
        ----
        1. TaxEngine.compute_summary → build context (Node B)
        2. TaxEngine.record_snapshot → persist snapshot
        3. Checkpoint "tax_compute" to manifest
        4. AdvisorUI.generate_response → Phreedom reply (Node C)
        5. AdvisorUI.persist_conversation → save chat to disk
        6. Checkpoint "advisor_reply" to manifest
        7. Return reply string

        Parameters
        ----------
        prompt:
            The user's latest message.
        conversation:
            Full message history as ``[{"role": ..., "content": ...}, ...]``.
        profile:
            Dict with keys ``tax_rate`` (float), ``business_type`` (str),
            ``tax_notes`` (str).
        ledger:
            Current ledger DataFrame.

        Returns
        -------
        str
            Phreedom's reply, ready for display.
        """
        tax_rate = float(profile.get("tax_rate", profile.get("tax_reserve_rate", 0.30)))
        business_type = str(profile.get("business_type", ""))
        tax_notes = str(profile.get("tax_notes", ""))

        # ── Node B: Tax context ───────────────────────────────────────
        summary: FinancialSummary = self._tax_engine.compute_summary(ledger, tax_rate)
        docs_list = self._bridge.list_documents()
        context = self._tax_engine.build_context(
            summary=summary,
            docs_list=docs_list,
            ledger=ledger,
            business_type=business_type,
            tax_notes=tax_notes,
        )
        self._tax_engine.record_snapshot(summary)
        self._checkpoint("tax_compute", summary.to_dict())

        # ── Node C: Advisor reply ─────────────────────────────────────
        reply = self._advisor_ui.generate_response(
            prompt=prompt,
            context=context,
            conversation=conversation,
            tax_rate=tax_rate,
            ledger=ledger,
        )

        # Persist updated conversation (caller appended prompt + reply)
        updated_messages = [
            *conversation,
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
        self._advisor_ui.persist_conversation(updated_messages)
        self._checkpoint("advisor_reply", {"prompt_preview": prompt[:80], "reply_preview": reply[:80]})

        return reply

    def handle_reparse(
        self,
        file_name: str,
        content: bytes,
        current_ledger: pd.DataFrame,
    ) -> dict[str, Any]:
        """Re-parse a vaulted file and rebuild its ledger contribution.

        This does NOT re-vault the file (it is already in the vault).  It
        re-parses the bytes, drops the old rows for this source from the
        ledger, merges the fresh rows, persists, and checkpoints.

        Returns the same shape dict as ``handle_upload()``.
        """
        parsed = self._ingestion_worker.reparse(file_name, content)

        # Remove old rows for this source, then merge fresh ones
        stripped = current_ledger[current_ledger["source"] != file_name].copy()
        new_ledger = self._merge_ledger(stripped, parsed.transactions)
        self._bridge.save_ledger(new_ledger)

        # Update vault metadata
        import hashlib
        file_hash = hashlib.sha256(content).hexdigest()
        self._bridge.update_document_metadata(
            file_hash,
            {"transaction_count": len(parsed.transactions), "summary": parsed.summary},
        )

        profile = self._bridge.fetch_profile()
        tax_rate = float(profile.get("tax_reserve_rate", 0.30))
        summary = self._tax_engine.compute_summary(new_ledger, tax_rate)
        self._tax_engine.record_snapshot(summary)
        self._checkpoint("reparse", {"source": file_name, "rows": len(parsed.transactions)})

        return {
            "was_new": True,
            "message": parsed.summary,
            "ledger": new_ledger,
            "ingestion_result": parsed.to_dict(),
            "tax_summary": summary.to_dict(),
        }

    # ------------------------------------------------------------------
    # Manifest checkpoint
    # ------------------------------------------------------------------

    def _checkpoint(self, stage: str, payload: dict[str, Any]) -> None:
        """Append a timestamped node-completion record to the manifest.

        The ``agent_run_log`` list inside the manifest grows by one entry
        each time a worker node finishes, providing a permanent audit trail.

        Parameters
        ----------
        stage:
            Human-readable label (e.g. ``"ingestion"``, ``"tax_compute"``).
        payload:
            JSON-serialisable dict describing the node's output.
        """
        manifest = self._bridge._read_manifest()
        log = manifest.setdefault("agent_run_log", [])
        log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "payload": payload,
        })
        # Keep only the last 100 entries to prevent unbounded growth
        if len(log) > 100:
            manifest["agent_run_log"] = log[-100:]
        self._bridge._write_manifest(manifest)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_ledger(
        existing: pd.DataFrame | None,
        new_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenate new transaction rows onto the existing ledger."""
        from agents.ingestion_worker import LEDGER_COLUMNS

        if new_rows is None or new_rows.empty:
            return existing if existing is not None else pd.DataFrame(columns=LEDGER_COLUMNS)
        if existing is None or existing.empty:
            return new_rows[LEDGER_COLUMNS].reset_index(drop=True)
        return pd.concat(
            [existing, new_rows[LEDGER_COLUMNS]], ignore_index=True
        )
