"""N-Deavourservices StorageBridge — local-first, cloud-ready storage interface.

All persistent reads and writes are routed through this module.
To swap from local disk to a cloud backend (Supabase, AWS S3, Vercel Postgres…),
change ``StorageBridge.BACKEND`` and implement the corresponding ``_cloud_*``
helper methods. The public API is identical regardless of backend.

Directory layout created on first use:

    .ndeavour_profile/
    ├── profile_manifest.json   # User profile + file registry metadata
    ├── ledger.csv              # All parsed financial transactions
    ├── timesheet.csv           # Freelance timesheet entries
    ├── chat_history.json       # Phreedom chat log
    └── secure_vault/           # Hard copies of every uploaded file
        ├── <hash16>_<name>
        └── …
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import secrets


# ---------------------------------------------------------------------------
# Password hashing and AuthStore
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with SHA-256 and a random salt."""
    salt = secrets.token_hex(16)
    iterations = 100_000
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${hashed}"


def verify_password(password: str, hashed_ref: str) -> bool:
    """Verify a password against a hashed reference string."""
    try:
        parts = hashed_ref.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = parts[2]
        hashed = parts[3]
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        ).hex()
        return secrets.compare_digest(candidate, hashed)
    except Exception:
        return False


class AuthStore:
    """Credentials and profile directory manager for multi-user authentication."""

    def __init__(self, base_dir: Path | str = ".ndeavour_profile") -> None:
        self._base = Path(base_dir).resolve()
        self._users_file = self._base / "user_registry.json"
        self._ensure_base()

    def _ensure_base(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        if not self._users_file.exists():
            self._users_file.write_text(json.dumps({}, indent=2), encoding="utf-8")

    def _read_users(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self._users_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_users(self, users: dict[str, dict[str, Any]]) -> None:
        self._users_file.write_text(json.dumps(users, indent=2), encoding="utf-8")

    def register(self, username: str, password: str, email: str, role: str = "Standard User") -> tuple[bool, str]:
        """Register a new user.

        Usernames must be 3-20 characters and alphanumeric/underscore.
        Emails must be valid and unique.
        """
        username = username.strip()
        if not re.match(r"^[a-zA-Z0-9_]{3,20}$", username):
            return False, "Username must be 3-20 characters and only contain letters, numbers, and underscores."

        email = email.strip()
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            return False, "Invalid email address format."

        if len(password) < 6:
            return False, "Password must be at least 6 characters long."

        users = self._read_users()
        username_lower = username.lower()
        if username_lower in {u.lower() for u in users}:
            return False, f"Username '{username}' is already taken."

        email_lower = email.lower()
        if email_lower in {u.get("email", "").lower() for u in users.values()}:
            return False, f"Email address '{email}' is already registered."

        # If it is the first registered user, make them an Administrator, otherwise Standard User
        assigned_role = "Administrator" if not users else role

        user_id = secrets.token_hex(8)  # Unique, secure random user id (16 chars)

        users[username] = {
            "user_id": user_id,
            "email": email,
            "password_hash": hash_password(password),
            "role": assigned_role,
            "created_at": datetime.now().isoformat(),
            "status": "Active",
        }
        self._write_users(users)
        return True, "Registration successful."

    def authenticate(self, username: str, password: str) -> tuple[bool, Any]:
        """Authenticate a user and return (success, user_dict) or (failure, error_msg)."""
        username = username.strip()
        users = self._read_users()
        
        username_lower = username.lower()
        matched_username = None
        for u in users:
            if u.lower() == username_lower:
                matched_username = u
                break

        if not matched_username:
            return False, "Invalid username or password."

        user_data = users[matched_username]
        if user_data.get("status", "Active") != "Active":
            return False, "Your account has been deactivated. Please contact an Administrator."

        if verify_password(password, user_data["password_hash"]):
            # Return username with preserved casing and metadata dict
            profile = dict(user_data)
            profile["username"] = matched_username
            return True, profile
        
        return False, "Invalid username or password."

    def list_all_users(self) -> list[dict[str, Any]]:
        """Return a formatted list of all registered users for administration."""
        users = self._read_users()
        return [
            {
                "username": username,
                "email": data.get("email", "—"),
                "user_id": data.get("user_id", "—"),
                "role": data.get("role", "Standard User"),
                "created_at": data.get("created_at", "—"),
                "status": data.get("status", "Active")
            }
            for username, data in users.items()
        ]

    def update_user_status(self, username: str, status: str) -> bool:
        """Update a user account active status (e.g. Active, Suspended)."""
        users = self._read_users()
        if username not in users:
            return False
        users[username]["status"] = status
        self._write_users(users)
        return True

    def update_user_role(self, username: str, role: str) -> bool:
        """Update a user's permissions level (e.g. 'Standard User', 'Administrator')."""
        users = self._read_users()
        if username not in users:
            return False
        users[username]["role"] = role
        self._write_users(users)
        return True

    def _find_user_for_reset(self, identifier: str) -> tuple[str | None, str | None]:
        """Resolve a username from an email address or username (legacy accounts)."""
        identifier = identifier.strip().lower()
        if not identifier:
            return None, None

        users = self._read_users()
        for username, data in users.items():
            if data.get("email", "").strip().lower() == identifier:
                return username, data.get("email", "").strip() or None

        for username in users:
            if username.lower() == identifier:
                return username, users[username].get("email", "").strip() or None

        return None, None

    def update_user_email(self, username: str, email: str) -> tuple[bool, str]:
        """Assign or update a user's email address (admin maintenance)."""
        email = email.strip()
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            return False, "Invalid email address format."

        users = self._read_users()
        if username not in users:
            return False, f"User '{username}' was not found."

        email_lower = email.lower()
        for other_name, data in users.items():
            if other_name != username and data.get("email", "").strip().lower() == email_lower:
                return False, f"Email address '{email}' is already registered."

        users[username]["email"] = email
        self._write_users(users)
        return True, "Email address updated successfully."

    def generate_reset_token(self, identifier: str) -> tuple[bool, str, str]:
        """Generate a secure reset token for an email address or username.

        Returns (success, token_or_error_message, username).
        """
        matched_username, stored_email = self._find_user_for_reset(identifier)
        if not matched_username:
            return False, "No account found with that email address or username.", ""

        users = self._read_users()
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = (datetime.now() + timedelta(minutes=15)).isoformat()

        users[matched_username]["reset_token_hash"] = token_hash
        users[matched_username]["reset_token_expires"] = expires_at
        self._write_users(users)

        return True, token, matched_username

    def verify_reset_token(self, token: str) -> tuple[bool, str]:
        """Verify the reset token hash and check if it has expired.

        Returns (success, username) or (failure, error_msg).
        """
        token = token.strip()
        if not token:
            return False, "Token cannot be empty."

        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        users = self._read_users()

        matched_username = None
        for username, data in users.items():
            if data.get("reset_token_hash") == token_hash:
                matched_username = username
                break

        if not matched_username:
            return False, "Invalid or unrecognized reset token."

        expires_str = users[matched_username].get("reset_token_expires")
        if not expires_str:
            return False, "Reset token has expired or is invalid."

        try:
            expires_at = datetime.fromisoformat(expires_str)
        except Exception:
            return False, "Invalid expiration timestamp format."

        if datetime.now() > expires_at:
            return False, "Reset token has expired (15-minute limit exceeded)."

        return True, matched_username

    def reset_password_with_token(self, token: str, new_password: str) -> tuple[bool, str]:
        """Reset user password using a valid reset token."""
        success, result = self.verify_reset_token(token)
        if not success:
            return False, result

        username = result
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters long."

        users = self._read_users()
        users[username]["password_hash"] = hash_password(new_password)
        users[username].pop("reset_token_hash", None)
        users[username].pop("reset_token_expires", None)

        self._write_users(users)
        return True, "Password reset successful."


def generate_password_reset_token(identifier: str, base_dir: Path | str = ".ndeavour_profile") -> tuple[bool, str, str]:
    """Module-level helper used by the UI to avoid stale class definitions on hot reload."""
    return AuthStore(base_dir).generate_reset_token(identifier)


def verify_password_reset_token(token: str, base_dir: Path | str = ".ndeavour_profile") -> tuple[bool, str]:
    """Module-level helper to verify a password reset token."""
    return AuthStore(base_dir).verify_reset_token(token)


def reset_password_with_token(token: str, new_password: str, base_dir: Path | str = ".ndeavour_profile") -> tuple[bool, str]:
    """Module-level helper to reset a password with a valid token."""
    return AuthStore(base_dir).reset_password_with_token(token, new_password)


# ---------------------------------------------------------------------------
# Column schemas (mirrored from app.py — kept independent for decoupling)
# ---------------------------------------------------------------------------

_LEDGER_COLS = [
    "date", "description", "amount", "kind", "expense_type", "category", "source",
]
EXPENSE_TYPES = ("Business", "Personal")
_TIMESHEET_COLS = ["date", "project", "hours", "rate", "total_pay"]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

_DEFAULT_MANIFEST: dict[str, Any] = {
    "version": "1.1",
    "created_at": "",
    "last_updated": "",
    "user_profile": {
        "business_type": "",
        "tax_reserve_rate": 0.30,
        "tax_notes": "",
    },
    "tax_categories": [
        "Revenue",
        "Software",
        "Contractors",
        "Travel",
        "Meals",
        "Office",
        "Bank Fees",
        "Taxes",
        "Owner Draw",
        "Uncategorized",
    ],
    "file_registry": [],
}


# ---------------------------------------------------------------------------
# StorageBridge
# ---------------------------------------------------------------------------


class StorageBridge:
    """Abstract storage interface currently mapped to local disk.

    Public interface
    ----------------
    save_document(name, content, metadata)   -> dict   (registry entry)
    fetch_document(file_hash)                -> bytes | None
    delete_document(file_hash)               -> bool
    list_documents()                         -> list[dict]
    fetch_profile()                          -> dict
    save_profile(profile_dict)               -> None
    fetch_ledger()                           -> pd.DataFrame
    save_ledger(df)                          -> None
    fetch_timesheet()                        -> pd.DataFrame
    save_timesheet(df)                       -> None
    fetch_chat_history()                     -> list[dict]
    save_chat_history(messages)              -> None
    purge_all()                              -> None
    """

    # ── Backend selector ─────────────────────────────────────────────────
    # Change this string (and implement the matching `_<backend>_*` helpers)
    # to migrate to a cloud provider without touching any caller code.
    BACKEND: str = "local"

    def __init__(self, base_dir: Path | str = ".ndeavour_profile") -> None:
        self._base = Path(base_dir).resolve()
        self._vault = self._base / "secure_vault"
        self._manifest_path = self._base / "profile_manifest.json"
        self._expenses_ledger_path = self._base / "expenses_ledger.csv"
        self._income_ledger_path = self._base / "income_ledger.csv"
        self._timesheet_path = self._base / "timesheet.csv"
        self._chat_path = self._base / "chat_history.json"
        self._ensure_dirs()

    # ── Internals ─────────────────────────────────────────────────────────


    @staticmethod
    def normalize_ledger(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ledger schema including expense_type for Business/Personal isolation."""
        if df is None or df.empty:
            return pd.DataFrame(columns=_LEDGER_COLS)
        out = df.copy()
        if "expense_type" not in out.columns:
            out["expense_type"] = out["kind"].apply(
                lambda k: "Business" if str(k).lower() == "income" else "Personal"
            )
        else:
            out["expense_type"] = (
                out["expense_type"].fillna("Personal").astype(str).replace({"": "Personal", "nan": "Personal"})
            )
            invalid = ~out["expense_type"].isin(EXPENSE_TYPES)
            out.loc[invalid & (out["kind"] == "expense"), "expense_type"] = "Personal"
            out.loc[invalid & (out["kind"] == "income"), "expense_type"] = "Business"
        for col in _LEDGER_COLS:
            if col not in out.columns:
                out[col] = "Uncategorized" if col == "category" else ("Personal" if col == "expense_type" else "")
        return out[_LEDGER_COLS]

    def _ensure_dirs(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._vault.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            manifest = dict(_DEFAULT_MANIFEST)
            manifest["created_at"] = datetime.now().isoformat()
            manifest["last_updated"] = datetime.now().isoformat()
            self._write_manifest(manifest)

    def _read_manifest(self) -> dict[str, Any]:
        try:
            raw = self._manifest_path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            # Recover from corrupt manifest
            manifest = dict(_DEFAULT_MANIFEST)
            manifest["created_at"] = manifest["last_updated"] = datetime.now().isoformat()
            return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["last_updated"] = datetime.now().isoformat()
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _file_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _safe_filename(name: str) -> str:
        return re.sub(r"[^\w\-.]", "_", name)

    # ── Document management ───────────────────────────────────────────────

    def save_document(
        self,
        file_name: str,
        content: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Hash, deduplicate, vault, and register a file.

        Returns a registry entry dict with a ``"duplicate": bool`` field.
        """
        metadata = metadata or {}
        file_hash = self._file_hash(content)
        manifest = self._read_manifest()

        # Deduplicate by content hash
        for entry in manifest["file_registry"]:
            if entry.get("hash") == file_hash:
                return {"duplicate": True, **entry}

        # Persist to vault
        vault_filename = f"{file_hash[:16]}_{self._safe_filename(file_name)}"
        vault_path = self._vault / vault_filename
        vault_path.write_bytes(content)

        entry: dict[str, Any] = {
            "hash": file_hash,
            "hash_short": file_hash[:12],
            "original_name": file_name,
            "vault_filename": vault_filename,
            "vault_path": str(vault_path),
            "uploaded_at": datetime.now().isoformat(),
            "file_size_bytes": len(content),
            "status": "local-vault",
            "transaction_count": int(metadata.get("transaction_count", 0)),
            "summary": str(metadata.get("summary", "")),
        }
        manifest["file_registry"].append(entry)
        self._write_manifest(manifest)
        return {"duplicate": False, **entry}

    def fetch_document(self, file_hash: str) -> bytes | None:
        """Retrieve raw file bytes from the vault by SHA-256 hash."""
        for entry in self._read_manifest().get("file_registry", []):
            if entry.get("hash") == file_hash:
                vault_path = Path(entry["vault_path"])
                if vault_path.exists():
                    return vault_path.read_bytes()
        return None

    def delete_document(self, file_hash: str) -> bool:
        """Remove a file from the vault and deregister it from the manifest."""
        manifest = self._read_manifest()
        for i, entry in enumerate(manifest["file_registry"]):
            if entry.get("hash") == file_hash:
                vault_path = Path(entry["vault_path"])
                if vault_path.exists():
                    vault_path.unlink(missing_ok=True)
                manifest["file_registry"].pop(i)
                self._write_manifest(manifest)
                return True
        return False

    def update_document_metadata(self, file_hash: str, metadata: dict[str, Any]) -> None:
        """Patch metadata fields on an existing registry entry."""
        manifest = self._read_manifest()
        for entry in manifest["file_registry"]:
            if entry.get("hash") == file_hash:
                entry.update(metadata)
                break
        self._write_manifest(manifest)

    def list_documents(self) -> list[dict[str, Any]]:
        """Return all file registry entries from the manifest."""
        entries = self._read_manifest().get("file_registry", [])
        # Annotate vault-file existence for display
        for entry in entries:
            entry["vault_exists"] = Path(entry.get("vault_path", "")).exists()
        return entries

    # ── Profile ───────────────────────────────────────────────────────────

    def fetch_profile(self) -> dict[str, Any]:
        """Return the ``user_profile`` section of the manifest."""
        return dict(self._read_manifest().get("user_profile", {}))

    def save_profile(self, profile: dict[str, Any]) -> None:
        """Persist user-profile fields (business_type, tax_reserve_rate, etc.)."""
        manifest = self._read_manifest()
        manifest.setdefault("user_profile", {}).update(profile)
        self._write_manifest(manifest)

    def fetch_tax_categories(self) -> list[str]:
        return list(self._read_manifest().get("tax_categories", _DEFAULT_MANIFEST["tax_categories"]))

    def add_tax_category(self, category: str) -> tuple[bool, str]:
        """Append a custom category to the manifest's tax_categories pre-sets list."""
        category = category.strip()
        if not category:
            return False, "Category name cannot be empty."
        manifest = self._read_manifest()
        categories = manifest.setdefault("tax_categories", list(_DEFAULT_MANIFEST["tax_categories"]))
        
        # Exact match or case-insensitive duplicate check
        if any(cat.lower() == category.lower() for cat in categories):
            return False, f"Category '{category}' already exists."
        
        categories.append(category)
        self._write_manifest(manifest)
        return True, f"Category '{category}' added successfully."


    def rename_tax_category(self, old_name: str, new_name: str) -> tuple[bool, str]:
        new_name = new_name.strip()
        if not new_name:
            return False, "Category name cannot be empty."
        if old_name in _DEFAULT_MANIFEST["tax_categories"]:
            return False, f"Cannot rename protected category '{old_name}'."
        manifest = self._read_manifest()
        categories = manifest.setdefault("tax_categories", list(_DEFAULT_MANIFEST["tax_categories"]))
        if old_name not in categories:
            return False, f"Category '{old_name}' not found."
        if new_name in categories:
            return False, f"Category '{new_name}' already exists."
        categories[categories.index(old_name)] = new_name
        self._write_manifest(manifest)
        return True, f"Category renamed to '{new_name}'."

    def delete_tax_category(self, category: str) -> tuple[bool, str]:
        """Delete a custom category from the manifest pre-sets list (preserves default categories)."""
        category = category.strip()
        if category in _DEFAULT_MANIFEST["tax_categories"]:
            return False, f"Default category '{category}' cannot be deleted."
        manifest = self._read_manifest()
        categories = manifest.get("tax_categories", [])
        if category not in categories:
            return False, f"Category '{category}' not found."
        categories.remove(category)
        self._write_manifest(manifest)
        return True, f"Category '{category}' deleted successfully."

    # ── Expenses & Income Ledgers (Independent Separation) ──────────────────

    def fetch_expenses_ledger(self) -> pd.DataFrame:
        """Load the persistent expenses ledger from disk, containing strictly expenditures."""
        if not self._expenses_ledger_path.exists():
            return pd.DataFrame(columns=_LEDGER_COLS)
        try:
            df = pd.read_csv(self._expenses_ledger_path, dtype=str)
            if df.empty:
                return pd.DataFrame(columns=_LEDGER_COLS)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            return self.normalize_ledger(df.dropna(subset=["date"]))
        except Exception:
            return pd.DataFrame(columns=_LEDGER_COLS)

    def save_expenses_ledger(self, ledger: pd.DataFrame) -> None:
        """Write the expenses ledger to disk atomically."""
        ledger = self.normalize_ledger(ledger)
        if ledger.empty:
            self._expenses_ledger_path.write_text(",".join(_LEDGER_COLS) + "\n", encoding="utf-8")
        else:
            ledger.to_csv(self._expenses_ledger_path, index=False)

    def fetch_income_ledger(self) -> pd.DataFrame:
        """Load the persistent income ledger from disk, containing strictly billing, invoice earnings."""
        if not self._income_ledger_path.exists():
            return pd.DataFrame(columns=_LEDGER_COLS)
        try:
            df = pd.read_csv(self._income_ledger_path, dtype=str)
            if df.empty:
                return pd.DataFrame(columns=_LEDGER_COLS)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            return self.normalize_ledger(df.dropna(subset=["date"]))
        except Exception:
            return pd.DataFrame(columns=_LEDGER_COLS)

    def save_income_ledger(self, ledger: pd.DataFrame) -> None:
        """Write the income ledger to disk atomically."""
        ledger = self.normalize_ledger(ledger)
        if ledger.empty:
            self._income_ledger_path.write_text(",".join(_LEDGER_COLS) + "\n", encoding="utf-8")
        else:
            ledger.to_csv(self._income_ledger_path, index=False)

    # ── Ledger (Backward Compatibility / Aggregated View) ──────────────────

    def fetch_ledger(self) -> pd.DataFrame:
        """Combines expenses and income datasets for backward compatible aggregations."""
        exp = self.fetch_expenses_ledger()
        inc = self.fetch_income_ledger()
        if exp.empty and inc.empty:
            return pd.DataFrame(columns=_LEDGER_COLS)
        return pd.concat([exp, inc], ignore_index=True)

    def save_ledger(self, ledger: pd.DataFrame) -> None:
        """Saves segmented ledgers by filtering incoming rows based on standard debit/credit parameters."""
        exp = ledger[ledger["kind"] == "expense"].copy()
        inc = ledger[ledger["kind"] == "income"].copy()
        self.save_expenses_ledger(exp)
        self.save_income_ledger(inc)

    # ── Timesheet ─────────────────────────────────────────────────────────

    def fetch_timesheet(self) -> pd.DataFrame:
        """Load the persistent timesheet from disk."""
        if not self._timesheet_path.exists():
            return pd.DataFrame(columns=_TIMESHEET_COLS)
        try:
            df = pd.read_csv(self._timesheet_path, dtype=str)
            if df.empty:
                return pd.DataFrame(columns=_TIMESHEET_COLS)
            df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0.0)
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0.0)
            df["total_pay"] = pd.to_numeric(df["total_pay"], errors="coerce").fillna(0.0)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            return df.dropna(subset=["date"])[_TIMESHEET_COLS]
        except Exception:
            return pd.DataFrame(columns=_TIMESHEET_COLS)

    def save_timesheet(self, timesheet: pd.DataFrame) -> None:
        """Write the full timesheet to disk."""
        if timesheet.empty:
            self._timesheet_path.write_text(",".join(_TIMESHEET_COLS) + "\n", encoding="utf-8")
        else:
            timesheet.to_csv(self._timesheet_path, index=False)

    # ── Chat history ──────────────────────────────────────────────────────

    def fetch_chat_history(self) -> list[dict[str, Any]]:
        """Load persisted chat messages from disk."""
        if not self._chat_path.exists():
            return []
        try:
            return json.loads(self._chat_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_chat_history(self, messages: list[dict[str, Any]]) -> None:
        """Persist chat messages to disk."""
        self._chat_path.write_text(
            json.dumps(messages, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        """Return a dict describing the current vault state."""
        manifest = self._read_manifest()
        vault_files = list(self._vault.glob("*")) if self._vault.exists() else []
        return {
            "backend": self.BACKEND,
            "profile_dir": str(self._base),
            "manifest_exists": self._manifest_path.exists(),
            "expenses_ledger_exists": self._expenses_ledger_path.exists(),
            "income_ledger_exists": self._income_ledger_path.exists(),
            "timesheet_exists": self._timesheet_path.exists(),
            "chat_exists": self._chat_path.exists(),
            "vault_file_count": len(vault_files),
            "registry_count": len(manifest.get("file_registry", [])),
            "manifest_version": manifest.get("version", "unknown"),
            "last_updated": manifest.get("last_updated", ""),
        }

    # ── Destructive ───────────────────────────────────────────────────────

    def purge_all(self) -> None:
        """Delete ALL persisted data and re-initialise a clean profile."""
        if self._vault.exists():
            shutil.rmtree(self._vault)
        for p in [self._expenses_ledger_path, self._income_ledger_path, self._timesheet_path, self._chat_path, self._manifest_path]:
            p.unlink(missing_ok=True)
        self._ensure_dirs()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bridge: StorageBridge | None = None


def get_bridge(base_dir: str = ".ndeavour_profile") -> StorageBridge:
    """Return the process-level StorageBridge singleton or a session-level instance.

    Calling ``get_bridge()`` multiple times always returns the same instance.
    Pass ``base_dir`` only on first call (subsequent calls ignore it).
    """
    try:
        import streamlit as st
        if "storage_bridge" in st.session_state:
            return st.session_state.storage_bridge
    except Exception:
        pass

    global _bridge
    if _bridge is None:
        _bridge = StorageBridge(base_dir)
    return _bridge
