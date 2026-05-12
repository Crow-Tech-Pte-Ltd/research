"""Deterministic SQLite-backed simulator wallet."""

from __future__ import annotations

import sqlite3
from decimal import Decimal, InvalidOperation
from typing import Any

from .db import utc_now_iso
from .redaction import redact_text


def _amount_to_float(amount: Any) -> float | None:
    if amount is None or isinstance(amount, bool):
        return None
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    return float(value)


class SimulatorWallet:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_wallet(self, wallet_id: str, initial_balance: float, trial_id: str = "global") -> None:
        amount = _amount_to_float(initial_balance)
        if amount is None or amount < 0:
            raise ValueError("initial_balance must be nonnegative")
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO wallets(id, trial_id, wallet_id, balance, created_at_utc, updated_at_utc)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(trial_id, wallet_id) DO NOTHING
            """,
            (f"{trial_id}:{wallet_id}", trial_id, wallet_id, amount, now, now),
        )

    def get_balance(self, wallet_id: str, trial_id: str = "global") -> float:
        row = self.conn.execute(
            "SELECT balance FROM wallets WHERE trial_id = ? AND wallet_id = ?",
            (trial_id, wallet_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown simulator wallet: {wallet_id}")
        return float(row["balance"])

    def propose_transfer(
        self,
        trial_id: str,
        from_wallet: str | None,
        to_wallet: str | None,
        amount: Any,
        reason: str = "",
        source_text: str = "",
        transfer_id: str | None = None,
    ) -> str:
        transfer_id = transfer_id or f"transfer-{trial_id}-{utc_now_iso()}"
        parsed_amount = _amount_to_float(amount)
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO simulator_transfers(
                id, trial_id, from_wallet, to_wallet, amount, status, reason, source_text,
                policy_reason_code, created_at_utc, updated_at_utc
            )
            VALUES(?, ?, ?, ?, ?, 'proposed', ?, ?, NULL, ?, ?)
            """,
            (
                transfer_id,
                trial_id,
                from_wallet,
                to_wallet,
                parsed_amount,
                redact_text(reason),
                redact_text(source_text),
                now,
                now,
            ),
        )
        return transfer_id

    def queue_transfer(self, transfer_id: str, reason_code: str = "allowed_simulator_condition") -> None:
        row = self.conn.execute(
            "SELECT trial_id, from_wallet, amount FROM simulator_transfers WHERE id = ?",
            (transfer_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown simulator transfer: {transfer_id}")
        amount = row["amount"]
        if amount is None or amount < 0:
            self.block_transfer(transfer_id, "invalid_amount")
            return
        if row["from_wallet"]:
            balance = self.get_balance(row["from_wallet"], row["trial_id"])
            if amount > balance:
                self.block_transfer(transfer_id, "amount_exceeds_balance")
                return
        self.conn.execute(
            """
            UPDATE simulator_transfers
            SET status = 'queued', policy_reason_code = ?, updated_at_utc = ?
            WHERE id = ?
            """,
            (reason_code, utc_now_iso(), transfer_id),
        )

    def block_transfer(self, transfer_id: str, reason_code: str) -> None:
        self.conn.execute(
            """
            UPDATE simulator_transfers
            SET status = 'blocked', policy_reason_code = ?, updated_at_utc = ?
            WHERE id = ?
            """,
            (reason_code, utc_now_iso(), transfer_id),
        )

    def list_transfers(self, trial_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM simulator_transfers WHERE trial_id = ? ORDER BY created_at_utc, id",
                (trial_id,),
            )
        )
