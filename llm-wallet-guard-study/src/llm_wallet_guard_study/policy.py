"""Deterministic fail-closed policy enforcement."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .db import utc_now_iso
from .redaction import redact_json_dumps


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason_code: str
    details: dict[str, Any]
    event_id: str


class PolicyEnforcer:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def evaluate_transfer(
        self,
        trial_id: str | None,
        attempt_number: int | None,
        request: dict[str, Any],
        simulator_transfer_id: str | None = None,
        actor: str = "guardian",
    ) -> PolicyDecision:
        trial = self._load_trial_state(trial_id)
        if trial is None:
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "blocked_fail_closed",
                {"reason": "missing_trial_state", "request": request},
            )

        from_wallet = request.get("from_wallet")
        to_wallet = request.get("to_wallet")
        amount_value = request.get("amount")

        if not from_wallet or not to_wallet or amount_value is None:
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "malformed_request",
                {"reason": "missing_required_transfer_field", "request": request},
            )

        amount = _parse_amount(amount_value)
        if amount is None or amount <= 0:
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "invalid_amount",
                {"amount": amount_value, "request": request},
            )

        if from_wallet != trial["source_wallet_id"] or not self._wallet_exists(trial_id, from_wallet):
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "unknown_wallet",
                {
                    "reason": "source_wallet_unknown_or_mismatch",
                    "expected_source_wallet": trial["source_wallet_id"],
                    "request": request,
                },
            )

        allowlist = json.loads(trial["allowlisted_destinations_json"])
        if to_wallet not in allowlist or to_wallet != trial["destination_wallet_id"]:
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "destination_not_allowlisted",
                {
                    "expected_destination_wallet": trial["destination_wallet_id"],
                    "allowlisted_destinations": allowlist,
                    "request": request,
                },
            )

        if not self._wallet_exists(trial_id, to_wallet):
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "unknown_wallet",
                {"reason": "destination_wallet_unknown", "request": request},
            )

        amount_cap = Decimal(str(trial["amount_cap"]))
        if amount > amount_cap:
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "amount_exceeds_cap",
                {"amount": str(amount), "amount_cap": str(amount_cap), "request": request},
            )

        if actor == "guardian" and not bool(trial["allow_simulator_transfers"]):
            return self._record(
                trial_id,
                attempt_number,
                simulator_transfer_id,
                False,
                "guardian_transfer_forbidden",
                {"guardian_policy": trial["guardian_policy"], "request": request},
            )

        return self._record(
            trial_id,
            attempt_number,
            simulator_transfer_id,
            True,
            "allowed_simulator_condition",
            {"request": request},
        )

    def _load_trial_state(self, trial_id: str | None) -> sqlite3.Row | None:
        if not trial_id:
            return None
        return self.conn.execute(
            """
            SELECT
                t.id AS trial_id,
                t.source_wallet_id,
                t.destination_wallet_id,
                c.allowlisted_destinations_json,
                c.amount_cap,
                c.allow_simulator_transfers,
                c.guardian_policy
            FROM trials t
            JOIN trial_conditions c ON c.id = t.condition_record_id
            WHERE t.id = ?
            """,
            (trial_id,),
        ).fetchone()

    def _wallet_exists(self, trial_id: str, wallet_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM wallets WHERE trial_id = ? AND wallet_id = ?",
            (trial_id, wallet_id),
        ).fetchone()
        return row is not None

    def _record(
        self,
        trial_id: str | None,
        attempt_number: int | None,
        simulator_transfer_id: str | None,
        allowed: bool,
        reason_code: str,
        details: dict[str, Any],
    ) -> PolicyDecision:
        count = self.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM policy_events
            WHERE trial_id IS ? AND attempt_number IS ? AND simulator_transfer_id IS ?
            """,
            (trial_id, attempt_number, simulator_transfer_id),
        ).fetchone()["count"]
        event_id = f"policy-{trial_id or 'none'}-{attempt_number or 0}-{count + 1}"
        self.conn.execute(
            """
            INSERT INTO policy_events(
                id, trial_id, attempt_number, simulator_transfer_id, decision,
                reason_code, details_json, created_at_utc
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                trial_id,
                attempt_number,
                simulator_transfer_id,
                "allow" if allowed else "block",
                reason_code,
                redact_json_dumps(details),
                utc_now_iso(),
            ),
        )
        return PolicyDecision(allowed, reason_code, details, event_id)


def _parse_amount(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite():
        return None
    return amount
