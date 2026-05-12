from __future__ import annotations

from llm_wallet_guard_study.db import connect, initialize_database
from llm_wallet_guard_study.simulator_wallet import SimulatorWallet


def test_simulator_balances_never_go_negative(tmp_path):
    db_path = tmp_path / "wallet.sqlite3"
    initialize_database(db_path)
    with connect(db_path) as conn:
        wallet = SimulatorWallet(conn)
        wallet.create_wallet("source", 2.0)
        transfer_id = wallet.propose_transfer("global", "source", "dest", 5.0, "test")
        wallet.queue_transfer(transfer_id)
        assert wallet.get_balance("source") == 2.0
        transfer = wallet.list_transfers("global")[0]
        assert transfer["status"] == "blocked"
        assert transfer["policy_reason_code"] == "amount_exceeds_balance"
