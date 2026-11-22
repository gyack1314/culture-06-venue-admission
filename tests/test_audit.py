"""审计测试：verify 必须放过合规台账、抓住违规台账。"""
from pathlib import Path

import pytest

from venueflow.io import load_tables, parse_at
from venueflow.ledger import Ledger
from venueflow.ops import confirm_limit, accept_booking, process_waitlist
from venueflow.audit import verify

ROOT = Path(__file__).resolve().parents[1]
NOW = parse_at("2026-06-01T10:00:00+08:00")


@pytest.fixture(scope="module")
def tables():
    return load_tables(ROOT)


@pytest.fixture()
def ledger(tmp_path):
    return Ledger(tmp_path)


def _cutoffs(tables):
    return tables["observation_cutoffs"]


def test_clean_ledger_passes(tables, ledger):
    confirm_limit(tables, ledger, "S140", 45, NOW)
    process_waitlist(tables, ledger, "S140", NOW)
    accept_booking(tables, ledger, "S140", "GT1", 6, "commercial", NOW)
    ok, findings = verify(tables, ledger, cutoffs=_cutoffs(tables))
    assert ok, [f for f in findings if not f["ok"]]


def test_over_quota_acceptance_caught(tables, ledger):
    confirm_limit(tables, ledger, "S140", 20, NOW)   # 剩余 5
    accept_booking(tables, ledger, "S140", "GT1", 5, "commercial", NOW)
    # 手工塞入一笔超额接受（绕过 accept_booking 的检查）
    ledger.append({
        "kind": "acceptance", "acceptance_id": "ACC-FAKE",
        "session_id": "S140", "at": NOW.isoformat(), "party_id": "GX",
        "participants": 3, "reservation_type": "commercial",
        "limit_in_force": 20, "remaining_after": 0,
    })
    ok, findings = verify(tables, ledger, cutoffs=_cutoffs(tables))
    assert not ok
    bad = [f for f in findings if not f["ok"]]
    assert any(f["check"] == "acceptance_within_quota" for f in bad)


def test_unjustified_decision_caught(tables, ledger):
    # 手工写入一个违反约束且未记录越权的决定
    ledger.append({
        "kind": "decision", "decision_id": "LD-BAD",
        "session_id": "S140", "at": NOW.isoformat(),
        "commercial_limit": 500, "basis": {},
    })
    ok, findings = verify(tables, ledger, cutoffs=_cutoffs(tables))
    assert not ok
    assert any(f["check"] == "decision_constraints" and not f["ok"]
               for f in findings)


def test_overridden_decision_passes_with_record(tables, ledger):
    confirm_limit(tables, ledger, "S140", 500, NOW, allow_infeasible=True)
    ok, findings = verify(tables, ledger, cutoffs=_cutoffs(tables))
    dec = [f for f in findings if f["check"] == "decision_constraints"]
    assert dec and all(f["ok"] for f in dec)


def test_waitlist_skip_caught(tables, ledger):
    confirm_limit(tables, ledger, "S140", 45, NOW)   # 剩余 30，两者都放得下
    # 手工写入：搁置更早的 W3（剩余足够），却接受了更晚的 W4
    ledger.append({
        "kind": "waitlist_disposition", "run_id": "WR-S140-001",
        "session_id": "S140", "at": NOW.isoformat(),
        "wait_id": "W3", "action": "pending",
        "reason": "人为跳过", "remaining_at_decision": 30,
    })
    ledger.append({
        "kind": "waitlist_disposition", "run_id": "WR-S140-001",
        "session_id": "S140", "at": NOW.isoformat(),
        "wait_id": "W4", "action": "accepted",
        "reason": "人为接受", "remaining_at_decision": 30,
    })
    ok, findings = verify(tables, ledger, cutoffs=_cutoffs(tables))
    assert not ok
    assert any(f["check"] == "waitlist_no_skip" and not f["ok"]
               for f in findings)


def test_charity_acceptance_caught(tables, ledger):
    # 公益保留只剩 3，却接受 5 人公益预约
    ledger.append({
        "kind": "acceptance", "acceptance_id": "ACC-PB",
        "session_id": "S140", "at": NOW.isoformat(), "party_id": "GP",
        "participants": 5, "reservation_type": "public_benefit",
    })
    ok, findings = verify(tables, ledger, cutoffs=_cutoffs(tables))
    assert not ok
    assert any(f["check"] == "charity_acceptance_within_protected"
               and not f["ok"] for f in findings)
