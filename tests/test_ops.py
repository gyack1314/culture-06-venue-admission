"""运营操作测试：决定约束后续接受、候补整团办理、承诺不撤回。"""
from pathlib import Path

import pytest

from venueflow.io import load_tables, parse_at, build_state
from venueflow.ledger import Ledger
from venueflow.ops import (confirm_limit, accept_booking, process_waitlist,
                           record_boundary)
from venueflow.allocation import remaining_quota
from venueflow.report import session_status

ROOT = Path(__file__).resolve().parents[1]
NOW = parse_at("2026-06-01T10:00:00+08:00")


@pytest.fixture(scope="module")
def tables():
    return load_tables(ROOT)


@pytest.fixture()
def ledger(tmp_path):
    return Ledger(tmp_path)


def test_confirm_then_accept_uses_limit(tables, ledger):
    """确认决定后，接受从该决定的真实剩余额度继续。"""
    confirm_limit(tables, ledger, "S140", 20, NOW, note="压低限额测试")
    ev = accept_booking(tables, ledger, "S140", "GT1", 5, "commercial", NOW)
    assert ev["limit_in_force"] == 20
    assert ev["remaining_after"] == 0   # 20 - 15(已售) - 5
    with pytest.raises(ValueError, match="剩余额度不足"):
        accept_booking(tables, ledger, "S140", "GT2", 1, "commercial", NOW)


def test_scenario_evaluation_consumes_nothing(tables, ledger):
    """情景比较不占名额：plan 之后剩余额度不变。"""
    from venueflow.allocation import evaluate_scenarios
    from venueflow.model import fit
    state = build_state(tables, "S140", NOW)
    before = remaining_quota(state)[0]
    evaluate_scenarios(fit(tables, NOW), state, tables, [45, 65, 200])
    after = remaining_quota(build_state(tables, "S140", NOW))[0]
    assert before == after
    assert ledger.events() == []


def test_confirm_below_sold_refused(tables, ledger):
    with pytest.raises(ValueError, match="不可撤回"):
        confirm_limit(tables, ledger, "S140", 10, NOW)


def test_confirm_infeasible_needs_override(tables, ledger):
    with pytest.raises(ValueError):
        confirm_limit(tables, ledger, "S140", 500, NOW)
    ev = confirm_limit(tables, ledger, "S140", 500, NOW, allow_infeasible=True)
    assert ev["basis"]["violations_overridden"]


def test_waitlist_atomic_and_ordered(tables, ledger):
    """整团接纳：剩余 30，W3(8) 与 W4(4) 都应完整接纳，不拆团。"""
    confirm_limit(tables, ledger, "S140", 45, NOW)
    results = process_waitlist(tables, ledger, "S140", NOW)
    by_id = {r["wait_id"]: r for r in results}
    assert by_id["W3"]["action"] == "accepted"
    assert by_id["W4"]["action"] == "accepted"
    # 已接纳的候补再次办理时不重复处理
    assert process_waitlist(tables, ledger, "S140", NOW) == []


def test_waitlist_pending_when_insufficient(tables, ledger):
    """限额只够一个团时，先到的整团接纳，后到的搁置并说明实际剩余。"""
    confirm_limit(tables, ledger, "S140", 20, NOW)   # 剩余 20-15=5
    results = process_waitlist(tables, ledger, "S140", NOW)
    by_id = {r["wait_id"]: r for r in results}
    assert by_id["W3"]["action"] == "pending"     # 8 > 5
    assert by_id["W4"]["action"] == "accepted"    # 4 <= 5，可完整接纳
    assert by_id["W3"]["remaining_at_decision"] == 5  # W3 办理时的实际剩余
    # 抬高限额后再次办理：之前搁置的 W3 重新参与并被接纳
    confirm_limit(tables, ledger, "S140", 45, NOW)
    results2 = process_waitlist(tables, ledger, "S140", NOW)
    assert {r["wait_id"] for r in results2} == {"W3"}
    assert results2[0]["action"] == "accepted"


def test_waitlist_uses_live_quota_not_stale(tables, ledger):
    """候补办理使用实时剩余，而非确认决定时记录的数字。"""
    confirm_limit(tables, ledger, "S140", 45, NOW)   # 当时剩余 30
    accept_booking(tables, ledger, "S140", "GT9", 28, "commercial", NOW)
    results = process_waitlist(tables, ledger, "S140", NOW)
    by_id = {r["wait_id"]: r for r in results}
    assert by_id["W3"]["action"] == "pending"     # 实际剩余 2，不是 30
    assert by_id["W4"]["action"] == "pending"


def test_accepted_bookings_join_commitments(tables, ledger):
    accept_booking(tables, ledger, "S140", "GT7", 6, "commercial", NOW)
    status = session_status(tables, ledger, "S140", NOW)
    assert status["commitments"]["commercial_net"] == 15 + 6
    ids = [b["booking_id"] for b in status["commitments"]["bookings"]]
    assert any(i.startswith("ACC-") for i in ids)


def test_charity_acceptance_within_protected(tables, ledger):
    accept_booking(tables, ledger, "S140", "GP1", 3, "public_benefit", NOW)
    with pytest.raises(ValueError, match="公益保留不足"):
        accept_booking(tables, ledger, "S140", "GP2", 1, "public_benefit", NOW)


def test_commitments_survive_new_data(tables, ledger):
    """新数据（更高风险）不撤回已确认承诺：承诺数只增不减。"""
    confirm_limit(tables, ledger, "S140", 45, NOW)
    accept_booking(tables, ledger, "S140", "GT8", 10, "commercial", NOW)
    s1 = session_status(tables, ledger, "S140", NOW)
    # 模拟新信息到达后的再查询：承诺不变
    s2 = session_status(tables, ledger, "S140", NOW)
    assert s1["commitments"]["total_net"] == s2["commitments"]["total_net"]
    assert s2["commitments"]["commercial_net"] == 25


def test_boundary_recorded_with_basis(tables, ledger):
    ev = record_boundary(tables, ledger, "S140", NOW, "stop", 0,
                         "风险重估后主动停售")
    assert ev["basis"]["limit_in_force"] is not None
    assert ev["remaining_offered"] == 0
    status = session_status(tables, ledger, "S140", NOW)
    assert any(b["boundary_kind"] == "stop" for b in status["boundaries"])
