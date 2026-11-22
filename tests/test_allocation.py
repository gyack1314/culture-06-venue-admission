"""票额安排测试：公益释放、限额约束、超放与补偿预算。"""
from pathlib import Path

import pytest

from venueflow.io import load_tables, parse_at, build_state
from venueflow.model import fit
from venueflow.allocation import (charity_status, base_limit, effective_limit,
                                  remaining_quota, check_limit,
                                  evaluate_scenarios, max_feasible_limit)

ROOT = Path(__file__).resolve().parents[1]
BEFORE_RELEASE = parse_at("2026-06-01T07:00:00+08:00")
AFTER_RELEASE = parse_at("2026-06-01T10:00:00+08:00")


@pytest.fixture(scope="module")
def tables():
    return load_tables(ROOT)


@pytest.fixture(scope="module")
def model(tables):
    return fit(tables, AFTER_RELEASE)


def test_charity_release_timing(tables):
    """公开截止前公益保留不释放；截止后只释放未确认部分。"""
    before = build_state(tables, "S140", BEFORE_RELEASE)
    after = build_state(tables, "S140", AFTER_RELEASE)
    cb, ca = charity_status(before), charity_status(after)
    assert cb["released"] == 0 and cb["still_locked"] == 3
    assert ca["released"] == 3 and ca["still_locked"] == 0
    assert ca["confirmed_net"] == 17 and ca["protected_capacity"] == 20
    assert base_limit(before) == 80 and base_limit(after) == 83


def test_effective_limit_prefers_decision(tables):
    state = build_state(tables, "S140", AFTER_RELEASE)
    limit, source = effective_limit(state)
    assert limit == 45 and source == "D1"   # 原始决定表中的已确认决定
    state2 = build_state(tables, "S141", AFTER_RELEASE)
    limit2, source2 = effective_limit(state2)
    assert source2 == "default-physical"    # 无决定时退回物理上限


def test_remaining_quota_live(tables):
    state = build_state(tables, "S140", AFTER_RELEASE)
    remaining, limit, sold, source = remaining_quota(state)
    assert limit == 45 and sold == 15 and remaining == 30


def test_limit_below_sold_rejected(tables, model):
    state = build_state(tables, "S140", AFTER_RELEASE)
    check = check_limit(model, state, tables, 10)   # 已售商业净额 15
    assert not check.feasible
    assert any("不可撤回" in v for v in check.violations)


def test_oversell_flag_and_alpha(tables, model):
    """超过基础可售即超放；把限额推到容量之上时 alpha 约束必须起作用。"""
    state = build_state(tables, "S140", AFTER_RELEASE)
    base = base_limit(state)
    check = check_limit(model, state, tables, base + 1)
    assert check.oversell
    # 极大限额：满售假设下到访均值远超容量，必违反 alpha
    huge = check_limit(model, state, tables, 400)
    assert not huge.feasible
    assert any("alpha" in v for v in huge.violations)


def test_compensation_budget_binds(tables, model):
    """把预算调小后，同样的限额必须被补偿约束拦下。"""
    state = build_state(tables, "S140", AFTER_RELEASE)
    session = dict(state.session)
    session["compensation_budget"] = 0
    session["alpha"] = 1.0   # 先放开 alpha，单独验证预算约束
    state.session = session
    check = check_limit(model, state, tables, 400)
    assert not check.feasible
    assert any("预算" in v for v in check.violations)


def test_scenario_evaluation_readonly(tables, model):
    """情景比较给出收入代理/满载风险/公益覆盖，且 feasible 判定一致。"""
    state = build_state(tables, "S140", AFTER_RELEASE)
    results = evaluate_scenarios(model, state, tables, [45, 65])
    assert len(results) == 2
    for r in results:
        assert r["feasible"]
        assert r["revenue_proxy"]["commercial_seats_sold"] == r["commercial_limit"]
        assert r["charity_coverage"]["confirmed_net"] == 17
        assert 0.0 <= r["p_over_capacity"] <= 1.0


def test_max_feasible_limit(tables, model):
    state = build_state(tables, "S140", AFTER_RELEASE)
    best = max_feasible_limit(model, state, tables)
    assert best is not None and best >= 65
    # 最大值 +1 必须不可行（或达到搜索上界）
    from venueflow.allocation import check_limit as _check
    assert not _check(model, state, tables, best + 1).feasible or best >= 200
