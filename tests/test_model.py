"""模型层测试：版本消解、成熟性、团体相关结构、卷积与界限。"""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from venueflow.io import (load_tables, resolve_bookings, build_state,
                          matured_bookings, parse_at, net_participants)
from venueflow.model import (fit, convolve, predict_session, evidence_tier,
                             Distribution)
from venueflow import TIER_CALIBRATED

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))


@pytest.fixture(scope="module")
def tables():
    return load_tables(ROOT)


def test_revision_full_replacement(tables):
    """同 booking_id 多 revision 完整替代，不相加；截点控制可见版本。"""
    allr = resolve_bookings(tables["bookings"])
    b = next(x for x in allr if x["booking_id"] == "B0841")
    assert b["revision"] == 2 and b["party_id"] == "G11"
    assert b["reservation_type"] == "public_benefit"
    early = resolve_bookings(tables["bookings"],
                             as_of=parse_at("2026-03-01T00:00:00+08:00"))
    b1 = next(x for x in early if x["booking_id"] == "B0841")
    assert b1["revision"] == 1 and b1["reservation_type"] == "commercial"


def test_immature_never_label(tables):
    """未闭场场次不进入拟合标签；actual_count 缺失不补 0。"""
    matured = matured_bookings(tables, None)
    sids = {b["session_id"] for b, _ in matured}
    assert "S140" not in sids  # 未闭场
    assert all(b["actual_count"] is not None for b, _ in matured)
    # 截点 C1 之前闭场的场次才计入
    at = parse_at("2026-04-15T00:00:00+08:00")
    matured_c1 = matured_bookings(tables, at)
    assert len(matured_c1) < len(matured)
    assert all(s["date"] < "2026-04-15" for _, s in matured_c1)


def test_public_benefit_deterministic(tables):
    model = fit(tables)
    tm = model.types["public_benefit"]
    assert tm.deterministic_net
    v, p = tm.show_pmf(7)
    assert list(v) == [7] and list(p) == [1.0]


def test_commercial_group_coherence(tables):
    """商业团体缺席分布支撑集很小：拒绝逐人独立硬币假设。"""
    model = fit(tables)
    tm = model.types["commercial"]
    support, pmf = tm.by_size[8]
    assert max(support) <= 3  # 8 人团缺席不超过 3
    # 逐人二项( p≈0.23 )下 P(缺席>=5) 明显为正，经验分布必须为 0
    assert pmf[support.index(max(support))] < 0.5


def test_convolve_matches_brute_force():
    pmfs = [(np.array([0, 1, 2]), np.array([0.2, 0.5, 0.3])),
            (np.array([0, 1]), np.array([0.4, 0.6]))]
    values, probs = convolve(pmfs)
    brute = {}
    for a, pa in zip(*pmfs[0]):
        for b, pb in zip(*pmfs[1]):
            brute[a + b] = brute.get(a + b, 0) + pa * pb
    for v, p in zip(values, probs):
        assert p == pytest.approx(brute[v])
    assert probs.sum() == pytest.approx(1.0)


def test_bounds_dominate_exact(tables):
    """无分布界限必须覆盖精确尾部概率。"""
    model = fit(tables)
    state = build_state(tables, "S140", parse_at("2026-06-01T07:00:00+08:00"))
    values, probs, _ = predict_session(model, state)
    dist = Distribution(values=values, probs=probs, tier=TIER_CALIBRATED)
    cap = state.capacity
    exact = dist.prob_over(cap)
    assert dist.markov_bound(cap + 1) >= exact - 1e-9
    assert dist.cantelli_bound(cap + 1) >= exact - 1e-9


def test_session_level_tighter_than_independent(tables):
    """已知构成族：场次残差模型方差远小于独立卷积（数据事实）。"""
    model = fit(tables)
    state = build_state(tables, "S140", parse_at("2026-06-01T07:00:00+08:00"))
    values, probs, notes = predict_session(model, state)
    assert not any("独立卷积" in n for n in notes)  # 未整单退回独立卷积
    var = float(((values - values @ probs) ** 2) @ probs)
    assert var < 4.0  # 独立卷积会给出 ~2.6^2 ≈ 6.8 以上


def test_unfamiliar_composition_falls_back(tables):
    """陌生构成（预约数超过历史最大值）退回独立卷积并标注。"""
    model = fit(tables)
    state = build_state(tables, "S140", parse_at("2026-06-01T07:00:00+08:00"))
    extra = [dict(b, booking_id=f"X{i}", party_id=f"GX{i}")
             for i, b in enumerate(state.bookings[:5])]
    state.accepted = extra
    values, probs, notes = predict_session(model, state)
    assert any("独立卷积" in n for n in notes)


def test_gross_mode_ignores_cancellations(tables):
    """gross 模式不读取 cancelled_count（截点验证不使用截点后取消）。"""
    model = fit(tables)
    tm = model.types[("commercial", "gross")]
    assert tm.mode == "gross"
    # 同一 gross 规模同一 pmf，与取消数无关
    v1, p1 = tm.show_pmf(8)
    v2, p2 = tm.show_pmf(8)
    assert list(v1) == list(v2) and list(p1) == list(p2)


def test_evidence_tier_new_venue(tables):
    """成熟证据不足的场馆降级；全新场馆只给界限。"""
    model = fit(tables)
    tier, notes = evidence_tier(model, {"venue_id": "Z", "capacity": 50})
    assert tier == "extrapolated"
    assert any("Z" in n for n in notes)
    tier, notes = evidence_tier(model, {"venue_id": "C", "capacity": 100},
                                planned_net_ratio=0.99)
    assert tier == "extrapolated"
    assert any("外推" in n for n in notes)
