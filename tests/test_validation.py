"""验证层测试：时间外验证、迁移性、需求删失分类。"""
from pathlib import Path

import pytest

from venueflow.io import load_tables, parse_at
from venueflow.model import fit
from venueflow.validation import (out_of_time, summarize_quality,
                                  transferability, calibration_gate,
                                  demand_censoring)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tables():
    return load_tables(ROOT)


def test_out_of_time_respects_cutoff(tables):
    """截点拟合只能用截点前闭场的证据。"""
    c1 = parse_at("2026-04-15T00:00:00+08:00")
    c3 = parse_at("2026-06-01T10:00:00+08:00")
    m1 = fit(tables, c1)
    m3 = fit(tables, c3)
    assert m1.n_matured_sessions < m3.n_matured_sessions
    assert m1.n_matured_bookings < m3.n_matured_bookings


def test_out_of_time_rows_and_quality(tables):
    rows = out_of_time(tables, tables["observation_cutoffs"])
    assert rows, "应产生时间外对照样本"
    # 所有对照场次闭场日都晚于各自截点
    assert all(r["lead_days"] >= 0 for r in rows)
    summary = summarize_quality(rows)
    assert summary["overall"]["n"] == len(rows)
    assert "by_lead_days" in summary and "by_venue" in summary


def test_calibration_gate(tables):
    ok, detail = calibration_gate(tables, tables["observation_cutoffs"])
    assert ok, detail
    assert detail["n"] > 0


def test_transferability(tables):
    rep = transferability(tables)
    assert set(rep["venue_rates"]) == {"A", "B", "C"}
    assert rep["venue_transferable"]          # 三馆出席率差在噪声内
    assert rep["party_icc"] is not None
    regimes = {(r["venue_id"], r["regime"]) for r in rep["policy_regimes"]}
    assert ("A", "POL-old") in regimes and ("A", "POL-new") in regimes
    assert ("C", "无政策记录") in regimes
    # 政策前后取消率差异应在噪声内（若未来数据改变，此检验应报警）
    a_old = next(r for r in rep["policy_regimes"] if r["regime"] == "POL-old")
    a_new = next(r for r in rep["policy_regimes"] if r["regime"] == "POL-new")
    assert abs(a_old["cancel_rate"] - a_new["cancel_rate"]) < 0.05


def test_demand_censoring_classification(tables):
    """区分限制停售 / 正常销售 / 无人需求；停售区间不计为低需求。"""
    s000 = demand_censoring(tables, "S000")
    assert s000["classification"] == "限制停售"
    assert s000["stop_windows"][0]["rejected_demand_persons"] > 0

    # 无售停、有被接受询价的场次
    free = None
    for s in tables["sessions"]:
        sid = s["session_id"]
        if not any(c["session_id"] == sid for c in tables["sales_controls"]):
            if any(e["session_id"] == sid and e["accepted"]
                   for e in tables["enquiries"]):
                free = sid
                break
    assert free is not None
    assert demand_censoring(tables, free)["classification"] == "正常销售"

    # 无售停且无询价的场次（若数据中存在）
    quiet = None
    for s in tables["sessions"]:
        sid = s["session_id"]
        if not any(c["session_id"] == sid for c in tables["sales_controls"]) \
                and not any(e["session_id"] == sid for e in tables["enquiries"]):
            quiet = sid
            break
    if quiet is not None:
        assert demand_censoring(tables, quiet)["classification"] == "无人需求"
