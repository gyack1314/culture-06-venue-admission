"""时间外验证、分提前期预测质量与证据迁移性检验。

原则：
- 验证一律使用 gross 模式（不依赖取消时点），避免把截点后才发生的
  取消当作截点已知；运营预测使用的 net 模式其保守性由上界性质保证；
- 截点之后才闭场的场次不进入该截点的拟合；
- 接待部门主动限售的区间（sales_controls）不是需求为 0 的证据，
  需求侧结论只使用 enquiries 中可见的到达与被拒记录。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np

from . import TIER_CALIBRATED
from .io import (matured_bookings, parse_at, parse_day, resolve_bookings,
                 net_participants, build_state)
from .model import fit, predict_session

LEAD_BUCKETS = ((0, 7), (7, 30), (30, 90), (90, 366))
COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def _interval_hit(dist_values, dist_probs, level, actual):
    cdf = np.cumsum(dist_probs)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lo = dist_values[int(np.searchsorted(cdf, lo_q, side="left"))]
    hi = dist_values[min(int(np.searchsorted(cdf, hi_q, side="left")), len(dist_values) - 1)]
    return lo <= actual <= hi


def out_of_time(tables, cutoffs):
    """对每个截点：用截点前成熟证据拟合，预测截点后闭场的场次。

    返回每场一条的预测-实际对照，以及按提前期/场馆的聚合。
    """
    rows = []
    for cutoff in cutoffs:
        at = parse_at(cutoff["at"]) if isinstance(cutoff, dict) else parse_at(cutoff)
        model = fit(tables, at)
        sessions = {s["session_id"]: s for s in tables["sessions"]}
        actual = defaultdict(int)
        for b, session in matured_bookings(tables, None):
            actual[b["session_id"]] += b["actual_count"]
        for sid, session in sessions.items():
            if parse_day(session["date"]) >= at.date():
                # 截点后才闭场：属于该截点的"未来"
                if sid not in actual:
                    continue  # 至今未成熟，无法对照
                state = build_state(tables, sid, at)
                values, probs, _ = predict_session(model, state, mode="gross")
                lead = (parse_day(session["date"]) - at.date()).days
                row = {
                    "cutoff": cutoff["cutoff_id"] if isinstance(cutoff, dict) else str(cutoff),
                    "session_id": sid,
                    "venue_id": session["venue_id"],
                    "lead_days": lead,
                    "actual": actual[sid],
                    "mean": float(values @ probs),
                    "hits": {lv: _interval_hit(values, probs, lv, actual[sid])
                             for lv in COVERAGE_LEVELS},
                }
                row["abs_err"] = abs(row["mean"] - row["actual"])
                rows.append(row)
    return rows


def summarize_quality(rows):
    """按提前期桶与场馆聚合覆盖率/误差。"""
    def bucket(lead):
        for lo, hi in LEAD_BUCKETS:
            if lo <= lead < hi:
                return f"{lo}-{hi}"
        return "other"

    def agg(group):
        n = len(group)
        if not n:
            return None
        out = {"n": n,
               "mae": round(sum(r["abs_err"] for r in group) / n, 3),
               "bias": round(sum(r["mean"] - r["actual"] for r in group) / n, 3)}
        for lv in COVERAGE_LEVELS:
            out[f"coverage_{int(lv*100)}"] = round(
                sum(1 for r in group if r["hits"][lv]) / n, 3)
        return out

    by_lead, by_venue = defaultdict(list), defaultdict(list)
    for r in rows:
        by_lead[bucket(r["lead_days"])].append(r)
        by_venue[r["venue_id"]].append(r)
    return {
        "overall": agg(rows),
        "by_lead_days": {k: agg(v) for k, v in sorted(by_lead.items())},
        "by_venue": {k: agg(v) for k, v in sorted(by_venue.items())},
    }


def transferability(tables, as_of=None):
    """检验哪些场馆与时段的证据足够迁移，哪些必须降级为界限。"""
    pairs = defaultdict(lambda: defaultdict(list))   # venue -> session -> (actual, net)
    weekday = defaultdict(lambda: [0, 0])
    policy_rows = defaultdict(lambda: [0, 0, 0, 0])  # (venue, regime) -> actual, net, canc, gross
    party_ratios = defaultdict(list)
    sessions = {s["session_id"]: s for s in tables["sessions"]}
    policies = tables["policies"]

    def regime(session):
        for p in policies:
            if p["venue_id"] == session["venue_id"] and \
               p["start"] <= session["date"] < p["end"]:
                return p["policy_id"]
        return "无政策记录"

    for b, session in matured_bookings(tables, as_of):
        net = net_participants(b)
        if net <= 0:
            continue
        ratio = b["actual_count"] / net
        pairs[session["venue_id"]][b["session_id"]].append((b["actual_count"], net))
        wd = parse_day(session["date"]).weekday()
        weekday[wd][0] += b["actual_count"]
        weekday[wd][1] += net
        reg = regime(session)
        key = (session["venue_id"], reg)
        policy_rows[key][0] += b["actual_count"]
        policy_rows[key][1] += net
        policy_rows[key][2] += b["cancelled_count"]
        policy_rows[key][3] += b["participants"]
        party_ratios[(b["party_id"], b["session_id"])].append(ratio)

    venue_rates = {}
    for venue, sess in pairs.items():
        a = sum(x for ses in sess.values() for x, _ in ses)
        n = sum(y for ses in sess.values() for _, y in ses)
        ns = len(sess)
        venue_rates[venue] = {
            "rate": round(a / n, 4), "sessions": ns,
            "se": round((a / n * (1 - a / n) / max(n, 1)) ** 0.5, 4),
        }
    rates = [v["rate"] for v in venue_rates.values()]
    venue_spread = (max(rates) - min(rates)) if rates else None

    # 团体内部相关（同场次同团体多条预约的出席率 ICC，ANOVA 估计）
    groups = [v for v in party_ratios.values() if len(v) > 1]
    icc = None
    if len(groups) >= 10:
        allr = [r for g in groups for r in g]
        grand = sum(allr) / len(allr)
        k = len(groups)
        n_tot = len(allr)
        msb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups) / (k - 1)
        msw = sum(sum((r - sum(g) / len(g)) ** 2 for r in g) for g in groups) / (n_tot - k)
        n0 = (n_tot - sum(len(g) ** 2 for g in groups) / n_tot) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) else 0.0

    policy_report = []
    for (venue, reg), (a, n, c, g) in sorted(policy_rows.items()):
        policy_report.append({
            "venue_id": venue, "regime": reg,
            "show_rate": round(a / n, 4) if n else None,
            "cancel_rate": round(c / g, 4) if g else None,
            "net_persons": n,
        })

    weekday_rates = {d: round(a / n, 4) for d, (a, n) in sorted(weekday.items())}
    return {
        "venue_rates": venue_rates,
        "venue_spread": round(venue_spread, 4) if venue_spread is not None else None,
        "venue_transferable": venue_spread is not None and venue_spread < 0.05,
        "weekday_rates": weekday_rates,
        "weekday_spread": round(max(weekday_rates.values()) - min(weekday_rates.values()), 4)
        if weekday_rates else None,
        "policy_regimes": policy_report,
        "party_icc": round(icc, 4) if icc is not None else None,
        "party_icc_note": "同团体跨预约相关未检出" if icc is not None and icc < 0.05
        else "同团体存在可检出的跨预约相关，需按团体整体建模",
    }


def demand_censoring(tables, session_id, as_of=None):
    """区分未售、限制停售与无人需求。

    返回该场次的售停区间、区间内被拒需求（enquiries reason=sales_stop）
    与需求证据分类。售停区间内的低销量不得作为低需求证据。
    """
    state = build_state(tables, session_id, as_of)
    stops = []
    for c in state.controls:
        start, end = parse_at(c["start_at"]), parse_at(c["end_at"])
        rejected = [e for e in state.enquiries
                    if not e["accepted"] and e["reason"] == "sales_stop"
                    and start <= parse_at(e["at"]) < end]
        stops.append({
            "control_id": c["control_id"], "kind": c["kind"],
            "start_at": c["start_at"], "end_at": c["end_at"],
            "remaining_offered": c["remaining_offered"],
            "rejected_demand_persons": sum(e["party_size"] for e in rejected),
            "rejected_demand_parties": len(rejected),
        })
    accepted = [e for e in state.enquiries if e["accepted"]]
    if stops:
        classification = "限制停售"
        note = "存在售停区间，区间内销量被主动限制，不能作为需求证据"
    elif accepted:
        classification = "正常销售"
        note = "无售停限制，存在被接受的询价"
    elif state.enquiries:
        classification = "未售"
        note = "无售停限制，询价未被接受但非限售原因"
    else:
        classification = "无人需求"
        note = "无售停限制且无任何询价记录"
    return {
        "session_id": session_id,
        "classification": classification,
        "note": note,
        "stop_windows": stops,
        "accepted_enquiries": len(accepted),
    }


def calibration_gate(tables, cutoffs, z=2.5):
    """校准门槛：时间外覆盖率与名义水平的偏差在二项抽样噪声内。

    对每个名义水平 lv，以 n 个时间外场次做二项检验：|observed-lv|
    <= z * sqrt(lv(1-lv)/n) 视为通过。返回 (是否通过, 明细)。
    运营确认决定时必须附带该结果。
    """
    rows = out_of_time(tables, cutoffs)
    summary = summarize_quality(rows)
    overall = summary["overall"] or {}
    n = overall.get("n", 0)
    checks = []
    ok = True
    for lv in COVERAGE_LEVELS:
        key = f"coverage_{int(lv*100)}"
        got = overall.get(key)
        if got is None or not n:
            ok = False
            checks.append({"level": lv, "ok": False, "note": "无时间外样本"})
            continue
        tol = z * (lv * (1 - lv) / n) ** 0.5
        good = abs(got - lv) <= tol
        ok = ok and good
        checks.append({"level": lv, "nominal": lv, "actual": got,
                       "tolerance": round(tol, 4), "ok": good})
    return ok, {"checks": checks, "n": n, "overall": overall,
                "by_lead_days": summary["by_lead_days"],
                "by_venue": summary["by_venue"]}
