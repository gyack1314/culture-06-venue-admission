"""单场查询：风险、既有承诺、公益释放、候补处置与补偿暴露。"""
from __future__ import annotations

from .allocation import (charity_status, effective_limit, remaining_quota,
                         base_limit)
from .io import build_state, net_participants
from .model import Distribution, evidence_tier, fit, predict_session
from .validation import demand_censoring


def policy_context(tables, session_record, decisions):
    """场次的政策背景与决定中政策引用的一致性检查。"""
    venue = session_record["venue_id"]
    day = session_record["date"]
    regimes = [p for p in tables["policies"] if p["venue_id"] == venue]
    active = [p for p in regimes if p["start"] <= day < p["end"]]
    notes = []
    if not regimes:
        notes.append(f"场馆 {venue} 无任何政策记录，取消时限等政策参数无依据可查")
    elif not active:
        notes.append(f"场次日期 {day} 不在场馆 {venue} 任何政策生效区间内")
    for d in decisions:
        pid = d.get("policy_id")
        if not pid:
            continue
        ref = next((p for p in tables["policies"] if p["policy_id"] == pid), None)
        if ref is None:
            notes.append(f"决定 {d.get('decision_id')} 引用的政策 {pid} 不存在")
        elif ref["venue_id"] != venue:
            notes.append(
                f"决定 {d.get('decision_id')} 引用的政策 {pid} 属于场馆 "
                f"{ref['venue_id']}，与本场场馆 {venue} 不一致，"
                "不据此推断本场取消行为")
    return {
        "venue_policies": regimes,
        "active_policy": active[0] if active else None,
        "notes": notes,
    }


def session_status(tables, ledger, session_id, as_of):
    """汇总一场在 as_of 截点的全部运营视图。"""
    state = build_state(
        tables, session_id, as_of,
        accepted=ledger.accepted_bookings(session_id, as_of),
        decisions=ledger.decisions(session_id, as_of))
    model = fit(tables, as_of)
    charity = charity_status(state)
    limit, limit_source = effective_limit(state)
    remaining, _, sold, _ = remaining_quota(state)

    values, probs, notes = predict_session(model, state, mode="net")
    planned_ratio = (sold + charity["confirmed_net"]) / state.capacity \
        if state.capacity else 0.0
    tier, tier_notes = evidence_tier(model, state.session,
                                     planned_net_ratio=planned_ratio)
    dist = Distribution(values=values, probs=probs, tier=tier,
                        notes=tier_notes + notes)

    dispositions = ledger.waitlist_dispositions(session_id)
    waitlist_view = []
    for w in state.waitlist:
        disp = dispositions.get(w["wait_id"])
        waitlist_view.append({
            "wait_id": w["wait_id"], "party_id": w["party_id"],
            "participants": w["participants"], "submitted_at": w["submitted_at"],
            "disposition": (disp["action"] if disp else "未办理"),
            "reason": disp["reason"] if disp else None,
        })

    commitments = [{
        "booking_id": b["booking_id"], "party_id": b["party_id"],
        "reservation_type": b["reservation_type"],
        "participants": b["participants"],
        "cancelled_count": b["cancelled_count"],
        "net": net_participants(b),
    } for b in state.confirmed_bookings()]

    comp_budget = state.session["compensation_budget"]
    per_person = state.session["compensation_per_person"]
    exp_over = dist.expected_overage(state.capacity)

    return {
        "session_id": session_id,
        "as_of": as_of.isoformat() if as_of else None,
        "venue_id": state.session["venue_id"],
        "date": state.session["date"],
        "capacity": state.capacity,
        "matured": state.matured(),
        "risk": dist.summary(state.capacity),
        "risk_notes": dist.notes,
        "commitments": {
            "total_net": sum(c["net"] for c in commitments),
            "commercial_net": sold,
            "public_benefit_net": charity["confirmed_net"],
            "bookings": commitments,
        },
        "charity": charity,
        "limit": {
            "effective": limit, "source": limit_source,
            "base_without_oversell": base_limit(state),
            "remaining_commercial": remaining,
        },
        "waitlist": waitlist_view,
        "compensation_exposure": {
            "expected": round(exp_over * per_person, 2),
            "budget": comp_budget,
            "headroom": round(comp_budget - exp_over * per_person, 2),
            "p95_overage_persons": max(0, dist.quantile(0.95) - state.capacity),
        },
        "demand": demand_censoring(tables, session_id, as_of),
        "decisions": [{
            "decision_id": d.get("decision_id"), "at": d["at"],
            "commercial_limit": d["commercial_limit"],
            "policy_id": d.get("policy_id"),
            "origin": d.get("kind", "raw"),
        } for d in state.decisions],
        "policy_context": policy_context(tables, state.session, state.decisions),
        "boundaries": ledger.boundaries(session_id),
        "model_evidence": {
            "matured_bookings": model.n_matured_bookings,
            "matured_sessions": model.n_matured_sessions,
            "venue_sessions": model.venue_sessions,
            "max_historical_net_ratio": round(model.max_net_ratio, 4),
            "warnings": model.warnings,
        },
    }
