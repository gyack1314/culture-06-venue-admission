"""运营操作：确认限额、接受预约、办理候补。

铁律：
- 任何已接受的新预约都从同场次真实剩余额度继续（实时重算，不用旧建议）；
- 决定确认后影响后续接受；情景评估永远不写台账；
- 已确认承诺不因预测变化被无声撤回——风险升高只能产生新的销售边界
  （停售/降额建议），由 verify 复核一致性。
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .io import build_state, net_participants
from .allocation import (charity_status, check_limit, effective_limit,
                         remaining_quota)
from .model import fit


def _state_with_ledger(tables, ledger, session_id, as_of):
    return build_state(
        tables, session_id, as_of,
        accepted=ledger.accepted_bookings(session_id, as_of),
        decisions=ledger.decisions(session_id, as_of),
    )


def confirm_limit(tables, ledger, session_id, limit, at, note="", allow_infeasible=False):
    """确认一个商业限额：写入台账并约束后续接受。

    确认前重新拟合模型、重算约束；不满足约束默认拒绝。
    返回决定事件（含分析依据）。
    """
    state = _state_with_ledger(tables, ledger, session_id, at)
    model = fit(tables, at)
    check = check_limit(model, state, tables, limit)
    if not check.feasible and not allow_infeasible:
        raise ValueError("限额不满足约束，拒绝确认: " + "; ".join(check.violations))
    seq = len(ledger.decisions(session_id)) + len(
        [d for d in tables["decisions"] if d["session_id"] == session_id])
    event = {
        "kind": "decision",
        "decision_id": f"LD-{session_id}-{seq:03d}",
        "session_id": session_id,
        "at": at.isoformat(),
        "commercial_limit": limit,
        "note": note,
        "basis": {
            "model_matured_bookings": model.n_matured_bookings,
            "model_matured_sessions": model.n_matured_sessions,
            "risk": check.risk,
            "compensation": check.compensation,
            "oversell": check.oversell,
            "warnings": check.warnings,
            "violations_overridden": check.violations if allow_infeasible else [],
        },
    }
    ledger.append(event)
    return event


def record_boundary(tables, ledger, session_id, at, boundary_kind,
                    remaining_offered, reason):
    """记录一次销售边界（停售/重开/调额建议）及其依据。

    用于区分未售、限制停售与无人需求：边界事件是"限制停售"的唯一来源。
    """
    state = _state_with_ledger(tables, ledger, session_id, at)
    limit, source = effective_limit(state)
    sold = state.net_by_type("commercial")
    event = {
        "kind": "boundary",
        "boundary_id": f"BND-{len(ledger.boundaries()) + 1:04d}",
        "session_id": session_id,
        "at": at.isoformat(),
        "boundary_kind": boundary_kind,   # stop | reopen | limit_advice
        "remaining_offered": remaining_offered,
        "reason": reason,
        "basis": {
            "limit_in_force": limit, "limit_source": source,
            "commercial_net_sold": sold,
        },
    }
    ledger.append(event)
    return event


def accept_booking(tables, ledger, session_id, party_id, participants,
                   reservation_type, at):
    """接受新预约。返回接受事件；额度不足时拒绝并说明。

    商业：占用生效限额下的实际剩余；公益：占用 protected_capacity 内
    尚未确认部分（公益保留是承诺，不挤占商业池，也不被商业挤占）。
    """
    if participants <= 0:
        raise ValueError("participants 必须为正")
    state = _state_with_ledger(tables, ledger, session_id, at)
    if reservation_type == "commercial":
        remaining, limit, sold, source = remaining_quota(state)
        if participants > remaining:
            raise ValueError(
                f"商业剩余额度不足：剩余 {remaining}（限额 {limit} 来源 {source}，"
                f"已售净额 {sold}），申请 {participants}")
    elif reservation_type == "public_benefit":
        charity = charity_status(state)
        free = charity["protected_capacity"] - charity["confirmed_net"]
        if participants > free:
            raise ValueError(
                f"公益保留不足：剩余 {free}（保留 {charity['protected_capacity']}，"
                f"已确认净额 {charity['confirmed_net']}），申请 {participants}")
    else:
        raise ValueError(f"未知预约类型: {reservation_type}")
    event = {
        "kind": "acceptance",
        "acceptance_id": f"ACC-{len(ledger.events('acceptance')) + 1:04d}",
        "session_id": session_id,
        "at": at.isoformat(),
        "party_id": party_id,
        "participants": participants,
        "reservation_type": reservation_type,
        "limit_in_force": (effective_limit(state)[0]
                           if reservation_type == "commercial" else None),
        "remaining_after": (remaining_quota(state)[0] - participants
                            if reservation_type == "commercial"
                            else None),
    }
    ledger.append(event)
    return event


def process_waitlist(tables, ledger, session_id, at):
    """按 submitted_at 办理候补：整团接纳，不拆团，不跳过可完整接受的更早团体。

    同一 submitted_at（同优先级）内允许比较可行组合，取能接纳团体数最多
    的组合；剩余额度每次实时重算。已被接纳的候补不再办理；被搁置的候补
    在下次办理时重新参与（额度可能已变化）。
    """
    state = _state_with_ledger(tables, ledger, session_id, at)
    done = {wid: d for wid, d in ledger.waitlist_dispositions(session_id).items()
            if d["action"] == "accepted"}
    pending = [w for w in state.waitlist if w["wait_id"] not in done]
    pending.sort(key=lambda w: (w["submitted_at"], w["wait_id"]))
    if not pending:
        return []
    past_runs = {d.get("run_id") for d in ledger.events("waitlist_disposition")}
    run_id = f"WR-{session_id}-{len(past_runs) + 1:03d}"
    results = []
    for ts, group in _group_by_time(pending):
        remaining, limit, sold, source = remaining_quota(
            _state_with_ledger(tables, ledger, session_id, at))
        chosen = _best_feasible_subset(group, remaining)
        chosen_ids = {w["wait_id"] for w in chosen}
        leftover = remaining - sum(w["participants"] for w in chosen)
        running = remaining
        for w in group:
            if w["wait_id"] in chosen_ids:
                ev = accept_booking(tables, ledger, session_id, w["party_id"],
                                    w["participants"], "commercial", at)
                disp = {
                    "kind": "waitlist_disposition",
                    "run_id": run_id,
                    "session_id": session_id, "at": at.isoformat(),
                    "wait_id": w["wait_id"], "action": "accepted",
                    "reason": "整团接纳，额度充足",
                    "remaining_at_decision": running,
                    "acceptance_id": ev["acceptance_id"],
                }
                running -= w["participants"]
            else:
                disp = {
                    "kind": "waitlist_disposition",
                    "run_id": run_id,
                    "session_id": session_id, "at": at.isoformat(),
                    "wait_id": w["wait_id"], "action": "pending",
                    "reason": f"整团接纳后剩余 {leftover} 不足以接纳本团 "
                              f"{w['participants']} 人（限额 {limit} 来源 {source}）",
                    "remaining_at_decision": leftover,
                }
            ledger.append(disp)
            results.append(disp)
    return results


def _group_by_time(pending):
    groups = defaultdict(list)
    for w in pending:
        groups[w["submitted_at"]].append(w)
    for ts in sorted(groups):
        yield ts, groups[ts]


def _best_feasible_subset(group, remaining):
    """同优先级内选择可整团接纳的最优组合（团体数最多，再按总人数最多）。"""
    best = []
    for r in range(len(group), 0, -1):
        feasible = []
        for combo in combinations(group, r):
            total = sum(w["participants"] for w in combo)
            if total <= remaining:
                feasible.append((total, combo))
        if feasible:
            best = list(max(feasible, key=lambda x: x[0])[1])
            break
    return best
