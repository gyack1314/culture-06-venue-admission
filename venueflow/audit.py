"""端到端自动验证：贯穿预测校准、决定、接受与候补的一致性审计。

verify 重放台账，用各事件当时已知的状态重算约束，检查：
1. 模型校准门槛（时间外覆盖率）；
2. 每笔已确认决定在确认时满足容量/公益/补偿约束（或明确记录越权）；
3. 每笔接受都不超过当时生效限额下的真实剩余；
4. 公益保留在公开截止前未被商业占用，截止后只释放未确认部分；
5. 承诺不被撤回：台账事件类型中不存在撤销语义，已确认预约始终计入；
6. 候补按提交次序整团办理，未跳过可完整接受的更早团体；
7. 每次销售边界都带有分析依据。
"""
from __future__ import annotations

from .allocation import base_limit, charity_status, check_limit, remaining_quota
from .io import build_state, parse_at
from .model import fit
from .validation import calibration_gate


class Finding:
    def __init__(self, ok, check, detail):
        self.ok, self.check, self.detail = ok, check, detail

    def to_dict(self):
        return {"ok": self.ok, "check": self.check, "detail": self.detail}


def _replay_events(ledger, session_id=None):
    evs = ledger.events(session_id=session_id)
    return sorted(evs, key=lambda e: e.get("seq", 0))


def verify(tables, ledger, cutoffs=None, session_id=None):
    findings = []

    # 1. 校准门槛
    if cutoffs:
        ok, detail = calibration_gate(tables, cutoffs)
        findings.append(Finding(ok, "calibration_gate", detail))

    # 2-4. 逐决定复核
    for ev in _replay_events(ledger, session_id):
        at = parse_at(ev["at"])
        sid = ev["session_id"]
        if ev["kind"] == "decision":
            state = _state_with_ledger_before(tables, ledger, sid, ev)
            model = fit(tables, at)
            check = check_limit(model, state, tables, ev["commercial_limit"])
            overridden = ev.get("basis", {}).get("violations_overridden", [])
            ok = check.feasible or (sorted(check.violations) == sorted(overridden)
                                    and len(overridden) > 0)
            findings.append(Finding(ok, "decision_constraints", {
                "decision_id": ev.get("decision_id"), "session_id": sid,
                "limit": ev["commercial_limit"],
                "recomputed_violations": check.violations,
                "recorded_overrides": overridden,
            }))
            # 公益：超放必须经风险与预算约束（check_limit 已含），此处单独标注
            base = base_limit(state)
            if ev["commercial_limit"] > base:
                findings.append(Finding(
                    check.feasible, "charity_oversell_justified", {
                        "decision_id": ev.get("decision_id"),
                        "limit": ev["commercial_limit"], "base": base,
                        "note": "超过基础可售的部分必须同时通过 alpha 与补偿预算",
                    }))
        elif ev["kind"] == "acceptance" and ev["reservation_type"] == "commercial":
            state = _state_with_ledger_before(tables, ledger, sid, ev)
            remaining, limit, sold, source = remaining_quota(state)
            ok = ev["participants"] <= remaining
            findings.append(Finding(ok, "acceptance_within_quota", {
                "acceptance_id": ev.get("acceptance_id"), "session_id": sid,
                "participants": ev["participants"],
                "remaining_before": remaining, "limit": limit, "source": source,
            }))
        elif ev["kind"] == "acceptance" and ev["reservation_type"] == "public_benefit":
            state = _state_with_ledger_before(tables, ledger, sid, ev)
            charity = charity_status(state)
            free = charity["protected_capacity"] - charity["confirmed_net"]
            findings.append(Finding(ev["participants"] <= free,
                                    "charity_acceptance_within_protected", {
                                        "acceptance_id": ev.get("acceptance_id"),
                                        "participants": ev["participants"],
                                        "protected_free_before": free,
                                    }))
        elif ev["kind"] == "boundary":
            has_basis = bool(ev.get("basis"))
            findings.append(Finding(has_basis, "boundary_has_basis", {
                "boundary_id": ev.get("boundary_id"),
                "boundary_kind": ev.get("boundary_kind"),
            }))

    # 5. 承诺不撤回：台账不存在撤销类事件
    kinds = {ev["kind"] for ev in ledger.events()}
    findings.append(Finding(not (kinds - {"decision", "boundary", "acceptance",
                                          "waitlist_disposition"}),
                            "no_withdrawal_semantics",
                            {"event_kinds": sorted(kinds)}))

    # 6. 候补次序：同一次办理中，被接受团体之前不存在剩余足够却被搁置的更早团体
    for sid in {ev["session_id"] for ev in ledger.events("waitlist_disposition")}:
        disps = [ev for ev in _replay_events(ledger, sid)
                 if ev["kind"] == "waitlist_disposition"]
        waits = {w["wait_id"]: w for w in tables["waitlist"]
                 if w["session_id"] == sid}
        # 一次办理产生一个 run_id 的处置记录
        runs_by_id = {}
        for d in disps:
            runs_by_id.setdefault(d.get("run_id", d["at"]), []).append(d)
        runs = list(runs_by_id.values())
        for run in runs:
            run_at = run[0]["at"]
            run.sort(key=lambda d: (waits.get(d["wait_id"], {}).get("submitted_at", ""),
                                    d["wait_id"]))
            earlier = []
            for d in run:
                w = waits.get(d["wait_id"])
                if w is None:
                    continue
                if d["action"] == "accepted":
                    for prev_d, prev_w in earlier:
                        ok = (prev_d["action"] == "accepted"
                              or prev_d.get("remaining_at_decision", 0)
                              < prev_w["participants"])
                        findings.append(Finding(ok, "waitlist_no_skip", {
                            "session_id": sid, "run_at": run_at,
                            "skipped_wait_id": prev_d["wait_id"],
                            "skipped_size": prev_w["participants"],
                            "remaining_then": prev_d.get("remaining_at_decision"),
                            "later_accepted": d["wait_id"],
                        }))
                earlier.append((d, w))

    ok_all = all(f.ok for f in findings)
    return ok_all, [f.to_dict() for f in findings]


def _state_with_ledger_before(tables, ledger, session_id, event):
    """重建事件生效前一瞬间的状态（按台账序号，不含该事件及其后事件）。"""
    at = parse_at(event["at"])
    seq = event.get("seq", 10 ** 18)
    accepted = []
    for ev in ledger.events("acceptance", session_id):
        if ev.get("seq", 0) >= seq or parse_at(ev["at"]) > at:
            continue
        accepted.append({
            "booking_id": ev["acceptance_id"],
            "session_id": ev["session_id"],
            "party_id": ev["party_id"],
            "participants": ev["participants"],
            "cancelled_count": ev.get("cancelled_count", 0),
            "revision": 1,
            "submitted_at": ev["at"],
            "reservation_type": ev["reservation_type"],
            "status": "confirmed",
            "actual_count": None,
        })
    decisions = [d for d in ledger.decisions(session_id, at)
                 if d.get("seq", 0) < seq]
    return build_state(tables, session_id, at,
                       accepted=accepted, decisions=decisions)
