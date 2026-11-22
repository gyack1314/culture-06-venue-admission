"""运营台账：决定、销售边界、接受与候补处置的追加式持久化。

原始资料（data/）不可变；一切运营状态变化只追加到本台账。
每次销售边界（限额确认、停售、重开）都记录当时的分析依据，
供审计命令 verify 复核"预测 → 决定 → 接受"链条。
"""
from __future__ import annotations

from pathlib import Path
import json

EVENT_KINDS = ("decision", "boundary", "acceptance", "waitlist_disposition")


class Ledger:
    def __init__(self, var_dir):
        self.var_dir = Path(var_dir)
        self.path = self.var_dir / "ledger.jsonl"

    def append(self, event):
        self.var_dir.mkdir(parents=True, exist_ok=True)
        kind = event.get("kind")
        if kind not in EVENT_KINDS:
            raise ValueError(f"未知台账事件类型: {kind}")
        # 单调序号：同一时刻戳的多个事件仍可按写入先后重放
        event["seq"] = len(self.events()) + 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def events(self, kind=None, session_id=None):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if kind and ev.get("kind") != kind:
                continue
            if session_id and ev.get("session_id") != session_id:
                continue
            out.append(ev)
        return out

    def decisions(self, session_id=None, as_of=None):
        from .io import parse_at
        evs = self.events("decision", session_id)
        if as_of is not None:
            evs = [e for e in evs if parse_at(e["at"]) <= as_of]
        return sorted(evs, key=lambda e: (e["at"], e.get("decision_id", "")))

    def accepted_bookings(self, session_id=None, as_of=None):
        """台账中已接受的新预约，形态与原始 booking 对齐（revision=1）。"""
        from .io import parse_at
        out = []
        for ev in self.events("acceptance", session_id):
            if as_of is not None and parse_at(ev["at"]) > as_of:
                continue
            out.append({
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
        return out

    def waitlist_dispositions(self, session_id=None):
        """每个 wait_id 的最新处置。"""
        latest = {}
        for ev in self.events("waitlist_disposition", session_id):
            latest[ev["wait_id"]] = ev
        return latest

    def boundaries(self, session_id=None):
        return self.events("boundary", session_id)
