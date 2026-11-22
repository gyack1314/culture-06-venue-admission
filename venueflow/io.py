"""资料加载、版本消解与"截至某时刻已知"状态重建。

约定（见 docs/业务约定.md）：
- 同 booking_id 的多 revision 是完整替代记录，人数不相加；
- 未闭场场次的 actual_count 缺失不补 0，也不进入拟合标签；
- 截点重建只使用该截点已获知的事实（按各自 submitted_at / at / start_at）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import json

from tools.inspect_inputs import inspect as inspect_inputs

TABLES = (
    "sessions", "bookings", "waitlist", "sales_controls",
    "enquiries", "policies", "scenarios", "decisions", "observation_cutoffs",
)


def parse_at(text):
    """带时区时间戳。"""
    return datetime.fromisoformat(text)


def parse_day(text):
    return date.fromisoformat(text)


def load_tables(root, verify=True):
    """加载全部表；默认先核对 catalog 校验和、行数与字段。"""
    root = Path(root)
    if verify:
        inspect_inputs(root)
    tables = {}
    for name in TABLES:
        path = root / "data" / (name + ".jsonl")
        tables[name] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return tables


def resolve_bookings(rows, as_of=None):
    """每个 booking_id 取截至 as_of 已提交的最高 revision（完整替代）。

    as_of 为 None 时不过滤。返回按 submitted_at 排序的列表。
    """
    latest = {}
    for row in rows:
        if as_of is not None and parse_at(row["submitted_at"]) > as_of:
            continue
        key = row["booking_id"]
        if key not in latest or row["revision"] > latest[key]["revision"]:
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (r["submitted_at"], r["booking_id"]))


def net_participants(booking):
    """当前净预约人数（预约减去累计取消）。"""
    return booking["participants"] - booking["cancelled_count"]


def session_matured(session, as_of):
    """截至 as_of 该场次是否已闭场（actual_count 才有效）。

    以接待日整天为准：as_of 日期严格晚于场次日期才算已成熟。
    """
    if as_of is None:
        return session["status"] == "closed"
    return parse_day(session["date"]) < as_of.date()


@dataclass
class SessionState:
    """单场在某一截点的完整已知状态。"""

    session: dict
    as_of: datetime | None
    bookings: list = field(default_factory=list)      # 已消解 revision
    controls: list = field(default_factory=list)
    enquiries: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    waitlist: list = field(default_factory=list)
    accepted: list = field(default_factory=list)      # 台账中已接受的新预约

    @property
    def session_id(self):
        return self.session["session_id"]

    @property
    def capacity(self):
        return self.session["capacity"]

    def confirmed_bookings(self):
        """已确认预约（原始资料 + 台账接受），承诺不可无声撤回。"""
        return [b for b in self.bookings if b["status"] == "confirmed"] + list(self.accepted)

    def by_type(self, reservation_type):
        return [b for b in self.confirmed_bookings()
                if b["reservation_type"] == reservation_type]

    def net_by_type(self, reservation_type):
        return sum(net_participants(b) for b in self.by_type(reservation_type))

    def matured(self):
        return session_matured(self.session, self.as_of)


def _known(rows, time_field, as_of):
    if as_of is None:
        return list(rows)
    return [r for r in rows if parse_at(r[time_field]) <= as_of]


def build_state(tables, session_id, as_of=None, accepted=(), decisions=()):
    """组装单场截点状态。decisions 含原始决定表与台账决定（调用方合并）。"""
    session = next(s for s in tables["sessions"] if s["session_id"] == session_id)
    bookings = [b for b in resolve_bookings(tables["bookings"], as_of)
                if b["session_id"] == session_id]
    raw_decisions = [d for d in tables["decisions"] if d["session_id"] == session_id]
    if as_of is not None:
        raw_decisions = [d for d in raw_decisions if parse_at(d["at"]) <= as_of]
    controls = [c for c in tables["sales_controls"] if c["session_id"] == session_id]
    if as_of is not None:
        controls = [c for c in controls if parse_at(c["start_at"]) <= as_of]
    return SessionState(
        session=session,
        as_of=as_of,
        bookings=bookings,
        controls=controls,
        enquiries=[e for e in _known(tables["enquiries"], "at", as_of)
                   if e["session_id"] == session_id],
        decisions=sorted(raw_decisions + list(decisions),
                         key=lambda d: (d["at"], d.get("decision_id", ""))),
        waitlist=[w for w in _known(tables["waitlist"], "submitted_at", as_of)
                  if w["session_id"] == session_id],
        accepted=[a for a in accepted if a["session_id"] == session_id],
    )


def matured_bookings(tables, as_of=None):
    """已闭场场次、actual_count 有效的已消解预约（拟合标签）。

    未成熟场次一律排除；actual_count 为 null 的记录不补 0、不进入。
    """
    sessions = {s["session_id"]: s for s in tables["sessions"]}
    out = []
    for b in resolve_bookings(tables["bookings"], as_of):
        session = sessions[b["session_id"]]
        if not session_matured(session, as_of):
            continue
        if b["actual_count"] is None:
            continue
        out.append((b, session))
    return out


def default_now(tables):
    """项目时钟：缺省取最晚观察截点，避免把未来信息当作已知。"""
    cutoffs = [parse_at(c["at"]) for c in tables["observation_cutoffs"]]
    return max(cutoffs) if cutoffs else None
