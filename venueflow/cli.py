"""操作入口：分析、票额安排、确认、接受、候补、查询与审计。

用法示例：
    python -m venueflow.cli validate
    python -m venueflow.cli risk S140
    python -m venueflow.cli plan S140 --limits 45 65
    python -m venueflow.cli confirm S140 --limit 45 --note "春季方案"
    python -m venueflow.cli accept S140 --party G9 --participants 6
    python -m venueflow.cli waitlist S140
    python -m venueflow.cli status S140
    python -m venueflow.cli verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import load_tables, default_now, parse_at
from .ledger import Ledger


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False))


def _now(tables, args):
    if getattr(args, "at", None):
        return parse_at(args.at)
    if getattr(args, "as_of", None):
        return parse_at(args.as_of)
    return default_now(tables)


def cmd_validate(tables, ledger, args):
    from .validation import (out_of_time, summarize_quality, transferability,
                             calibration_gate)
    cutoffs = tables["observation_cutoffs"]
    rows = out_of_time(tables, cutoffs)
    ok, gate = calibration_gate(tables, cutoffs)
    _print({
        "calibration_gate_pass": ok,
        "gate_detail": gate,
        "out_of_time_quality": summarize_quality(rows),
        "n_out_of_time_sessions": len(rows),
        "transferability": transferability(tables, _now(tables, args)),
    })
    return 0 if ok else 1


def cmd_risk(tables, ledger, args):
    from .report import session_status
    status = session_status(tables, ledger, args.session, _now(tables, args))
    _print({k: status[k] for k in
            ("session_id", "as_of", "capacity", "matured", "risk",
             "risk_notes", "model_evidence")})
    return 0


def cmd_plan(tables, ledger, args):
    from .allocation import evaluate_scenarios, max_feasible_limit
    from .io import build_state
    from .model import fit
    as_of = _now(tables, args)
    state = build_state(tables, args.session, as_of,
                        accepted=ledger.accepted_bookings(args.session, as_of),
                        decisions=ledger.decisions(args.session, as_of))
    model = fit(tables, as_of)
    limits = args.limits
    if not limits:
        limits = [s["commercial_limit"] for s in tables["scenarios"]
                  if s["session_id"] == args.session]
    if not limits:
        limits = [state.net_by_type("commercial")]
    results = evaluate_scenarios(model, state, tables, limits)
    _print({
        "session_id": args.session, "as_of": as_of.isoformat(),
        "note": "情景比较只读，不占名额；确认请使用 confirm",
        "scenarios": results,
        "max_feasible_limit": max_feasible_limit(model, state, tables),
    })
    return 0


def cmd_confirm(tables, ledger, args):
    from .ops import confirm_limit
    event = confirm_limit(tables, ledger, args.session, args.limit,
                          _now(tables, args), note=args.note,
                          allow_infeasible=args.allow_infeasible)
    _print(event)
    return 0


def cmd_accept(tables, ledger, args):
    from .ops import accept_booking
    event = accept_booking(tables, ledger, args.session, args.party,
                           args.participants, args.type, _now(tables, args))
    _print(event)
    return 0


def cmd_waitlist(tables, ledger, args):
    from .ops import process_waitlist
    results = process_waitlist(tables, ledger, args.session, _now(tables, args))
    _print({"session_id": args.session, "dispositions": results})
    return 0


def cmd_boundary(tables, ledger, args):
    from .ops import record_boundary
    event = record_boundary(tables, ledger, args.session, _now(tables, args),
                            args.kind, args.remaining, args.reason)
    _print(event)
    return 0


def cmd_status(tables, ledger, args):
    from .report import session_status
    _print(session_status(tables, ledger, args.session, _now(tables, args)))
    return 0


def cmd_verify(tables, ledger, args):
    from .audit import verify
    ok, findings = verify(tables, ledger,
                          cutoffs=tables["observation_cutoffs"],
                          session_id=getattr(args, "session", None))
    _print({"ok": ok, "findings": findings})
    return 0 if ok else 1


def build_parser():
    parser = argparse.ArgumentParser(prog="venueflow",
                                     description="预约到访风险与票额联合管理")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="资料根目录（含 data/）")
    parser.add_argument("--var", type=Path, default=None,
                        help="台账目录（默认 <root>/var）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_time(p):
        p.add_argument("--as-of", default=None, help="截点（ISO 时间），缺省取最晚观察截点")

    p = sub.add_parser("validate", help="时间外验证与迁移性检验")
    add_time(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("risk", help="单场到访分布与界限")
    p.add_argument("session")
    add_time(p)
    p.set_defaults(func=cmd_risk)

    p = sub.add_parser("plan", help="情景比较（只读，不占名额）")
    p.add_argument("session")
    p.add_argument("--limits", type=int, nargs="*", default=None)
    add_time(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("confirm", help="确认商业限额（写入台账，约束后续接受）")
    p.add_argument("session")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--note", default="")
    p.add_argument("--at", default=None)
    p.add_argument("--allow-infeasible", action="store_true",
                   help="强制确认并记录被覆盖的违约项（审计将复核）")
    p.set_defaults(func=cmd_confirm)

    p = sub.add_parser("accept", help="接受新预约（占用真实剩余额度）")
    p.add_argument("session")
    p.add_argument("--party", required=True)
    p.add_argument("--participants", type=int, required=True)
    p.add_argument("--type", default="commercial",
                   choices=["commercial", "public_benefit"])
    p.add_argument("--at", default=None)
    p.set_defaults(func=cmd_accept)

    p = sub.add_parser("waitlist", help="办理候补（整团、按序、实时剩余）")
    p.add_argument("session")
    p.add_argument("--at", default=None)
    p.set_defaults(func=cmd_waitlist)

    p = sub.add_parser("boundary", help="记录销售边界（停售/重开/调额建议）")
    p.add_argument("session")
    p.add_argument("--kind", required=True,
                   choices=["stop", "reopen", "limit_advice"])
    p.add_argument("--remaining", type=int, required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--at", default=None)
    p.set_defaults(func=cmd_boundary)

    p = sub.add_parser("status", help="单场综合查询")
    p.add_argument("session")
    add_time(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("verify", help="端到端一致性审计")
    p.add_argument("--session", default=None)
    add_time(p)
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tables = load_tables(args.root)
    var = args.var if args.var is not None else Path(args.root) / "var"
    ledger = Ledger(var)
    try:
        return args.func(tables, ledger, args)
    except (ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
