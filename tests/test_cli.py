"""CLI 冒烟测试：入口、退出码与台账隔离。"""
import json
from pathlib import Path

import pytest

from venueflow.cli import main

ROOT = Path(__file__).resolve().parents[1]


def run_cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip().startswith("{") else out


def test_status_and_risk(capsys, tmp_path):
    code, out = run_cli(capsys, "--var", str(tmp_path), "status", "S140")
    assert code == 0
    assert out["session_id"] == "S140"
    assert out["charity"]["protected_capacity"] == 20
    code, out = run_cli(capsys, "--var", str(tmp_path), "risk", "S140")
    assert code == 0
    assert out["risk"]["tier"] in ("calibrated", "extrapolated", "unsupported")


def test_plan_is_readonly(capsys, tmp_path):
    code, out = run_cli(capsys, "--var", str(tmp_path), "plan", "S140",
                        "--limits", "45", "65")
    assert code == 0
    assert len(out["scenarios"]) == 2
    assert not (tmp_path / "ledger.jsonl").exists()   # 情景比较不写台账


def test_confirm_accept_verify_flow(capsys, tmp_path):
    code, _ = run_cli(capsys, "--var", str(tmp_path), "confirm", "S140",
                      "--limit", "45")
    assert code == 0
    code, _ = run_cli(capsys, "--var", str(tmp_path), "waitlist", "S140")
    assert code == 0
    code, out = run_cli(capsys, "--var", str(tmp_path), "verify")
    assert code == 0 and out["ok"]


def test_reject_exit_code(capsys, tmp_path):
    code, _ = run_cli(capsys, "--var", str(tmp_path), "confirm", "S140",
                      "--limit", "45")
    assert code == 0
    # 剩余 30，申请 40 人被拒，退出码 2
    code = main(["--var", str(tmp_path), "accept", "S140", "--party", "GX",
                 "--participants", "40"])
    assert code == 2
    capsys.readouterr()


def test_validate_exit_code(capsys, tmp_path):
    code, out = run_cli(capsys, "--var", str(tmp_path), "validate")
    assert code == 0
    assert out["calibration_gate_pass"] is True
