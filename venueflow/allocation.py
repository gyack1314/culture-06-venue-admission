"""票额安排：公益保留与释放、有效限额、约束检查与情景评估。

约束语义（docs/业务约定.md 与 docs/方法说明.md）：
- base_limit(t)：公开截止前 = capacity - protected_capacity（公益名额不参与
  商业超放）；截止后只释放尚未确认部分 = capacity - protected + max(0,
  protected - 公益已确认净额)；
- commercial_limit 不超过 base_limit 为常规销售；超过即为超放，仅当
  P(到访 > 容量) <= alpha 且期望补偿 <= 预算时才允许确认；
- commercial_limit 不得低于已售商业净额（已确认预约不可无声撤回）；
- 情景评估只读不写；确认后的决定进入台账并约束后续接受。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .io import net_participants, parse_at
from .model import Distribution, predict_session, evidence_tier


def charity_status(state):
    """公益保留的确认、未确认与释放情况。"""
    session = state.session
    protected = session["protected_capacity"]
    confirmed = state.net_by_type("public_benefit")
    unconfirmed = max(0, protected - confirmed)
    release_at = parse_at(session["release_at"])
    released = 0
    if state.as_of is not None and state.as_of >= release_at:
        released = unconfirmed
    anomaly = None
    if confirmed > protected:
        anomaly = (f"公益已确认净额 {confirmed} 超过保留量 {protected}，"
                   "超出部分仍属承诺，容量约束按实际承诺计算")
    return {
        "protected_capacity": protected,
        "confirmed_net": confirmed,
        "unconfirmed": unconfirmed,
        "release_at": session["release_at"],
        "released": released,
        "still_locked": unconfirmed - released,
        "anomaly": anomaly,
    }


def base_limit(state):
    """不超放情形下商业可售的基础上限。"""
    charity = charity_status(state)
    cap = state.capacity - charity["protected_capacity"] + charity["released"]
    return max(0, cap)


def effective_limit(state):
    """当前生效的商业限额：最新已确认决定，否则保守的物理上限。

    返回 (limit, source)。source 为 decision_id 或 "default-physical"。
    """
    if state.decisions:
        latest = state.decisions[-1]
        return latest["commercial_limit"], latest.get("decision_id", "decision")
    return base_limit(state), "default-physical"


def remaining_quota(state):
    """实际剩余商业额度 = 生效限额 - 当前商业净售（实时重算，不用旧建议）。"""
    limit, source = effective_limit(state)
    sold = state.net_by_type("commercial")
    return max(0, limit - sold), limit, sold, source


@dataclass
class LimitCheck:
    limit: int
    feasible: bool
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    oversell: bool = False
    risk: dict = field(default_factory=dict)
    compensation: dict = field(default_factory=dict)


def commercial_size_pool(tables, as_of=None):
    """历史商业团体规模池（用于情景中未来售出的构成模拟）。"""
    from .io import matured_bookings
    sizes = [b["participants"] for b, _ in matured_bookings(tables, as_of)
             if b["reservation_type"] == "commercial" and b["participants"] > 0]
    return sizes or [1]


def check_limit(model, state, tables, limit, mode="net"):
    """评估一个候选商业限额是否满足全部约束（只读，不占名额）。"""
    sold = state.net_by_type("commercial")
    charity = charity_status(state)
    base = base_limit(state)
    check = LimitCheck(limit=limit, feasible=True)

    if limit < sold:
        check.feasible = False
        check.violations.append(
            f"限额 {limit} 低于已售商业净额 {sold}，已确认预约不可撤回")
    check.oversell = limit > base
    if check.oversell:
        check.warnings.append(
            f"限额 {limit} 超过基础可售 {base}（公益保留不参与部分），属超放，"
            "需同时通过超员概率与补偿预算约束")

    # 风险：已确认 + 假设商业售出到 limit（保守取满售）
    extra = max(0, limit - sold)
    planned_net = sold + extra + charity["confirmed_net"]
    ratio = planned_net / state.capacity if state.capacity else 0.0
    tier, tier_notes = evidence_tier(model, state.session, planned_net_ratio=ratio)
    values, probs, notes = predict_session(
        model, state, extra_commercial=extra, mode=mode,
        commercial_sizes=commercial_size_pool(tables, state.as_of))
    dist = Distribution(values=values, probs=probs, tier=tier, notes=tier_notes + notes)

    cap = state.capacity
    alpha = state.session["alpha"]
    p_over = dist.prob_over(cap)
    exp_over = dist.expected_overage(cap)
    per_person = state.session["compensation_per_person"]
    budget = state.session["compensation_budget"]
    exp_comp = exp_over * per_person

    check.risk = dist.summary(cap)
    check.risk["alpha"] = alpha
    check.risk["notes"] = dist.notes
    check.compensation = {
        "expected": round(exp_comp, 2),
        "budget": budget,
        "per_person": per_person,
        "p95_overage_persons": max(0, dist.quantile(0.95) - cap),
        "headroom": round(budget - exp_comp, 2),
    }
    if p_over > alpha:
        check.feasible = False
        check.violations.append(
            f"满售该限额时 P(到访>{cap}) = {p_over:.4f} 超过容忍度 alpha={alpha}")
    if exp_comp > budget:
        check.feasible = False
        check.violations.append(
            f"期望补偿 {exp_comp:.0f} 超过预算 {budget}")
    if tier != "calibrated":
        check.warnings.append("证据等级为 " + tier + "：概率结论仅供参考，"
                              "决策应同时查看 markov/cantelli 界限")
    return check


def evaluate_scenarios(model, state, tables, limits):
    """比较若干候选限额：收入代理、满载风险与公益覆盖（只读）。"""
    charity = charity_status(state)
    sold = state.net_by_type("commercial")
    results = []
    for limit in limits:
        check = check_limit(model, state, tables, limit)
        extra = max(0, limit - sold)
        results.append({
            "commercial_limit": limit,
            "feasible": check.feasible,
            "violations": check.violations,
            "warnings": check.warnings,
            "oversell": check.oversell,
            "revenue_proxy": {
                "note": "资料无价格字段，以商业净售名额为收入代理",
                "commercial_seats_sold": sold + extra,
            },
            "expected_attendance": check.risk["mean"],
            "p_over_capacity": check.risk["p_over_capacity"],
            "expected_compensation": check.compensation["expected"],
            "charity_coverage": {
                "confirmed_net": charity["confirmed_net"],
                "protected": charity["protected_capacity"],
                "ratio": round(charity["confirmed_net"] / charity["protected_capacity"], 4)
                if charity["protected_capacity"] else None,
                "released_to_commercial": charity["released"],
            },
            "risk": check.risk,
            "compensation": check.compensation,
        })
    return results


def max_feasible_limit(model, state, tables, ceiling=None):
    """在约束内可确认的最大商业限额（二分搜索，只读）。"""
    sold = state.net_by_type("commercial")
    hi = ceiling if ceiling is not None else state.capacity * 2
    lo = sold
    if not check_limit(model, state, tables, lo).feasible:
        return None  # 连现状都已违反约束（新数据抬高风险），由调用方处置
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if check_limit(model, state, tables, mid).feasible:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
