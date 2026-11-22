"""到访分布模型。

核心事实（由拟合数据支持，validate 命令输出检验明细）：
- 公益团体：历史上 actual_count 恒等于净预约人数 → 按净额确定到场建模；
- 商业团体：同一团体的成员不是相互独立的硬币——逐人独立伯努利会
  严重高估团体出席的方差，因此按"团体规模 → 出席均值/减员分布"建模；
- 场次层面：历史场次的总到访围绕"各预约条件均值之和"的残差远小于
  独立卷积隐含的离散度（sd≈1.1 vs ≈2.6），即同场预约间存在负相关
  （合成资料的实际结构）。因此对已知构成族（每场预约数不超过历史
  最大值、各 (类型,规模) 单元证据充足）采用"均值和 + 场次残差"模型；
  对陌生构成（更多预约、未见过的规模）退回逐预约独立卷积——该方向
  更宽、对超员风险更保守，并在输出中标注外推；
- 同团体跨预约相关未检出（ICC≈0），不额外加团体层随机效应；
- 当前净预约是未来净预约的上界（取消只会减少），net 模式的超员风险
  为保守方向；gross 模式不依赖取消时点，用于截点验证，避免把截点后
  才发生的取消当作已知。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import TIER_CALIBRATED, TIER_EXTRAPOLATED, TIER_UNSUPPORTED
from .io import matured_bookings, net_participants, resolve_bookings

SMOOTHING = 0.25          # 减员计数分布的 Dirichlet 平滑
MIN_SIZE_OBS = 20         # 某规模单独建模所需的最少成熟预约数
MIN_CELL_OBS = 15         # (类型,规模) 均值单元所需的最少成熟预约数
MIN_FIT_BOOKINGS = 60     # 模型可校准的最少成熟预约数
MIN_VENUE_SESSIONS = 10   # 场馆级证据所需的最少成熟场次


def _counts_to_pmf(counts, support):
    total = sum(counts.get(d, 0) for d in support) + SMOOTHING * len(support)
    return np.array([(counts.get(d, 0) + SMOOTHING) / total for d in support])


@dataclass
class TypeModel:
    """单一预约类型的逐预约减员分布（陌生构成的保守退路 / 未来售出模拟用）。"""

    mode: str
    deterministic_net: bool = False
    by_size: dict = field(default_factory=dict)   # size -> (support, pmf)
    pooled: tuple = None                          # (support, pmf)
    size_counts: dict = field(default_factory=dict)
    n_obs: int = 0

    def show_pmf(self, size):
        """给定规模，返回 (出席人数值数组, 概率数组)。"""
        if self.deterministic_net:
            return np.array([size]), np.array([1.0])
        if size <= 0:
            return np.array([0]), np.array([1.0])
        entry = self.by_size.get(size, self.pooled)
        if entry is None:
            # 无证据规模：保守退化为全到（上界方向）。
            return np.array([size]), np.array([1.0])
        support, pmf = entry
        values = np.array([max(0, size - d) for d in support])
        merged = {}
        for v, p in zip(values, pmf):
            merged[v] = merged.get(v, 0.0) + p
        vals = sorted(merged)
        return np.array(vals), np.array([merged[v] for v in vals])


@dataclass
class SessionLevelModel:
    """场次总量模型：Σ 单元均值 + 场次残差（已知构成族）。"""

    means: dict = field(default_factory=dict)        # (rtype, size) -> mean
    cell_counts: dict = field(default_factory=dict)  # (rtype, size) -> n
    residual_values: np.ndarray = None
    residual_probs: np.ndarray = None
    max_composition: int = 0                          # 拟合中每场最大预约数

    def covers(self, n_bookings):
        """构成族覆盖仅约束每场预约数；单元证据不足由均值层面降级处理。"""
        return n_bookings <= self.max_composition


@dataclass
class AttendanceModel:
    """按预约类型分组的出席模型，附拟合证据。"""

    as_of: object
    types: dict                       # reservation_type -> TypeModel (net 模式)
    session_models: dict              # mode -> SessionLevelModel
    n_matured_bookings: int
    n_matured_sessions: int
    venue_sessions: dict              # venue_id -> 成熟场次数
    max_net_ratio: float              # 历史净预约/容量最大值（外推阈值）
    warnings: list = field(default_factory=list)

    def type_model(self, rtype, mode):
        if mode == "net":
            return self.types[rtype]
        return self.types[(rtype, mode)]

    def cell_mean(self, rtype, size, mode):
        """(类型,规模) 单元的条件均值；证据不足时降级为合并证据均值。

        返回 (均值, 是否为校准单元)。
        """
        sm = self.session_models[mode]
        if sm.cell_counts.get((rtype, size), 0) >= MIN_CELL_OBS:
            return sm.means[(rtype, size)], True
        v, p = self.type_model(rtype, mode).show_pmf(size)
        return float(v @ p), False


def _fit_type(pairs, mode, deterministic_check=False):
    """pairs: (size, actual) 列表 → 逐预约减员分布族。"""
    attrition = {}
    for size, actual in pairs:
        d = size - actual
        if d < 0:
            raise ValueError("actual_count 超过可用人数，资料异常")
        attrition.setdefault(size, {})
        attrition[size][d] = attrition[size].get(d, 0) + 1
    tm = TypeModel(mode=mode)
    tm.n_obs = len(pairs)
    tm.size_counts = {s: sum(c.values()) for s, c in attrition.items()}
    if deterministic_check and all(d == 0 for c in attrition.values() for d in c):
        tm.deterministic_net = True
        return tm
    pooled_counts = {}
    for counts in attrition.values():
        for d, n in counts.items():
            pooled_counts[d] = pooled_counts.get(d, 0) + n
    if pooled_counts:
        # 支撑集取观测到的减员范围：不把概率摊到从未出现的取值上，
        # 无模型情形下由 Markov/Cantelli 界限兜底。
        lo, hi = min(pooled_counts), max(pooled_counts)
        tm.pooled = (list(range(lo, hi + 1)),
                     _counts_to_pmf(pooled_counts, range(lo, hi + 1)))
    for size, counts in attrition.items():
        if sum(counts.values()) >= MIN_SIZE_OBS:
            lo, hi = min(counts), max(counts)
            support = list(range(lo, hi + 1))
            tm.by_size[size] = (support, _counts_to_pmf(counts, support))
    return tm


def _fit_session_level(rows, size_of):
    """rows: (booking, session) 列表。拟合单元均值与场次残差分布。"""
    by_cell = {}
    for b, _ in rows:
        key = (b["reservation_type"], size_of(b))
        by_cell.setdefault(key, []).append(b["actual_count"])
    means = {k: float(np.mean(v)) for k, v in by_cell.items()}
    counts = {k: len(v) for k, v in by_cell.items()}
    per_session = {}
    comp = 0
    for b, session in rows:
        per_session.setdefault(b["session_id"], []).append(b)
    residuals = {}
    for sid, bs in per_session.items():
        comp = max(comp, len(bs))
        pred = sum(means[(b["reservation_type"], size_of(b))] for b in bs)
        actual = sum(b["actual_count"] for b in bs)
        r = round(actual - pred, 6)
        residuals[r] = residuals.get(r, 0) + 1
    sm = SessionLevelModel(means=means, cell_counts=counts,
                           max_composition=comp)
    if residuals:
        support = sorted(residuals)
        # 残差支撑取观测范围；场次残差近似整数，按原值保留
        sm.residual_values = np.array(support, dtype=float)
        sm.residual_probs = _counts_to_pmf(residuals, support)
    else:
        sm.residual_values = np.array([0.0])
        sm.residual_probs = np.array([1.0])
    return sm


def fit(tables, as_of=None):
    """用截至 as_of 已成熟（已闭场）的预约拟合。"""
    rows = list(matured_bookings(tables, as_of))
    pairs_net = {}
    pairs_gross = {}
    sessions_seen = {}
    venue_sessions = {}
    net_by_session = {}
    caps = {}
    for b, session in rows:
        rtype = b["reservation_type"]
        pairs_net.setdefault(rtype, []).append((net_participants(b), b["actual_count"]))
        pairs_gross.setdefault(rtype, []).append((b["participants"], b["actual_count"]))
        sessions_seen[b["session_id"]] = session
        venue_sessions.setdefault(session["venue_id"], set()).add(b["session_id"])
        caps[b["session_id"]] = session["capacity"]
    # 历史净预约强度（含最终取消后的净额），用于外推判定
    for b in resolve_bookings(tables["bookings"], as_of):
        sid = b["session_id"]
        if sid in sessions_seen:
            net_by_session[sid] = net_by_session.get(sid, 0) + net_participants(b)
    max_ratio = 0.0
    for sid, net in net_by_session.items():
        if caps.get(sid):
            max_ratio = max(max_ratio, net / caps[sid])

    types = {}
    warnings = []
    for rtype in sorted(set(list(pairs_net) + ["commercial", "public_benefit"])):
        net_pairs = pairs_net.get(rtype, [])
        gross_pairs = pairs_gross.get(rtype, [])
        if not net_pairs:
            warnings.append(f"类型 {rtype} 无成熟证据，相关出席按全到上界处理")
            types[rtype] = TypeModel(mode="net")
            types[(rtype, "gross")] = TypeModel(mode="gross")
            continue
        types[rtype] = _fit_type(net_pairs, "net",
                                 deterministic_check=(rtype == "public_benefit"))
        types[(rtype, "gross")] = _fit_type(gross_pairs, "gross")

    session_models = {
        "net": _fit_session_level(rows, net_participants),
        "gross": _fit_session_level(rows, lambda b: b["participants"]),
    }
    return AttendanceModel(
        as_of=as_of,
        types=types,
        session_models=session_models,
        n_matured_bookings=len(rows),
        n_matured_sessions=len(sessions_seen),
        venue_sessions={k: len(v) for k, v in venue_sessions.items()},
        max_net_ratio=max_ratio,
        warnings=warnings,
    )


def convolve(pmfs):
    """若干 (values, probs) 的精确卷积，返回 (values, probs)。"""
    values = np.array([0.0])
    probs = np.array([1.0])
    for v, p in pmfs:
        new = {}
        for sv, sp in zip(values, probs):
            for vv, vp in zip(v, p):
                key = round(float(sv + vv), 6)
                new[key] = new.get(key, 0.0) + float(sp * vp)
        values = np.array(sorted(new))
        probs = np.array([new[k] for k in values])
    return values, probs


@dataclass
class Distribution:
    """一场的到访人数分布及无分布界限。"""

    values: np.ndarray
    probs: np.ndarray
    tier: str
    notes: list = field(default_factory=list)

    @property
    def mean(self):
        return float(self.values @ self.probs)

    @property
    def var(self):
        m = self.mean
        return float(((self.values - m) ** 2) @ self.probs)

    def prob_over(self, threshold):
        """P(T > threshold)。"""
        return float(self.probs[self.values > threshold].sum())

    def expected_overage(self, capacity):
        over = np.maximum(0.0, self.values - capacity)
        return float(over @ self.probs)

    def quantile(self, q):
        cdf = np.cumsum(self.probs)
        idx = int(np.searchsorted(cdf, q, side="left"))
        return float(self.values[min(idx, len(self.values) - 1)])

    def interval(self, level):
        lo = (1 - level) / 2
        return self.quantile(lo), self.quantile(1 - lo)

    def markov_bound(self, threshold):
        """P(T >= threshold) <= E[T]/threshold，无需任何校准假设。"""
        if threshold <= 0:
            return 1.0
        return min(1.0, self.mean / threshold)

    def cantelli_bound(self, threshold):
        """P(T - E[T] >= threshold - E[T]) 的单侧切比雪夫界限。"""
        m, v = self.mean, self.var
        d = threshold - m
        if d <= 0:
            return 1.0
        if v <= 0:
            return 0.0
        return min(1.0, v / (v + d * d))

    def summary(self, capacity):
        return {
            "mean": round(self.mean, 3),
            "sd": round(self.var ** 0.5, 3),
            "p50": round(self.quantile(0.5), 2),
            "p90": round(self.quantile(0.9), 2),
            "p95": round(self.quantile(0.95), 2),
            "p_over_capacity": round(self.prob_over(capacity), 6),
            "expected_overage": round(self.expected_overage(capacity), 4),
            "markov_bound_over": round(self.markov_bound(capacity + 1), 6),
            "cantelli_bound_over": round(self.cantelli_bound(capacity + 1), 6),
            "tier": self.tier,
        }


def _size_of(booking, mode):
    return (net_participants(booking) if mode == "net"
            else booking["participants"])


def predict_session(model, state, extra_commercial=0, mode="net", seed=20260601,
                    n_mc=20000, commercial_sizes=None):
    """单场到访分布。

    已知构成族：Σ 单元均值 + 场次残差（校准模型）。
    陌生构成：逐预约独立卷积（更宽、对超员更保守）并标注。
    extra_commercial > 0 时（情景评估）：未来售出按历史商业团体规模
    分布拆成团体，固定种子蒙特卡洛并入。
    """
    bookings = state.confirmed_bookings()
    notes = []
    sm = model.session_models[mode]
    if bookings and sm.covers(len(bookings)):
        total_mean = 0.0
        weak = set()
        for b in bookings:
            m, calibrated = model.cell_mean(b["reservation_type"],
                                            _size_of(b, mode), mode)
            total_mean += m
            if not calibrated:
                weak.add((b["reservation_type"], _size_of(b, mode)))
        if weak:
            notes.append(f"单元 {sorted(weak)} 成熟证据不足 "
                         f"{MIN_CELL_OBS} 条，均值使用合并证据")
        values = sm.residual_values + total_mean
        probs = sm.residual_probs
    else:
        if bookings:
            notes.append("构成超出已校准族（每场预约数超过历史最大值），"
                         "退回逐预约独立卷积（更宽、保守方向）")
        pmfs = [model.type_model(b["reservation_type"], mode)
                .show_pmf(_size_of(b, mode)) for b in bookings]
        values, probs = convolve(pmfs) if pmfs else (np.array([0.0]),
                                                     np.array([1.0]))
    if extra_commercial > 0:
        sizes = commercial_sizes
        if not sizes:
            raise ValueError("缺少历史商业团体规模分布，无法评估未来售出")
        rng = np.random.default_rng(seed)
        tm = model.type_model("commercial", mode)
        draws = _simulate_future_sales(rng, tm, sizes, extra_commercial, n_mc)
        max_future = int(draws.max())
        future_probs = np.bincount(draws.astype(int),
                                   minlength=max_future + 1) / n_mc
        values, probs = convolve([
            (values, probs),
            (np.arange(max_future + 1, dtype=float), future_probs),
        ])
        notes.append(f"未来售出 {extra_commercial} 人按历史团体规模蒙特卡洛"
                     f"({n_mc})并入，新到团体按独立噪声处理")
    return values, probs, notes


def _simulate_future_sales(rng, type_model, sizes, extra, n_mc):
    """向量化模拟：把 extra 个未来席位按历史团体规模拆团并求出席和。

    团体构成：从历史规模池有放回抽样，最后一个团体按剩余席位截断；
    每个团体的出席数按其规模的分布抽样（团内相关由该分布承载）。
    """
    sizes_arr = np.asarray(sizes, dtype=int)
    min_size = int(sizes_arr.min())
    n_cols = extra // min_size + 2
    gs = rng.choice(sizes_arr, size=(n_mc, n_cols))
    cs = np.cumsum(gs, axis=1)
    cut = (cs >= extra).argmax(axis=1)
    rows = np.arange(n_mc)
    overshoot = cs[rows, cut] - extra
    gs[rows, cut] -= overshoot
    gs[np.arange(n_cols) > cut[:, None]] = 0
    # 每种规模一张逆 CDF 查值表，把均匀随机数映射为出席数
    uniq = np.unique(gs)
    lut = {}
    for s in uniq:
        v, p = type_model.show_pmf(int(s))
        cdf = np.cumsum(p)
        lut[int(s)] = np.array([v[min(int(np.searchsorted(cdf, u)), len(v) - 1)]
                                for u in (np.arange(1024) + 0.5) / 1024])
    size_to_row = {int(s): i for i, s in enumerate(uniq)}
    lut_mat = np.array([lut[int(s)] for s in uniq])
    u = rng.integers(0, 1024, size=gs.shape)
    row_idx = np.searchsorted(uniq, gs)   # uniq 有序且 gs 取值必在其中
    shows = lut_mat[row_idx, u]
    return shows.sum(axis=1)


def evidence_tier(model, session, planned_net_ratio=None):
    """判断该场次结论的证据等级。

    - 成熟证据不足 → unsupported（只给界限）；
    - 该场馆成熟场次不足 → extrapolated（pooled 证据，降级使用）；
    - 计划净预约强度超过历史最大值 → 尾部 extrapolated。
    """
    if model.n_matured_bookings < MIN_FIT_BOOKINGS:
        return TIER_UNSUPPORTED, ["成熟预约证据不足，概率结论不可用，仅提供界限"]
    notes = []
    tier = TIER_CALIBRATED
    venue = session["venue_id"]
    if model.venue_sessions.get(venue, 0) < MIN_VENUE_SESSIONS:
        tier = TIER_EXTRAPOLATED
        notes.append(f"场馆 {venue} 成熟场次 {model.venue_sessions.get(venue, 0)} "
                     f"< {MIN_VENUE_SESSIONS}，使用跨场馆 pooled 证据，结论降级")
    if planned_net_ratio is not None and planned_net_ratio > model.max_net_ratio:
        tier = TIER_EXTRAPOLATED
        notes.append(f"计划净预约强度 {planned_net_ratio:.2f} 超过历史最大 "
                     f"{model.max_net_ratio:.2f}，容量附近行为无直接证据，尾部为外推")
    return tier, notes
