"""预约到访风险与公益场次票额联合管理。

原始资料只读（data/，受 catalog 校验和保护）；所有运营动作追加写入
var/ledger.jsonl 台账。分析与票额安排共用同一状态重建逻辑。
"""

__version__ = "0.1.0"

# 模型证据分级
TIER_CALIBRATED = "calibrated"        # 可比证据充分，概率结论可用
TIER_EXTRAPOLATED = "extrapolated"    # 部分外推，概率仅供参考并附界限
TIER_UNSUPPORTED = "unsupported"      # 无可比依据，只给界限，不给概率结论
