# 预约到访风险与公益场次票额联合管理

本项目保存本地合成业务资料、输入约定和资料完整性检查，并提供从
到访风险分析到票额安排、预约接受、候补办理与审计的完整操作入口。
记录不对应真实个人、企业、场馆或合同；原始数据不带分析结论。

## 环境与现有入口

使用 Python 3.11。安装固定依赖：

```sh
python -m pip install -r requirements.txt
python tools/inspect_inputs.py
python -m pytest -q
```

`tools/inspect_inputs.py` 与 `tests/test_inputs.py` 只检查本地资料的
校验和、字段和主键，不计算业务结果。业务含义见
[业务约定](docs/业务约定.md)，文件与字段见
[资料目录](docs/资料目录.md)，模型与口径见
[方法说明](docs/方法说明.md)。data 中各表使用 UTF-8 JSON Lines，
catalog.json 记载行数、主键和校验和。全部功能不需要网络和账号。

## 业务系统（venueflow 包）

原始资料只读；一切运营动作追加写入台账 `var/ledger.jsonl`（不入库）。
项目时钟缺省取最晚观察截点，可用 `--as-of` / `--at` 覆盖。

```sh
# 分析：时间外验证、分提前期质量、迁移性检验、校准门槛
python -m venueflow.cli validate

# 单场风险：到访分布、分位数、超容概率、无分布界限、证据等级
python -m venueflow.cli risk S140

# 情景比较（只读，不占名额）：收入代理、满载风险、公益覆盖
python -m venueflow.cli plan S140 --limits 45 65

# 确认限额（写入台账，约束后续接受；不满足约束默认拒绝）
python -m venueflow.cli confirm S140 --limit 45 --note "春季方案"

# 接受预约（占用当时真实剩余额度；公益占用保留内未确认部分）
python -m venueflow.cli accept S140 --party G9 --participants 6 \
    --type commercial

# 办理候补（整团、按提交次序、实时剩余；搁置者下次重新参与）
python -m venueflow.cli waitlist S140

# 记录销售边界（停售/重开/调额建议，附分析依据）
python -m venueflow.cli boundary S140 --kind stop --remaining 0 \
    --reason "风险重估后主动停售"

# 单场综合查询：风险、承诺、公益释放、候补处置、补偿暴露、需求分类
python -m venueflow.cli status S140

# 端到端审计：重放台账，复核预测→决定→接受→候补全链条
python -m venueflow.cli verify
```

退出码：0 正常；1 校准门槛或审计未通过；2 业务拒绝（如额度不足、
限额低于已售承诺）。
