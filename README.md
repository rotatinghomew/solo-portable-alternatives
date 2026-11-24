# 家长互助活动的偏好外推与轮换排课

数据来自独立设计的合成业务记录，不含真实人员资料。文件为 UTF-8 JSON。

## 资料字典

`choices` 逐家庭逐轮填写当时可见选项、候补未获邀请、实际选项、实际改选和明确不选。`offered` 包含先登记需求后才进入候补的活动，`wait_not_invited` 中的项目不算真正可以选择；`switch_to` 只有发生过改选时非空。`activities` 给出场地、时段、内容、可用席位和照看资源；各个活动和照看池都是独立容量，轮换时以原活动的照看池归属为准。`limits` 的池约束按时段共用。`confirmed` 是已经占用课程和相应照看资源的回执。`rotation` 同一个提案内的行应一起比较，迁移变更原供应而增设保留原供应；`unshown_scenarios` 是提案形成后的可用容量变更。`evaluation_cycles` 指专作核验的历史整期，不拿其中的结果拟合该轮预测。

## 读取与核对

使用 Python 3.11。
本地虚拟环境可按 `python3 -m pip install -r requirements.txt` 安装锁定的数值工具。有隔离运行环境时可直接使用已安装版本。

用 `python3 -m json.tool data/records.json` 查看原始记录。用 `python3 -m unittest discover -s tests -v` 核对演练资料的格式、关联与下列分析约定。

## 分析

```bash
# 生成完整评估（JSON + 会议摘要），默认拟合 1–5 轮、留出 6–7 轮
PYTHONPATH=. python3 -m parent_aid.cli --data data/records.json --out reports/rotation

# 使用提案形成后的容量通告（unshown_scenarios）重算容量
PYTHONPATH=. python3 -m parent_aid.cli --capacity-vintage updated --out reports/rotation_updated

# 新一轮报名/场地调整后，与上一轮报告对比，判断旧结论是否仍成立
PYTHONPATH=. python3 -m parent_aid.cli --out reports/rotation --previous reports/rotation.json
```

`reports/rotation.md` 是给运营会看的摘要，`reports/rotation.json` 是同一批结论的机器可读版本；输出保留 6 位小数并固定随机种子，同一输入两次运行结果一致，可以直接 diff。

### 口径

- **选择集**：`offered` 去掉 `wait_not_invited`。候补未获邀请不等于可以选，也不等于被拒绝。
- **未揭晓**：`selected` 为空且 `declined` 为假的行是仍在候补上的家庭，只贡献“本轮未参与”的质量，不当作对任何活动的拒绝；不进入选择拟合。
- **改选**：`switch_to` 是最终去向，`selected` 只是第一次选。改选目的地按其余可选活动的均匀分布处理；记录不支持“改选一定去某个更好的活动”（均匀性卡方检验写在报告里）。
- **一人一位**：每个家庭每轮是一个决策单位，全部活动概率加未参与概率等于 1。选项再多也不会变成多位参与者。
- **照看**：`care_pool` 跟着活动走，不跟着场地走。`limits` 按 (池, 时段) 共用；`confirmed` 先占用；`unshown_scenarios` 只在 `--capacity-vintage updated` 时替换上限。
- **整组比较**：同一提案内的行一起算，与当前计划对比预计接续与流失，不把单个换班的收益相加。
- **外推边界**：每个片区在记录里只出现过一个时段，因此“同片区换时段”没有任何可比记录，这类提案只会被判为证据不足并进入试开流程。报告会逐一列出从未开放过的“地点×时段”、样本过薄的内容格以及每格只有 2–5 户的人群分层。
- **试开**：证据不足的时段只能试开，且只能用未被 `confirmed` 和当前计划占用的照看余量；试开结果下一轮才进入模型，不写进本轮正式轮换理由。

### 留出核验

模型在 1–5 轮拟合，在 6–7 轮两个完整轮次上比较预测的选择分布与改选后分布，给出总变差距离、首选命中率、改选率以及按家庭聚类的自助抽样区间；另附“只看活动被选比例、不看片区”的朴素基线作对照。

### 模块

| 文件 | 职责 |
| --- | --- |
| `parent_aid/data.py` | 读取记录、选择集、照看归属、输入契约核对 |
| `parent_aid/model.py` | 选择模型（MNL，多种设定按 BIC 加权）、改选模型、家庭聚类自助抽样 |
| `parent_aid/support.py` | 逐项判断“换地方/换时段”能借用哪些证据、哪些组合不能外推 |
| `parent_aid/schedule.py` | 整组提案对比、座位与照看名额核算、被挤出家庭重新安置 |
| `parent_aid/validate.py` | 留出整期的选择/改选分布核验与基线 |
| `parent_aid/report.py` | 汇总报告、不确定性区间、试开安排、与上一轮的对比 |
| `parent_aid/cli.py` | 命令行入口 |
