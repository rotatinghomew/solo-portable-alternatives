# 家长互助活动的偏好外推与轮换排课

数据来自独立设计的合成业务记录，不含真实人员资料。文件为 UTF-8 JSON。

## 资料字典

`choices` 逐家庭逐轮填写当时可见选项、候补未获邀请、实际选项、实际改选和明确不选。`offered` 包含先登记需求后才进入候补的活动，`wait_not_invited` 中的项目不算真正可以选择；`switch_to` 只有发生过改选时非空。`activities` 给出场地、时段、内容、可用席位和照看资源；各个活动和照看池都是独立容量，轮换时以原活动的照看池归属为准。`limits` 的池约束按时段共用。`confirmed` 是已经占用课程和相应照看资源的回执。`rotation` 同一个提案内的行应一起比较，迁移变更原供应而增设保留原供应；`unshown_scenarios` 是提案形成后的可用容量变更。`evaluation_cycles` 指专作核验的历史整期，不拿其中的结果拟合该轮预测。

## 读取与核对

使用 Python 3.11。
本地虚拟环境可按 `python3 -m pip install -r requirements.txt` 安装锁定的数值工具。有隔离运行环境时可直接使用已安装版本。

用 `python3 -m json.tool data/records.json` 查看原始记录。用 `python3 -m unittest discover -s tests -v` 核对演练资料的格式及关联。仓库目前只提供资料与输入核对，不提供业务分析结果。
