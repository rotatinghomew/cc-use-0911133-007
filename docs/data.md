# 业务样例数据

`fixtures/promise_risk.json` 使用本地合成数据，不含真实个人资料。

每条 received 的 quantity 是本次签收增量。at 与 known_at 分别表示事件和入账时间。只在 accepted 为真时视为双方接受改约；线路键用于表示共同运输环境。

`fixtures/demo.json` 是当前单笔业务流程的样例，和批量分析资料分开保存。
