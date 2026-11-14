# 业务样例数据

`fixtures/promise_risk.json` 使用本地合成数据，不含真实个人资料。

每条 received 的 quantity 是本次签收增量。at 与 known_at 分别表示事件和入账时间。只在 accepted 为真时视为双方接受改约；线路键用于表示共同运输环境。

`fixtures/demo.json` 是当前单笔业务流程的样例，和批量分析资料分开保存。

## 批量履约资料字段

- `schema_version`：资料格式版本，目前为 1。
- `seed`：合成数据种子，便于复现。
- `observed_until`：观察截止日。晚于该日的 `known_at` 不会进入分析。
- `orders[]`：历史订单。
  - `order_id`、`supplier`、`route`、`category`、`placed_at`（下单日）。
  - `events[]`：按业务时间排列的事件。
    - `contract`：原合同，必须带 `promised_at`（原承诺交期）与 `quantity`。
    - `amendment`：改约，必须带 `promised_at` 与 `accepted`。只有 `accepted: true`
      且当时可见的改约才改变生效承诺；原承诺始终保留。
    - `received`：签收增量，必须带 `quantity`。`at` 是到达时间，`known_at`
      是入账时间，通常晚一天；分析只在 `known_at` 之后使用。
    - `terminated`：终止。终止不等于送达，分析中作为竞争事件。
- `validation_origins[]`：按日期向前验证的起点。每个起点只用 `known_at <= 起点`
  的数据训练。
- `purchase`：本次采购。
  - `placed_at`、`deadline`、`quantity`、`category`。
  - `suppliers[]`：`id`、`minimum`（最小起订量）、`capacity`（产能）、`route`。
- `unsupported_purchase`（可选）：历史未覆盖的品类与规模，用于演示保留意见。

## 未结案与观察

- 未结案订单保留为未完成观察，不是准时或迟到的完整标签。
- 观察年龄不足采购截止日：保留为删失（censored），不参与成功/失败计数。
- 观察年龄已满采购截止日仍未交齐：对“截止日前交齐”这一问题记为未交齐，
  但不贴“迟到完成”标签。
- 终止：竞争事件，不算送达。

## 共同线路

`route` 相同的供应商被视作共享运输环境。分析用同日、同品类的匹配订单组
估计线路内相关，线路之间默认独立并明确标注为假设。`rho` 接近 1 时提示
历史数据无法区分共同冲击与完全正相关。
