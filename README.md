# b2b_reorder_network

商户返单业务处理：采购信任、需求记录、库存承诺、履约异常与对账。当前入口是 `b2b_reorder_network.dispatch.cli`，它使用本地 JSON 资料和 SQLite 状态；目录 `b2b_reorder_network` 存放业务模块，`tests/reorder_sagas` 验证既有业务行为。

## 环境

使用 Python 3.11，全部依赖来自标准库。`requirements.txt` 为空，不需要下载数据、模型或第三方包。数据库在运行时创建，可以删除临时数据库重新开始。

## 检查

```bash
python3 -m unittest discover -s tests/reorder_sagas -v
python3 -m unittest discover -s tests/analysis_context -v
python3 -m compileall -q b2b_reorder_network tests scripts
```

## 运行

```bash
python3 -m b2b_reorder_network.dispatch.cli  fixtures/demo.json --database /tmp/b2b_reorder_network-demo.sqlite3
python3 scripts/check_project.py
```

`check_project.py` 串行执行测试、编译和真实业务入口，将本次业务输出完整打印到终端；状态文件使用独立临时目录，重复运行不沿用上次状态。

## 交付承诺分析

```bash
python3 -m b2b_reorder_network.promise_risk fixtures/promise_risk.json --output /tmp/promise-risk
```

批量分析资料位于 `fixtures/promise_risk.json`，字段含义见 `docs/data.md`。它与当前单笔流程的输入格式不同，不能直接交给原来的入口处理。

该入口只读取资料并输出报告、摘要和 CSV，用来辅助比较整单与拆单，不会写数据库或下单。它回答“采购截止日之前能否交齐所需数量”，并把原承诺与双方接受后的承诺分开核对；未结案保留为未完成观察，终止不算送达；共同线路的联合风险单独建模。方法、假设与保留意见见 `docs/promise_risk.md`。

`check_project.py` 也会运行这个入口并把结果文件检查一遍；原有 `dispatch.cli` 入口与 `fixtures/demo.json` 流程保持不变。
