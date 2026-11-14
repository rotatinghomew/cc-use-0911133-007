"""交付承诺风险分析入口。

用法::

    python3 -m b2b_reorder_network.promise_risk fixtures/promise_risk.json \\
        --output /tmp/promise-risk

输出目录包含：

* ``report.json``：完整机器可读结果；
* ``summary.md``：给采购主管看的摘要；
* ``supplier_commitments.csv``：单供应商承诺概率；
* ``plan_comparison.csv``：整单与拆单方案的整批风险比较。

只读资料，不写数据库，不改变原有返单流程，也不会下单。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .analysis import analyze_dataset
from .data import FixtureError, load_dataset

SCHEMA = "b2b_reorder_network.promise_risk/v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _supplier_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    curves = report.get("quantity_curves", {})
    routes = report.get("data_quality", {}).get("routes", {})
    purchase = report.get("purchase", {})
    for supplier, curve in curves.items():
        for quantity, probability in curve.get("at_least", {}).items():
            rows.append(
                {
                    "supplier": supplier,
                    "route": routes.get(supplier, ""),
                    "quantity": int(quantity),
                    "probability_meet_by_deadline": probability,
                    "horizon_days": purchase.get("horizon_days"),
                    "category": purchase.get("category"),
                    "n_orders": curve.get("n_orders"),
                    "delivered": curve.get("outcomes", {}).get("delivered"),
                    "terminated": curve.get("outcomes", {}).get("terminated"),
                    "failed": curve.get("outcomes", {}).get("failed"),
                    "censored": curve.get("outcomes", {}).get("censored"),
                    "support": curve.get("support"),
                }
            )
    return rows


def _plan_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in report.get("plans", {}).get("evaluations", []):
        cohorts = evaluation.get("assumptions", {}).get("matched_cohorts", {})
        independence = evaluation.get("assumptions", {}).get("independence", {})
        rows.append(
            {
                "plan": json.dumps(evaluation.get("allocations", {}), ensure_ascii=False, sort_keys=True),
                "supplier_probabilities": json.dumps(
                    evaluation.get("supplier_probabilities", {}), ensure_ascii=False, sort_keys=True
                ),
                "naive_average_probability": evaluation.get("naive_average_probability"),
                "independent_product_probability": evaluation.get("independent_product_probability"),
                "joint_full_probability_common_route": evaluation.get("joint_full_probability"),
                "joint_full_probability_independence": independence.get("joint_full_probability"),
                "both_short_probability_common_route": evaluation.get("both_short_probability"),
                "both_short_probability_independence": independence.get("both_short_probability"),
                "expected_delivered_units": evaluation.get("expected_delivered_units"),
                "expected_shortfall_units": evaluation.get("expected_shortfall_units"),
                "shortage_loss": evaluation.get("shortage_loss"),
                "disruption_loss": evaluation.get("disruption_loss"),
                "hedging_loss": evaluation.get("hedging_loss"),
                "total_loss": evaluation.get("total_loss"),
                "matched_cohort_full_rate": cohorts.get("joint_full_rate"),
                "matched_cohorts": cohorts.get("cohorts"),
                "matched_cohort_supported": cohorts.get("supported"),
                "support": json.dumps(
                    evaluation.get("supplier_support", {}), ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.1f}"


def _allocation_text(allocations: Mapping[str, int] | None) -> str:
    if not allocations:
        return "—"
    return "；".join(f"{supplier} {quantity} 件" for supplier, quantity in sorted(allocations.items()))


def _summary_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    purchase = report.get("purchase", {})
    lines.append("# 交付承诺风险摘要")
    lines.append("")
    lines.append(
        f"采购：{purchase.get('quantity')} 件 {purchase.get('category')}，"
        f"{purchase.get('placed_at')} 下单，截止 {purchase.get('deadline')}"
        f"（{purchase.get('horizon_days')} 天）。"
    )
    lines.append("")
    lines.append("本报告只辅助比较整单与拆单，不会自动下单。")
    lines.append("")
    if not report.get("supported", False):
        reservation = report.get("reservation", {})
        lines.append("## 结论：保留意见")
        lines.append("")
        lines.append(reservation.get("statement", "历史资料不足，不能给出成功概率。"))
        lines.append("")
        for reason in reservation.get("reasons", []):
            lines.append(f"- {reason}")
        lines.append("")
        lines.append("### 什么情况下才能回答")
        lines.append("")
        for item in reservation.get("what_would_change_the_answer", []):
            lines.append(f"- {item}")
        return "\n".join(lines) + "\n"

    recommendation = report.get("plans", {}).get("primary_recommendation", {})
    lines.append("## 结论")
    lines.append("")
    lines.append(recommendation.get("statement", ""))
    lines.append("")
    if recommendation.get("status") != "supported":
        lines.append("### 现有资料能给出的只有保留意见")
        lines.append("")
        for reservation in report.get("reservations", []):
            lines.append(f"- {reservation}")
        lines.append("")
        return "\n".join(lines) + "\n"
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 推荐分配 | {_allocation_text(recommendation.get('allocations'))} |")
    for supplier, probability in recommendation.get("supplier_probabilities", {}).items():
        quantity = recommendation.get("allocations", {}).get(supplier)
        lines.append(
            f"| {supplier} 在截止日前交齐 {quantity} 件 | {_percent(probability)} |"
        )
    lines.append(f"| **整批全部交齐（共同线路假设）** | **{_percent(recommendation.get('joint_full_probability'))}** |")
    lines.append(f"| 各供应商概率的平均（会误导） | {_percent(recommendation.get('naive_average_probability'))} |")
    lines.append(f"| 各供应商概率的独立乘积 | {_percent(recommendation.get('independent_product_probability'))} |")
    lines.append(f"| 整批同时短缺 | {_percent(recommendation.get('both_short_probability'))} |")
    lines.append(f"| 截止日前期望交付 | {recommendation.get('expected_delivered_units')} 件 |")
    lines.append(f"| 期望短缺 | {recommendation.get('expected_shortfall_units')} 件 |")
    lines.append(f"| 期望业务损失 | {_money(recommendation.get('total_loss'))} |")
    lines.append(f"| 承诺阈值 | {_percent(recommendation.get('commit_threshold'))} |")
    lines.append("")
    lines.append(
        "整批交齐不是各供应商准时率的平均。也不能默认等于简单乘积：跨线路独立时"
        "乘积才近似成立；同线路共同冲击会同时抬高“都成功”和“都失败”的概率。"
    )
    lines.append("")

    lines.append("## 供应商画像")
    lines.append("")
    lines.append("| 供应商 | 线路 | 已完成 | 未结案 | 超过原承诺仍未结案 | 完成时长 p80 | 超原承诺完成 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for supplier, profile in report.get("suppliers", {}).items():
        lead = profile.get("completion_lead_days", {})
        lines.append(
            "| {supplier} | {route} | {completed} | {open} | {over} | {p80} 天 | {late} |".format(
                supplier=supplier,
                route=profile.get("route", ""),
                completed=profile.get("completed_orders"),
                open=profile.get("open_orders"),
                over=profile.get("open_over_contract_lead_days"),
                p80=lead.get("p80", "—"),
                late=_percent(profile.get("completed_after_contract_promise_share")),
            )
        )
    lines.append("")
    lines.append(
        "合同承诺交期可能都一样短；真正需要看的是完成时长分布、未结案数量和"
        "是否超过原承诺。未结案保留为未完成观察。"
    )
    lines.append("")

    lines.append("## 整单与拆单")
    lines.append("")
    reference = report.get("plans", {}).get("reference_plans", {})
    lines.append("整单交给单一供应商：")
    lines.append("")
    lines.append("| 方案 | 可行 | 原因 | 整批交齐概率 |")
    lines.append("| --- | --- | --- | --- |")
    for entry in reference.get("whole_order_single_supplier", []):
        evaluation = entry.get("evaluation") or {}
        lines.append(
            "| {alloc} | {feasible} | {why} | {probability} |".format(
                alloc=_allocation_text(entry.get("allocations")),
                feasible="是" if entry.get("feasible") else "否",
                why="；".join(entry.get("violations") or []) or "—",
                probability=_percent(evaluation.get("joint_full_probability")),
            )
        )
        if evaluation.get("supplier_support"):
            support = evaluation["supplier_support"]
            if any(value != "observed" for value in support.values()):
                lines.append(
                    f"|  |  | 覆盖提示：`{support}` | 外推，不作为承诺依据 |"
                )
    lines.append("")
    lines.append("两家拆单：先看每对供应商在历史观察规模内的最优方案，再看整批风险。")
    lines.append("")
    lines.append("| 组合 | 同线路 | 可行方案数 | 完全在观察规模内 | 代表方案（期望损失最小） | 整批交齐（共同线路） | 同时短缺 | 期望短缺 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in reference.get("split_two_suppliers", []):
        best = entry.get("best_supported") or {}
        lines.append(
            "| {pair} | {same} | {count} | {supported} | {alloc} | {joint} | {both} | {short} |".format(
                pair=" + ".join(entry.get("pair", [])),
                same="是" if entry.get("same_route") else "否",
                count=entry.get("feasible_plans"),
                supported=entry.get("supported_plans"),
                alloc=_allocation_text(best.get("allocations")) if best else "不可用",
                joint=_percent(best.get("joint_full_probability")) if best else "—",
                both=_percent(best.get("both_short_probability")) if best else "—",
                short=best.get("expected_shortfall_units") if best else "—",
            )
        )
    lines.append("")
    lines.append("同线路 vs 跨线路对照（每对供应商优先选完全落在历史观察规模内、整批交齐概率最高的方案）：")
    lines.append("")
    lines.append("| 组合 | 同线路 | 方案 | 整批交齐（共同线路） | 整批交齐（独立） | 同时短缺（共同线路） | 同时短缺（独立） |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report.get("plans", {}).get("route_comparison", []):
        lines.append(
            "| {pair} | {same} | {alloc} | {joint} | {ind} | {short} | {short_ind} |".format(
                pair=" + ".join(row.get("suppliers", [])),
                same="是" if row.get("same_route") else "否",
                alloc=_allocation_text(row.get("allocations")),
                joint=_percent(row.get("joint_full_probability_common_route")),
                ind=_percent(row.get("joint_full_probability_independence")),
                short=_percent(row.get("all_short_probability_common_route")),
                short_ind=_percent(row.get("all_short_probability_independence")),
            )
        )
    lines.append("")

    dependence = report.get("dependence", {})
    lines.append("## 共同线路")
    lines.append("")
    for detail in dependence.get("route_rho_detail", []):
        lines.append(
            f"- {detail.get('route')}：rho={detail.get('rho')}，"
            f"来源 {detail.get('source')}；{detail.get('note')}"
        )
    lines.append("")
    example = dependence.get("shared_route_example", {})
    pair = example.get("pair") or {}
    if pair:
        lines.append(
            f"- 同日匹配组 {pair.get('matched_cohorts')} 组；"
            f"两家都交齐 {pair.get('both_on_time')} 组，两家都短缺 {pair.get('both_short')} 组；"
            f"同时短缺经验率 {_percent(pair.get('both_short_rate_empirical'))}，"
            f"独立假设下 {_percent(pair.get('both_short_rate_independence'))}。"
        )
    late = example.get("late_cohort")
    if late:
        lines.append(
            f"- 示例批次 {late.get('placed_at')}："
            + "；".join(
                f"{supplier} 完成 {values.get('completion_lead_days')} 天"
                f"、截止日前交付 {values.get('delivered_by_purchase_deadline')} 件"
                for supplier, values in late.get("suppliers", {}).items()
            )
            + "。两家共用集货线路同时迟到，另一条线路按期。"
        )
    lines.append(f"- {example.get('interpretation', '')}")
    lines.append("")

    lines.append("## 推荐为何变化")
    lines.append("")
    for change in recommendation.get("changes", []):
        if change.get("recommendation_changes"):
            lines.append(
                f"- 按 `{change.get('rule')}` 排序会从 {_allocation_text(change.get('from'))} "
                f"变为 {_allocation_text(change.get('to'))}。{change.get('explanation')}"
            )
            comparison = change.get("comparison", {})
            lines.append(
                "  对应差距：整批交齐 {} → {}，同时短缺 {} → {}，期望短缺 {} → {} 件，"
                "业务损失 {} → {}。".format(
                    _percent(comparison.get("primary_joint_full_probability")),
                    _percent(comparison.get("candidate_joint_full_probability")),
                    _percent(comparison.get("primary_both_short_probability")),
                    _percent(comparison.get("candidate_both_short_probability")),
                    comparison.get("primary_expected_shortfall_units"),
                    comparison.get("candidate_expected_shortfall_units"),
                    _money(comparison.get("primary_total_loss")),
                    _money(comparison.get("candidate_total_loss")),
                )
            )
        else:
            lines.append(f"- 按 `{change.get('rule')}` 排序推荐不变。")
    gap = report.get("plans", {}).get("joint_vs_naive_gap", {})
    if gap:
        lines.append("")
        lines.append(gap.get("note", ""))
    lines.append("")

    lines.append("## 改约与未结案")
    lines.append("")
    promise = report.get("promise_basis", {})
    lines.append(
        "- 改约事件 {} 个：双方接受 {} 个、拒绝 {} 个；原承诺与接受后承诺分别核对。".format(
            promise.get("amendment_events"),
            promise.get("accepted_amendments"),
            promise.get("rejected_amendments"),
        )
    )
    original = promise.get("original", {})
    accepted = promise.get("accepted", {})
    lines.append(f"- 原承诺口径：{original.get('status_counts')}")
    lines.append(f"- 接受后承诺口径：{accepted.get('status_counts')}")
    for item in promise.get("reclassified_between_bases", []):
        lines.append(
            f"- {item.get('order_id')}：原承诺下 {item.get('original_status')}，"
            f"双方接受延期后 {item.get('accepted_status')}（完成于 {item.get('completed_at')}）。"
        )
    lines.append("")
    unclosed = report.get("unclosed_examples", {})
    for item in unclosed.get("open_orders", []):
        lines.append(
            f"- {item.get('order_id')}（{item.get('supplier')}）：{item.get('treatment')}，"
            f"{item.get('explanation')}"
        )
    for item in unclosed.get("terminated_orders", [])[:4]:
        lines.append(
            f"- {item.get('order_id')}（{item.get('supplier')}）：终止于第 {item.get('terminated_at_day')} 天，"
            "终止不等于送达。"
        )
    lines.append("")

    lines.append("## 校准与基准")
    lines.append("")
    validation = report.get("validation", {})
    lines.append(
        f"官方验证起点：{', '.join(validation.get('origins', []))}；"
        f"选中模型：`{validation.get('selected_model')}`（{validation.get('selection_criterion')}）。"
    )
    lines.append("")
    lines.append("| 模型 | 点数 | Brier | 对数损失 | 校准误差 | 观测交齐率 | 平均预测 | 期望成本/单 | 承诺比例 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for metric in validation.get("metrics", []):
        lines.append(
            "| {model} | {points} | {brier} | {ll} | {cal} | {obs} | {pred} | {cost} | {commit} |".format(
                model=metric.get("model"),
                points=metric.get("points"),
                brier=metric.get("brier"),
                ll=metric.get("log_loss"),
                cal=metric.get("calibration_error"),
                obs=_percent(metric.get("observed_rate")),
                pred=_percent(metric.get("mean_prediction")),
                cost=_money(metric.get("expected_cost_per_order")),
                commit=_percent(metric.get("commit_rate")),
            )
        )
    reference_costs = validation.get("reference_costs", {})
    lines.append(
        f"| 对照：一律承诺 / 一律准备替代 | — | — | — | — | — | — | "
        f"{_money(reference_costs.get('always_commit_per_order'))} / "
        f"{_money(reference_costs.get('always_hedge_per_order'))} | — |"
    )
    lines.append("")
    rolling = validation.get("supplementary_rolling")
    if rolling:
        lines.append(
            "补充每周滚动起点（只用于稳定性检查）："
            + "；".join(
                f"{metric.get('model')} Brier={metric.get('brier')}、对数损失={metric.get('log_loss')}"
                for metric in rolling.get("metrics", [])
            )
            + "。"
        )
        lines.append("")
    lines.append(
        "验证标签只使用看满截止日的观察；终止和已过截止日未结案都记为未交齐。"
        "模型选择也在同一批验证点上完成，因此选中模型的误差偏乐观。"
    )
    lines.append("")

    lines.append("## 保留意见")
    lines.append("")
    for reservation in report.get("reservations", []):
        lines.append(f"- {reservation}")
    unsupported = report.get("unsupported_purchase")
    if unsupported:
        lines.append(
            f"- 未覆盖品类 {unsupported.get('category')} / {unsupported.get('quantity')} 件："
            f"{unsupported.get('statement')}"
        )
    lines.append("")
    lines.append("## 关键假设")
    lines.append("")
    for assumption in report.get("assumptions", []):
        lines.append(f"- {assumption}")
    lines.append("")
    return "\n".join(lines)


def run(fixture: str | Path, output: str | Path) -> dict[str, Any]:
    dataset = load_dataset(fixture)
    report = analyze_dataset(dataset)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "report.json", {**report, "schema": SCHEMA})
    (output_path / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    _write_csv(
        output_path / "supplier_commitments.csv",
        _supplier_rows(report),
        [
            "supplier",
            "route",
            "quantity",
            "probability_meet_by_deadline",
            "horizon_days",
            "category",
            "n_orders",
            "delivered",
            "terminated",
            "failed",
            "censored",
            "support",
        ],
    )
    _write_csv(
        output_path / "plan_comparison.csv",
        _plan_rows(report),
        [
            "plan",
            "supplier_probabilities",
            "naive_average_probability",
            "independent_product_probability",
            "joint_full_probability_common_route",
            "joint_full_probability_independence",
            "both_short_probability_common_route",
            "both_short_probability_independence",
            "expected_delivered_units",
            "expected_shortfall_units",
            "shortage_loss",
            "disruption_loss",
            "hedging_loss",
            "total_loss",
            "matched_cohort_full_rate",
            "matched_cohorts",
            "matched_cohort_supported",
            "support",
        ],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="交付承诺风险分析：先比较整单与拆单，不自动下单"
    )
    parser.add_argument("fixture", help="批量履约资料 JSON")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出简短摘要，不打印完整 JSON",
    )
    args = parser.parse_args(argv)
    try:
        report = run(args.fixture, args.output)
    except FixtureError as exc:
        print(f"资料错误: {exc}", file=sys.stderr)
        return 2
    recommendation = report.get("plans", {}).get("primary_recommendation", {})
    payload: dict[str, Any] = {
        "output": str(Path(args.output)),
        "supported": report.get("supported"),
    }
    if report.get("supported"):
        payload.update(
            {
                "horizon_days": report.get("purchase", {}).get("horizon_days"),
                "quantity": report.get("purchase", {}).get("quantity"),
                "recommendation": recommendation.get("allocations"),
                "supplier_probabilities": recommendation.get("supplier_probabilities"),
                "joint_full_probability": recommendation.get("joint_full_probability"),
                "naive_average_probability": recommendation.get("naive_average_probability"),
                "both_short_probability": recommendation.get("both_short_probability"),
                "expected_shortfall_units": recommendation.get("expected_shortfall_units"),
                "meets_commit_threshold": recommendation.get("meets_threshold"),
                "selected_model": report.get("validation", {}).get("selected_model"),
                "reservations": report.get("reservations", []),
            }
        )
    else:
        payload["reservation"] = report.get("reservation", {}).get("statement")
    if args.quiet:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps({**report, "console": payload}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
