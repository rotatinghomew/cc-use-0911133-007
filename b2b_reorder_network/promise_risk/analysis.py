"""交付承诺分析的总编排：标签、估计、验证、方案与保留意见。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from .data import Dataset, Purchase
from .dependence import (
    matched_pair_dependence,
    rho_source,
    route_rho,
    route_rho_detail,
)
from .plans import (
    PlanEvaluation,
    evaluate_plans,
    feasible_allocations,
)
from .promises import (
    PROMISE_BASIS_ACCEPTED,
    PROMISE_BASIS_ORIGINAL,
    label_order,
    label_summary,
    original_promise,
    effective_promise,
)
from .survival import (
    Timeline,
    build_timelines,
    quantity_curve,
)
from .validation import baseline_comparison, forward_validation


KEY_QUANTITIES = (20, 40)
COMMIT_THRESHOLD = 0.8
SHORTAGE_COST_PER_UNIT = 20.0
DISRUPTION_COST_PER_UNIT = 8.0
HEDGE_COST_PER_UNIT = 6.0


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _promise_basis(dataset: Dataset) -> dict[str, Any]:
    original_labels = [label_order(order, PROMISE_BASIS_ORIGINAL, dataset.observed_until) for order in dataset.orders]
    accepted_labels = [label_order(order, PROMISE_BASIS_ACCEPTED, dataset.observed_until) for order in dataset.orders]
    reclassified = []
    for left, right in zip(original_labels, accepted_labels):
        if left.status != right.status:
            reclassified.append(
                {
                    "order_id": left.order_id,
                    "supplier": left.supplier,
                    "original_promise": left.promised_at.isoformat(),
                    "accepted_promise": right.promised_at.isoformat(),
                    "original_status": left.status,
                    "accepted_status": right.status,
                    "completed_at": left.completed_at.isoformat() if left.completed_at else None,
                }
            )
    amendments = [
        event
        for order in dataset.orders
        for event in order.visible_events(dataset.observed_until)
        if event.kind == "amendment"
    ]
    accepted = sum(1 for event in amendments if event.accepted is True)
    rejected = sum(1 for event in amendments if event.accepted is False)
    last_extension_orders = [
        order.order_id
        for order in dataset.orders
        if any(
            event.kind == "amendment" and event.accepted is True
            for event in order.visible_events(dataset.observed_until)
        )
    ]
    examples: list[dict[str, Any]] = []
    for order in dataset.orders:
        accepted_events = [
            event
            for event in order.visible_events(dataset.observed_until)
            if event.kind == "amendment" and event.accepted is True
        ]
        if not accepted_events:
            continue
        labels = label_order(order, PROMISE_BASIS_ORIGINAL, dataset.observed_until)
        accepted_label = label_order(order, PROMISE_BASIS_ACCEPTED, dataset.observed_until)
        examples.append(
            {
                "order_id": order.order_id,
                "supplier": order.supplier,
                "placed_at": order.placed_at.isoformat(),
                "original_promise": original_promise(order).isoformat(),
                "accepted_promise": effective_promise(order, PROMISE_BASIS_ACCEPTED, dataset.observed_until).isoformat(),
                "original_status": labels.status,
                "accepted_status": accepted_label.status,
                "completed_at": labels.completed_at.isoformat() if labels.completed_at else None,
                "terminated_at": labels.terminated_at.isoformat() if labels.terminated_at else None,
                "note": "原承诺保留；最后一次双方接受的延期只用于第二套口径。",
            }
        )
    return {
        "original": label_summary(original_labels),
        "accepted": label_summary(accepted_labels),
        "amendment_events": len(amendments),
        "accepted_amendments": accepted,
        "rejected_amendments": rejected,
        "orders_with_accepted_amendment": len(last_extension_orders),
        "reclassified_between_bases": reclassified,
        "examples": examples,
        "rules": [
            "只有 accepted=true 且当时可见的改约才改变生效承诺",
            "被拒绝的改约不改变承诺",
            "终止不等于送达",
            "已过承诺日但未结案是未完成观察，不是迟到完成标签",
        ],
    }


def _supplier_profiles(dataset: Dataset, horizon: int) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for supplier in dataset.suppliers:
        orders = dataset.orders_for(supplier=supplier)
        timelines = [timeline for timeline in build_timelines(dataset, dataset.observed_until, supplier=supplier)]
        completed = [
            timeline
            for timeline in timelines
            if timeline.completion_day(timeline.quantity) is not None
            and timeline.termination_day is None
        ]
        leads = sorted(timeline.completion_day(timeline.quantity) for timeline in completed)
        open_orders = [
            timeline
            for timeline in timelines
            if timeline.completion_day(timeline.quantity) is None
            and timeline.termination_day is None
        ]
        overdue = []
        for timeline in open_orders:
            order = next(order for order in orders if order.order_id == timeline.order_id)
            promised = effective_promise(order, PROMISE_BASIS_ACCEPTED, dataset.observed_until)
            if dataset.observed_until > promised:
                overdue.append(timeline.order_id)
        contract_leads = sorted(
            (original_promise(order) - order.placed_at).days for order in orders
        )
        late_vs_contract = 0
        for timeline in completed:
            order = next(order for order in orders if order.order_id == timeline.order_id)
            day = timeline.completion_day(timeline.quantity)
            if day is not None and day > (original_promise(order) - order.placed_at).days:
                late_vs_contract += 1
        profiles[supplier] = {
            "route": dataset.routes[supplier],
            "orders": len(orders),
            "contract_promised_lead_days": {
                "min": contract_leads[0] if contract_leads else None,
                "max": contract_leads[-1] if contract_leads else None,
            },
            "completed_orders": len(completed),
            "terminated_orders": sum(1 for timeline in timelines if timeline.termination_day is not None),
            "open_orders": len(open_orders),
            "open_order_ids": [timeline.order_id for timeline in open_orders],
            "overdue_unresolved_order_ids": overdue,
            "completion_lead_days": {
                "p50": _percentile(leads, 0.5),
                "p80": _percentile(leads, 0.8),
                "p90": _percentile(leads, 0.9),
                "max": leads[-1] if leads else None,
            },
            "completed_after_contract_promise": late_vs_contract,
            "completed_after_contract_promise_share": (
                round(late_vs_contract / len(completed), 4) if completed else None
            ),
            "delivered_by_purchase_horizon": sum(
                1
                for timeline in timelines
                if (lead := timeline.completion_day(timeline.quantity)) is not None and lead <= horizon
            ),
            "open_over_contract_lead_days": sum(
                1
                for timeline in open_orders
                if timeline.observed_age > (original_promise(
                    next(order for order in orders if order.order_id == timeline.order_id)
                ) - next(order for order in orders if order.order_id == timeline.order_id).placed_at).days
            ),
            "note": (
                "合同承诺交期可能相同，但实际完成时长、未结案数量和超原承诺比例不同；"
                "未结案保留为未完成观察，不当作准时或迟到标签。"
            ),
        }
    return profiles


def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    return ordered[int(round(position))]


def _dependence(dataset: Dataset, horizon: int, quantity: int) -> dict[str, Any]:
    pairs = matched_pair_dependence(
        dataset,
        dataset.purchase.category,
        horizon,
        quantity,
        dataset.observed_until,
    )
    rho_map = route_rho(pairs, min_cohorts=10, default=0.5)
    sources = rho_source(pairs, min_cohorts=10)
    for route in sorted({dataset.routes[supplier] for supplier in dataset.suppliers}):
        sources.setdefault(route, "assumption")
        rho_map.setdefault(route, 0.0)
    return {
        "pairs": [pair.as_dict() for pair in pairs],
        "route_rho": {route: _round(value, 4) for route, value in rho_map.items()},
        "route_rho_detail": route_rho_detail(pairs, rho_map),
        "rho_source": sources,
        "matched_cohort_definition": (
            "同一天、同品类、数量达到整单目标、且结局可判定的订单组；"
            "不足截止日观察且未交齐的组不进入匹配。"
        ),
        "assumption_notes": [
            "同线路内使用高斯 copula 的正相关，相关系数由匹配订单组的联合结果估计。",
            "线路之间默认独立；这是假设，不是从数据估计出的共同冲击。",
            "匹配组样本很小，rho 可能落在网格边界，此时同时短缺仍可能被低估。",
        ],
    }


def _plan_reference(evaluations: Sequence[PlanEvaluation], allocations: Mapping[str, int]) -> dict[str, Any] | None:
    for evaluation in evaluations:
        if evaluation.plan.allocations == dict(sorted(allocations.items())):
            return evaluation.as_dict()
    return None


def _rule_rankings(evaluations: Sequence[PlanEvaluation]) -> dict[str, list[dict[str, Any]]]:
    supported = [evaluation for evaluation in evaluations if _fully_supported(evaluation)]

    def rank(key, reverse=True, rows=None):
        rows = rows if rows is not None else supported
        rows = [row for row in rows if key(row) is not None]
        return sorted(rows, key=key, reverse=reverse)

    rules = {
        "max_joint_full_probability": rank(
            lambda row: row.plan.joint_probability
        ),
        "min_both_short_probability": rank(
            lambda row: row.plan.both_short_probability, reverse=False
        ),
        "min_expected_business_loss": rank(
            lambda row: row.plan.total_loss, reverse=False
        ),
        "max_naive_average_probability": rank(
            lambda row: row.plan.naive_average_probability
        ),
        "max_matched_cohort_full_rate": rank(
            lambda row: row.empirical_full if row.empirical_supported else None
        ),
    }
    return {
        rule: [
            {
                "allocations": dict(sorted(row.plan.allocations.items())),
                "joint_full_probability": _round(row.plan.joint_probability),
                "both_short_probability": _round(row.plan.both_short_probability),
                "naive_average_probability": _round(row.plan.naive_average_probability),
                "independent_product_probability": _round(row.plan.product_probability),
                "expected_delivered_units": _round(row.plan.expected_delivered, 3),
                "expected_shortfall_units": _round(row.plan.expected_shortfall, 3),
                "total_loss": _round(row.plan.total_loss, 2),
                "matched_cohort_full_rate": _round(row.empirical_full),
                "matched_cohorts": row.empirical_cohorts,
                "supported": _fully_supported(row),
            }
            for row in rows[:5]
        ]
        for rule, rows in rules.items()
    }


def _ranking_note(rankings: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """指出平均值/乘积口径与联合口径的差距，帮助解释推荐为何变化。"""

    notes: list[dict[str, Any]] = []
    for rule, rows in rankings.items():
        if not rows:
            continue
        top = rows[0]
        joint = top.get("joint_full_probability")
        average = top.get("naive_average_probability")
        product = top.get("independent_product_probability")
        if joint is None:
            continue
        notes.append(
            {
                "rule": rule,
                "top_plan": top["allocations"],
                "joint_full_probability": joint,
                "average_of_supplier_probabilities": average,
                "product_of_supplier_probabilities": product,
                "average_overstates_joint_by": _round(
                    (average - joint) if average is not None else None
                ),
                "product_overstates_joint_by": _round(
                    (product - joint) if product is not None else None
                ),
            }
        )
    return {
        "rule": "joint_vs_naive_gap",
        "note": (
            "整批概率必须用联合分布；单供应商概率的平均几乎总会高估整批交齐。"
            "乘积只有在跨线路独立时才成立；同线路正相关时，乘积会低估“都失败”"
            "的同时发生程度。负数表示联合口径比乘积更高（正相关让“都成功”也更集中），"
            "正数表示乘积高估了联合概率。"
        ),
        "comparisons": notes,
    }


def _fully_supported(evaluation: PlanEvaluation) -> bool:
    support = evaluation.plan.supplier_support
    if not support:
        return False
    return all(value == "observed" for value in support.values())


def _plan_route_comparison(evaluations: Sequence[PlanEvaluation]) -> list[dict[str, Any]]:
    """同线路与跨线路方案的风险对照，突出共同冲击的影响。

    每对供应商优先选完全落在历史观察规模内、且整批交齐概率最高的方案；
    只有该组合没有可支持方案时才退回外推方案，并标注 ``extrapolated``。
    """

    best: dict[tuple[str, ...], PlanEvaluation] = {}
    for evaluation in evaluations:
        suppliers = tuple(sorted(evaluation.plan.allocations))
        if len(suppliers) < 2:
            continue
        current = best.get(suppliers)
        if current is None:
            best[suppliers] = evaluation
            continue
        better_support = _fully_supported(evaluation) and not _fully_supported(current)
        same_support = _fully_supported(evaluation) == _fully_supported(current)
        better_joint = (evaluation.plan.joint_probability or 0.0) > (
            current.plan.joint_probability or 0.0
        )
        if better_support or (same_support and better_joint):
            best[suppliers] = evaluation
    rows: list[dict[str, Any]] = []
    for suppliers, evaluation in best.items():
        routes = {evaluation.plan.routes[supplier] for supplier in suppliers}
        rows.append(
            {
                "suppliers": list(suppliers),
                "allocations": dict(sorted(evaluation.plan.allocations.items())),
                "same_route": len(routes) == 1,
                "routes": sorted(routes),
                "joint_full_probability_common_route": _round(evaluation.plan.joint_probability),
                "joint_full_probability_independence": _round(evaluation.joint_independence),
                "all_short_probability_common_route": _round(evaluation.plan.both_short_probability),
                "all_short_probability_independence": _round(evaluation.both_short_independence),
                "expected_delivered_units": _round(evaluation.plan.expected_delivered, 3),
                "expected_shortfall_units": _round(evaluation.plan.expected_shortfall, 3),
                "total_loss": _round(evaluation.plan.total_loss, 2),
                "supplier_support": dict(sorted(evaluation.plan.supplier_support.items())),
                "fully_supported": _fully_supported(evaluation),
            }
        )
    rows.sort(
        key=lambda row: (
            not row["fully_supported"],
            not row["same_route"],
            -(row["joint_full_probability_common_route"] or 0.0),
        )
    )
    return rows


def _plan_changes(
    rankings: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    primary = rankings.get("min_expected_business_loss", [])
    if not primary:
        return []
    baseline = primary[0]
    changes: list[dict[str, Any]] = []
    for rule, rows in rankings.items():
        if rule == "min_expected_business_loss" or not rows:
            continue
        candidate = rows[0]
        if candidate["allocations"] == baseline["allocations"]:
            changes.append(
                {
                    "rule": rule,
                    "recommendation_changes": False,
                    "allocations": candidate["allocations"],
                    "explanation": "该规则与主推荐选择同一方案。",
                }
            )
            continue
        changes.append(
            {
                "rule": rule,
                "recommendation_changes": True,
                "from": baseline["allocations"],
                "to": candidate["allocations"],
                "comparison": {
                    "primary_joint_full_probability": baseline["joint_full_probability"],
                    "candidate_joint_full_probability": candidate["joint_full_probability"],
                    "primary_both_short_probability": baseline["both_short_probability"],
                    "candidate_both_short_probability": candidate["both_short_probability"],
                    "primary_expected_shortfall_units": baseline["expected_shortfall_units"],
                    "candidate_expected_shortfall_units": candidate["expected_shortfall_units"],
                    "primary_total_loss": baseline["total_loss"],
                    "candidate_total_loss": candidate["total_loss"],
                },
                "explanation": _change_explanation(rule),
            }
        )
    return changes


def _change_explanation(rule: str) -> str:
    if rule == "max_joint_full_probability":
        return (
            "只看“整批同时交齐”的最大概率时，共同线路的正相关会抬高两家同时成功的概率，"
            "因此可能偏向同线路组合；但同一相关性也抬高了两家同时失败的概率。"
            "概率更高不等于承诺更安全。"
        )
    if rule == "max_naive_average_probability":
        return (
            "简单平均把每个供应商的交齐概率平均，完全忽略联合事件；"
            "当方案把数量放在共同线路上时，平均值会系统性地掩盖同时短缺。"
        )
    if rule == "max_matched_cohort_full_rate":
        return (
            "匹配组经验率直接数历史同批订单，不依赖 copula；"
            "但匹配组只覆盖历史出现过的分配规模，超出该规模时不能外推。"
        )
    if rule == "min_both_short_probability":
        return (
            "最小化同时短缺概率会把订单分散到不同线路，即使整批同时交齐的概率略低；"
            "这是风险偏好不同的选择。"
        )
    return "不同决策规则的目标函数不同。"


def _reference_plans(
    dataset: Dataset,
    purchase: Purchase,
    evaluations: Sequence[PlanEvaluation],
) -> dict[str, Any]:
    total = purchase.quantity
    single = []
    for supplier in purchase.suppliers:
        allocation = {supplier.supplier_id: total}
        infeasible = []
        if total > supplier.capacity:
            infeasible.append(f"超过产能 {supplier.capacity}")
        if total < supplier.minimum:
            infeasible.append(f"低于最小起订量 {supplier.minimum}")
        entry: dict[str, Any] = {
            "allocations": allocation,
            "feasible": not infeasible,
            "violations": infeasible,
        }
        evaluation = _plan_reference(evaluations, allocation)
        if evaluation:
            entry["evaluation"] = evaluation
        single.append(entry)
    split_two = []
    for left_index, left in enumerate(purchase.suppliers):
        for right in purchase.suppliers[left_index + 1 :]:
            allocations = []
            for quantity in range(left.minimum, min(left.capacity, total) + 1):
                remainder = total - quantity
                if remainder < right.minimum or remainder > right.capacity:
                    continue
                evaluation = _plan_reference(evaluations, {left.supplier_id: quantity, right.supplier_id: remainder})
                if evaluation:
                    allocations.append(evaluation)
            supported = [
                item
                for item in allocations
                if all(value == "observed" for value in item.get("supplier_support", {}).values())
            ]
            supported.sort(
                key=lambda item: (
                    item.get("total_loss") is None,
                    item.get("total_loss") if item.get("total_loss") is not None else math.inf,
                )
            )
            split_two.append(
                {
                    "pair": [left.supplier_id, right.supplier_id],
                    "routes": [left.route, right.route],
                    "same_route": left.route == right.route,
                    "feasible_plans": len(allocations),
                    "supported_plans": len(supported),
                    "best_supported": supported[0] if supported else None,
                    "selection": (
                        "优先在完全落在历史观察规模内的方案里按期望业务损失选择；"
                        "没有这种方案时返回 None，不拿外推方案冒充可验证推荐。"
                    ),
                }
            )
    return {
        "whole_order_single_supplier": single,
        "whole_order_feasible": any(entry["feasible"] for entry in single),
        "split_two_suppliers": split_two,
        "note": (
            "整单交给单一供应商在产能约束下不可行时，拆单是唯一可行路径；"
            "但拆单不等于分散风险，必须看共同线路。"
        ),
    }


def _recommendation(
    dataset: Dataset,
    purchase: Purchase,
    evaluations: Sequence[PlanEvaluation],
    rankings: Mapping[str, list[dict[str, Any]]],
    *,
    commit_threshold: float = COMMIT_THRESHOLD,
) -> dict[str, Any]:
    supported_rows = [row for row in evaluations if _fully_supported(row)]
    primary = None
    if supported_rows:
        primary = min(
            supported_rows,
            key=lambda row: (
                row.plan.total_loss if row.plan.total_loss is not None else math.inf,
                -(row.plan.joint_probability or 0.0),
            ),
        )
    unrestricted = None
    candidates = [row for row in evaluations if row.plan.total_loss is not None]
    if candidates:
        unrestricted = min(candidates, key=lambda row: row.plan.total_loss)
    if primary is None:
        feasible = feasible_allocations(purchase)
        if not feasible:
            statement = (
                f"在最小起订量与产能约束下，没有可行方案能正好覆盖 {purchase.quantity} 件；"
                "需要调整采购数量、供应商产能或最小起订量后再评估。"
            )
        else:
            statement = (
                "没有方案完全落在历史观察规模内，不能给出可验证的整批交齐概率；"
                "所有候选都需要外推，应按保留意见处理。"
            )
        return {
            "status": "insufficient_support",
            "statement": statement,
            "changes": [],
        }
    probability = primary.plan.joint_probability
    meets = probability is not None and probability >= commit_threshold
    if meets:
        statement = (
            f"在共同线路假设下，推荐方案 {dict(sorted(primary.plan.allocations.items()))} "
            f"整批交齐概率 {probability:.1%}，不低于承诺阈值 {commit_threshold:.0%}。"
        )
    else:
        statement = (
            f"在共同线路假设下，最好的完全覆盖方案 {dict(sorted(primary.plan.allocations.items()))} "
            f"整批交齐概率只有 {probability:.1%}，低于承诺阈值 {commit_threshold:.0%}；"
            f"不应据此承诺整批 {purchase.quantity} 件全部按期到货，应准备替代资源或分批承诺。"
        )
    return {
        "status": "supported",
        "criterion": "min_expected_business_loss_under_common_route_dependence",
        "allocations": dict(sorted(primary.plan.allocations.items())),
        "supplier_probabilities": {
            key: _round(value) for key, value in sorted(primary.plan.supplier_probabilities.items())
        },
        "supplier_support": dict(sorted(primary.plan.supplier_support.items())),
        "joint_full_probability": _round(probability),
        "both_short_probability": _round(primary.plan.both_short_probability),
        "naive_average_probability": _round(primary.plan.naive_average_probability),
        "independent_product_probability": _round(primary.plan.product_probability),
        "expected_delivered_units": _round(primary.plan.expected_delivered, 3),
        "expected_shortfall_units": _round(primary.plan.expected_shortfall, 3),
        "total_loss": _round(primary.plan.total_loss, 2),
        "commit_threshold": commit_threshold,
        "meets_threshold": meets,
        "statement": statement,
        "unrestricted_alternative": (
            {
                "allocations": dict(sorted(unrestricted.plan.allocations.items())),
                "joint_full_probability": _round(unrestricted.plan.joint_probability),
                "total_loss": _round(unrestricted.plan.total_loss, 2),
                "warning": "该方案需要超出历史单笔观察规模的外推，只能作为保留意见，不作为承诺依据。",
            }
            if unrestricted is not None
            and dict(sorted(unrestricted.plan.allocations.items())) != dict(sorted(primary.plan.allocations.items()))
            else None
        ),
        "ranking_summary": rankings,
    }


def _unclosed_examples(dataset: Dataset, horizon: int) -> dict[str, Any]:
    timelines = build_timelines(dataset, dataset.observed_until)
    examples = []
    for timeline in timelines:
        if timeline.completion_day(timeline.quantity) is not None or timeline.termination_day is not None:
            continue
        age = timeline.observed_age
        if age < horizon:
            treatment = "censored_not_due"
            explanation = (
                f"观察年龄 {age} 天，不足截止日 {horizon} 天；保留为未完成观察，"
                "既不算准时也不算迟到，也不会从样本中删除。"
            )
        else:
            treatment = "failed_by_deadline"
            explanation = (
                f"观察年龄 {age} 天，已看满截止日 {horizon} 天仍未交齐；"
                "对“截止日前交齐”这个问题记为未交齐，但不贴“迟到完成”标签。"
            )
        examples.append(
            {
                "order_id": timeline.order_id,
                "supplier": timeline.supplier,
                "route": timeline.route,
                "category": timeline.category,
                "placed_at": timeline.placed_at.isoformat(),
                "observed_age_days": age,
                "delivered_units_by_deadline": timeline.delivered_quantity(horizon),
                "ordered_units": timeline.quantity,
                "treatment": treatment,
                "explanation": explanation,
            }
        )
    closed = []
    for timeline in timelines:
        if timeline.termination_day is None:
            continue
        delivered_before_termination = max(
            (
                cumulative
                for day, cumulative in timeline.receipt_cum
                if day <= (timeline.termination_day or 0)
            ),
            default=0,
        )
        closed.append(
            {
                "order_id": timeline.order_id,
                "supplier": timeline.supplier,
                "terminated_at_day": timeline.termination_day,
                "delivered_units": delivered_before_termination,
                "ordered_units": timeline.quantity,
                "treatment": "terminated_not_delivered",
                "explanation": "终止不等于送达；作为竞争事件，不计入交齐。",
            }
        )
    return {
        "open_orders": examples,
        "terminated_orders": closed,
        "impact_note": (
            "把所有未结案直接当失败会高估风险；直接丢掉会低估风险；"
            "把已过截止日的未结案当迟到完成则会污染标签。这里按观察年龄分别处理。"
        ),
    }


def _shared_route_example(dataset: Dataset, dependence: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    pair = None
    for item in dependence["pairs"]:
        if item["same_route"] and item["matched_cohorts"] >= 10:
            pair = item
            break
    late_cohort = None
    by_day: dict[date, dict[str, Timeline]] = {}
    for timeline in build_timelines(dataset, dataset.observed_until, category=dataset.purchase.category):
        by_day.setdefault(timeline.placed_at, {})[timeline.supplier] = timeline
    for day, members in sorted(by_day.items()):
        if len(members) != len(dataset.suppliers):
            continue
        if any(timeline.observed_age < horizon and timeline.completion_day(timeline.quantity) is None for timeline in members.values()):
            continue
        if pair is None:
            continue
        left, right = pair["suppliers"]
        left_day = members[left].completion_day(members[left].quantity)
        right_day = members[right].completion_day(members[right].quantity)
        other_days = [
            members[supplier].completion_day(members[supplier].quantity)
            for supplier in members
            if supplier not in {left, right}
        ]
        if (
            left_day is not None
            and right_day is not None
            and left_day > horizon
            and right_day > horizon
            and any(day is not None and day <= horizon for day in other_days)
        ):
            late_cohort = {
                "placed_at": day.isoformat(),
                "suppliers": {
                    supplier: {
                        "completion_lead_days": members[supplier].completion_day(members[supplier].quantity),
                        "delivered_by_purchase_deadline": members[supplier].delivered_quantity(horizon),
                        "ordered_units": members[supplier].quantity,
                    }
                    for supplier in sorted(members)
                },
                "explanation": (
                    "同一天下单，两家共用集货线路的订单都晚于采购截止日；"
                    "另一条线路的供应商按期交付。这正是“事后才知道共用线路”的形态。"
                ),
            }
            break
    return {
        "pair": pair,
        "late_cohort": late_cohort,
        "interpretation": (
            "共同线路既抬高“两家都成功”的联合概率，也抬高“两家都失败”的联合概率；"
            "简单平均或独立乘积都会漏掉后一半。"
        ),
    }


def _unsupported_reservation(dataset: Dataset) -> dict[str, Any] | None:
    raw = dataset.unsupported_purchase
    if not raw:
        return None
    category = raw.get("category")
    quantity = raw.get("quantity")
    return {
        "status": "insufficient_support",
        "category": category,
        "quantity": quantity,
        "has_category_history": category in dataset.categories,
        "observed_categories": list(dataset.categories),
        "max_observed_contract_quantity": max((order.quantity for order in dataset.orders), default=0),
        "reasons": [
            f"历史资料中没有任何 {category!r} 品类的订单，无法估计该品类的交期分布。",
            f"请求数量 {quantity} 超过历史单笔合同最大 {max((order.quantity for order in dataset.orders), default=0)} 件，"
            "数量与交期之间的关系没有被观察过。",
            "没有该品类的供应商产能、最小起订量和共同线路信息，无法做方案比较。",
            "在这种情况下给出一个“成功概率”会把模型外推伪装成验证过的结论。",
        ],
        "statement": (
            "保留意见：该采购品类与规模没有历史覆盖，不能给出交齐概率或推荐方案。"
            "建议先做小批量试单、补齐品类与线路资料，或按最坏情况准备替代资源；"
            "在拿到可验证的观察之前，不应把任何模型输出当作承诺依据。"
        ),
        "what_would_change_the_answer": [
            "该品类至少若干笔已结案订单，且能看到合同承诺、签收与线路",
            "该品类的供应商产能与最小起订量",
            "是否存在与历史供应商共用的集货线路",
        ],
    }


def analyze_dataset(
    dataset: Dataset,
    *,
    commit_threshold: float = COMMIT_THRESHOLD,
    shortage_cost_per_unit: float = SHORTAGE_COST_PER_UNIT,
    disruption_cost_per_unit: float = DISRUPTION_COST_PER_UNIT,
    hedge_cost_per_unit: float = HEDGE_COST_PER_UNIT,
) -> dict[str, Any]:
    purchase = dataset.purchase
    horizon = purchase.horizon_days
    if purchase.category not in dataset.categories:
        # 未覆盖品类：只给保留意见，不给概率。
        return {
            "schema_version": 1,
            "fixture": dataset.source,
            "observed_until": dataset.observed_until.isoformat(),
            "purchase": _purchase_dict(purchase),
            "supported": False,
            "reservation": _unsupported_reservation(dataset)
            or {
                "status": "insufficient_support",
                "statement": f"品类 {purchase.category!r} 没有历史覆盖，不能给出成功概率。",
            },
            "assumptions": _global_assumptions(),
        }
    dependence = _dependence(dataset, horizon, max(order.quantity for order in dataset.orders))
    rho_map = {
        route: value if value is not None else 0.5
        for route, value in dependence["route_rho"].items()
    }
    plan_result = evaluate_plans(
        dataset,
        purchase,
        as_of=dataset.observed_until,
        rho_by_route=rho_map,
        shortage_cost_per_unit=shortage_cost_per_unit,
        simultaneous_disruption_cost_per_unit=disruption_cost_per_unit,
        hedging_cost_per_unit=hedge_cost_per_unit,
        commit_threshold=commit_threshold,
        min_training_rows=10,
    )
    evaluations = plan_result["evaluations"]
    rankings = _rule_rankings(evaluations)
    recommendation = _recommendation(
        dataset, purchase, evaluations, rankings,
        commit_threshold=commit_threshold,
    )
    recommendation["changes"] = _plan_changes(rankings)
    # 方案概率使用观察截止日收敛后的完整资料；验证部分只用当时可见信息。
    validation = forward_validation(
        dataset,
        purchase,
        decision_threshold=commit_threshold,
        shortage_cost=shortage_cost_per_unit,
        hedge_cost=hedge_cost_per_unit,
    )
    validation["baseline_comparison"] = baseline_comparison(validation)
    curves = {
        supplier: quantity_curve(
            build_timelines(dataset, dataset.observed_until, supplier=supplier, category=purchase.category),
            horizon,
            max(item.capacity for item in purchase.suppliers),
            supplier=supplier,
            category=purchase.category,
            pooled=False,
            max_observed_quantity=max(
                (
                    timeline.quantity
                    for timeline in build_timelines(
                        dataset, dataset.observed_until, supplier=supplier, category=purchase.category
                    )
                ),
                default=0,
            ),
        )
        for supplier in dataset.suppliers
    }
    key_curves = {}
    capacity = max(item.capacity for item in purchase.suppliers)
    for supplier, curve in curves.items():
        key_curves[supplier] = {
            "n_orders": curve.n_orders,
            "outcomes": curve.outcomes,
            "max_observed_quantity": curve.max_observed_quantity,
            "at_least": {
                str(quantity): _round(curve.at_least(quantity), 4)
                for quantity in sorted(set(KEY_QUANTITIES) | {purchase.quantity // 2})
                if quantity <= curve.max_quantity
            },
            "expected_delivered_at_capacity": _round(curve.expected_delivered(capacity), 3),
            "support": curve.support(max(supplier.capacity for supplier in purchase.suppliers)),
        }
    return {
        "schema_version": 1,
        "fixture": dataset.source,
        "observed_until": dataset.observed_until.isoformat(),
        "purchase": _purchase_dict(purchase),
        "supported": True,
        "methodology": {
            "question": "采购截止日之前能否交齐所需数量",
            "primary_estimator": "离散时间 Aalen-Johansen 竞争风险（终止与未交齐为竞争事件）",
            "comparison_baselines": [
                "Kaplan-Meier（把终止与未完成当删失）",
                "已结案比例（丢掉未结案）",
                "全部假定未交齐（把未结案当失败）",
            ],
            "validation": "按 fixture 给定的日期起点向前验证；只用 known_at <= 起点的事件训练",
            "not_an_order": True,
        },
        "data_quality": {
            "orders": len(dataset.orders),
            "suppliers": list(dataset.suppliers),
            "categories": list(dataset.categories),
            "routes": dataset.routes,
            "notes": list(dataset.quality_notes),
        },
        "promise_basis": _promise_basis(dataset),
        "suppliers": _supplier_profiles(dataset, horizon),
        "quantity_curves": key_curves,
        "dependence": {
            **dependence,
            "shared_route_example": _shared_route_example(dataset, dependence, horizon),
        },
        "validation": validation,
        "plans": {
            "feasibility": {
                "candidate_plans": len(feasible_allocations(purchase)),
                "minimum_order_quantity_enforced": True,
                "capacity_enforced": True,
                "supported_plan_count": sum(1 for row in evaluations if _fully_supported(row)),
                "suppliers_without_history": sorted(
                    supplier
                    for supplier, value in plan_result.get("supplier_support", {}).items()
                    if value == "no_history"
                ),
            },
            "reference_plans": _reference_plans(dataset, purchase, evaluations),
            "route_comparison": _plan_route_comparison(evaluations),
            "evaluations": [row.as_dict() for row in evaluations],
            "ranking_summary": rankings,
            "joint_vs_naive_gap": _ranking_note(rankings),
            "primary_recommendation": recommendation,
            "cost_parameters": {
                "shortage_cost_per_unit": shortage_cost_per_unit,
                "simultaneous_disruption_cost_per_unit": disruption_cost_per_unit,
                "hedge_cost_per_unit": hedge_cost_per_unit,
                "commit_threshold": commit_threshold,
            },
            "cohort_count": plan_result["cohort_count"],
        },
        "unclosed_examples": _unclosed_examples(dataset, horizon),
        "unsupported_purchase": _unsupported_reservation(dataset),
        "assumptions": _global_assumptions(),
        "reservations": _reservations(dataset, plan_result, recommendation),
    }


def _purchase_dict(purchase: Purchase) -> dict[str, Any]:
    return {
        "placed_at": purchase.placed_at.isoformat(),
        "deadline": purchase.deadline.isoformat(),
        "horizon_days": purchase.horizon_days,
        "quantity": purchase.quantity,
        "category": purchase.category,
        "suppliers": [
            {
                "id": supplier.supplier_id,
                "minimum": supplier.minimum,
                "capacity": supplier.capacity,
                "route": supplier.route,
            }
            for supplier in purchase.suppliers
        ],
    }


def _global_assumptions() -> list[str]:
    return [
        "事件只在 known_at 之后可见；改约和签收不能提前进入训练样本。",
        "只有 accepted=true 的双方接受改约才改变生效承诺；原承诺始终保留。",
        "终止是竞争事件，不等于送达。",
        "未结案是未完成观察；不足截止日观察的保留为删失，看满截止日仍未交齐的记为未交齐。",
        "同线路共同冲击来自历史匹配组；线路之间默认独立，这是假设。",
        "超过历史单笔观察数量的分配需要外推，不能当成已验证的成功概率。",
        "结果只用于辅助比较整单与拆单，不自动下单。",
    ]


def _reservations(
    dataset: Dataset,
    plan_result: Mapping[str, Any],
    recommendation: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    max_observed = max((order.quantity for order in dataset.orders), default=0)
    if plan_result["quantity"] > max_observed:
        notes.append(
            f"请求总量 {plan_result['quantity']} 件超过历史单笔最大 {max_observed} 件；"
            "按数量拆分的曲线在超过观察规模后是外推。"
        )
    suppliers_without_history = [
        supplier
        for supplier, value in plan_result.get("supplier_support", {}).items()
        if value == "no_history"
    ]
    if suppliers_without_history:
        notes.append(
            "以下供应商没有任何历史履约资料，涉及它们的方案不给概率："
            + "、".join(sorted(suppliers_without_history))
            + "。"
        )
    if not any(
        getattr(row, "empirical_full", None) is not None
        for row in plan_result.get("evaluations", [])
    ):
        notes.append("没有可用的匹配订单组，共同线路经验率不可用。")
    if recommendation.get("joint_full_probability") is not None and not recommendation.get("meets_threshold"):
        notes.append(
            "整批交齐概率低于承诺阈值；建议准备替代资源或分批承诺，而不是把点估计当作保证。"
        )
    notes.append("没有价格、产能占用和合同罚则数据，业务损失参数是显式假设，需采购主管替换。")
    return notes
