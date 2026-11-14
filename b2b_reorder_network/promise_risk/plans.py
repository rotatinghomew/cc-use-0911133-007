"""候选供应方案与整批交齐风险。

候选方案受最小起订量和产能约束。对每个方案同时给出：

* 每个供应商按自身分配数量在截止日前交齐的概率；
* 整批全部交齐的概率（独立、共同线路 copula、匹配组经验三种口径）；
* 共同线路同时短缺的概率；
* 截止日前期望交付数量与期望短缺；
* 业务损失与历史覆盖保留意见。

整批概率不是各供应商概率的平均，也不是简单乘积：同线路供应商共享运输
冲击时，必须使用联合分布。``independence`` 只作为对照假设，不是默认结论。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

from .data import Dataset, Purchase
from .dependence import (
    METHOD_COMMON_ROUTE,
    METHOD_INDEPENDENCE,
    bivariate_normal_cdf,
    inv_norm,
    joint_both_late,
)
from .survival import QuantityCurve, Timeline, build_timelines, quantity_curve


METHOD_COPULA = METHOD_COMMON_ROUTE


def _group_joint_meet(
    probabilities: Sequence[float],
    rho: float,
    seed: int = 20261113,
) -> tuple[float, float]:
    """同线路“全部达标”和“全部短缺”的联合概率。

    两个供应商用二元正态 copula 的闭式累计概率；三个及以上用固定种子的
    等相关模拟（结果可复现）。``probabilities`` 是每个供应商达标的边缘概率。
    """

    if len(probabilities) == 1:
        meet = probabilities[0]
        return meet, 1.0 - meet
    if len(probabilities) == 2:
        short_left = 1.0 - probabilities[0]
        short_right = 1.0 - probabilities[1]
        meet = max(
            0.0,
            min(
                1.0,
                1.0
                - short_left
                - short_right
                + bivariate_normal_cdf(inv_norm(short_left), inv_norm(short_right), rho),
            ),
        )
        both_short = bivariate_normal_cdf(inv_norm(short_left), inv_norm(short_right), rho)
        return meet, max(0.0, min(1.0, both_short))
    rng = random.Random(seed)
    draws = 6000
    scale = math.sqrt(max(rho, 0.0))
    noise = math.sqrt(max(1.0 - rho, 0.0))
    thresholds = [inv_norm(probability) for probability in probabilities]
    meet_count = short_count = 0
    for _ in range(draws):
        common = rng.gauss(0.0, 1.0) * scale
        successes = []
        for threshold in thresholds:
            value = common + rng.gauss(0.0, 1.0) * noise
            successes.append(value >= threshold)
        if all(successes):
            meet_count += 1
        if not any(successes):
            short_count += 1
    return meet_count / draws, short_count / draws


@dataclass(frozen=True, slots=True)
class RouteOutcome:
    route: str
    members: tuple[str, ...]
    joint_meet_probability: float
    all_short_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "members": list(self.members),
            "joint_meet_probability": round(self.joint_meet_probability, 6),
            "all_short_probability": round(self.all_short_probability, 6),
        }


@dataclass(frozen=True, slots=True)
class PlanRisk:
    allocations: dict[str, int]
    supplier_probabilities: dict[str, float]
    supplier_support: dict[str, str]
    routes: dict[str, str]
    joint_probability: float | None
    any_short_probability: float | None
    both_short_probability: float | None
    expected_delivered: float | None
    expected_shortfall: float | None
    shortage_loss: float | None
    disruption_loss: float | None
    hedging_loss: float | None
    total_loss: float | None
    naive_average_probability: float
    product_probability: float
    route_outcomes: tuple[RouteOutcome, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "allocations": dict(sorted(self.allocations.items())),
            "supplier_probabilities": {
                key: round(value, 6) for key, value in sorted(self.supplier_probabilities.items())
            },
            "supplier_support": dict(sorted(self.supplier_support.items())),
            "routes": dict(sorted(self.routes.items())),
            "joint_full_probability": None if self.joint_probability is None else round(self.joint_probability, 6),
            "any_short_probability": None if self.any_short_probability is None else round(self.any_short_probability, 6),
            "both_short_probability": None if self.both_short_probability is None else round(self.both_short_probability, 6),
            "expected_delivered_units": None if self.expected_delivered is None else round(self.expected_delivered, 3),
            "expected_shortfall_units": None if self.expected_shortfall is None else round(self.expected_shortfall, 3),
            "shortage_loss": None if self.shortage_loss is None else round(self.shortage_loss, 2),
            "disruption_loss": None if self.disruption_loss is None else round(self.disruption_loss, 2),
            "hedging_loss": None if self.hedging_loss is None else round(self.hedging_loss, 2),
            "total_loss": None if self.total_loss is None else round(self.total_loss, 2),
            "naive_average_probability": round(self.naive_average_probability, 6),
            "independent_product_probability": round(self.product_probability, 6),
            "route_outcomes": [item.as_dict() for item in self.route_outcomes],
        }


@dataclass(frozen=True, slots=True)
class PlanEvaluation:
    plan: PlanRisk
    joint_independence: float | None
    both_short_independence: float | None
    empirical_full: float | None
    empirical_shortfall: float | None
    empirical_cohorts: int
    empirical_expected_delivered: float | None
    empirical_supported: bool

    def as_dict(self) -> dict[str, object]:
        return {
            **self.plan.as_dict(),
            "assumptions": {
                "independence": {
                    "joint_full_probability": None if self.joint_independence is None else round(self.joint_independence, 6),
                    "both_short_probability": None if self.both_short_independence is None else round(self.both_short_independence, 6),
                },
                "matched_cohorts": {
                    "joint_full_rate": None if self.empirical_full is None else round(self.empirical_full, 6),
                    "expected_shortfall_units": None if self.empirical_shortfall is None else round(self.empirical_shortfall, 3),
                    "expected_delivered_units": None if self.empirical_expected_delivered is None else round(self.empirical_expected_delivered, 3),
                    "cohorts": self.empirical_cohorts,
                    "supported": self.empirical_supported,
                },
            },
        }


def feasible_allocations(purchase: Purchase) -> list[dict[str, int]]:
    suppliers = purchase.suppliers
    results: list[dict[str, int]] = []

    def walk(index: int, remaining: int, current: dict[str, int]) -> None:
        if index == len(suppliers):
            if remaining == 0:
                results.append({key: value for key, value in sorted(current.items()) if value > 0})
            return
        supplier = suppliers[index]
        upper = min(supplier.capacity, remaining)
        for quantity in range(0, upper + 1):
            if quantity != 0 and quantity < supplier.minimum:
                continue
            current[supplier.supplier_id] = quantity
            walk(index + 1, remaining - quantity, current)
        current.pop(supplier.supplier_id, None)

    walk(0, purchase.quantity, {})
    return results


def _curves_by_supplier(
    dataset: Dataset,
    as_of: date,
    category: str,
    horizon: int,
    max_quantity: int,
    min_rows: int,
) -> tuple[dict[str, QuantityCurve], dict[str, str]]:
    curves: dict[str, QuantityCurve] = {}
    support: dict[str, str] = {}
    for supplier in dataset.suppliers:
        category_timelines = build_timelines(dataset, as_of, supplier=supplier, category=category)
        timelines = category_timelines
        pooled = False
        if len(timelines) < min_rows:
            timelines = build_timelines(dataset, as_of, supplier=supplier)
            pooled = True
        if not timelines:
            support[supplier] = "no_history"
            continue
        observed_max = max(
            (timeline.quantity for timeline in timelines),
            default=0,
        )
        curve = quantity_curve(
            timelines,
            horizon,
            max_quantity,
            supplier=supplier,
            category=category,
            pooled=pooled,
            max_observed_quantity=observed_max,
        )
        curves[supplier] = curve
        if not category_timelines:
            support[supplier] = "category_not_covered_pooled_other_categories"
        elif observed_max < max_quantity:
            support[supplier] = "partial_quantity_extrapolated"
        else:
            support[supplier] = "observed"
    return curves, support


def _group_by_route(
    allocations: Mapping[str, int],
    routes: Mapping[str, str],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for supplier in allocations:
        groups.setdefault(routes[supplier], []).append(supplier)
    return groups


def _route_outcomes(
    allocations: Mapping[str, int],
    curves: Mapping[str, QuantityCurve],
    routes: Mapping[str, str],
    rho_by_route: Mapping[str, float | None],
    method: str,
) -> tuple[list[RouteOutcome], float | None]:
    """按线路计算“本线路全部达标”与“本线路全部短缺”的概率。

    期望交付量 = Σ_s E[min(D_s, a_s)]，只需边缘分布；联合概率只在同线路
    内计算，线路之间按独立卷积。
    """

    groups = _group_by_route(allocations, routes)
    outcomes: list[RouteOutcome] = []
    joint = 1.0
    for route, members in sorted(groups.items()):
        members = sorted(members)
        probabilities = [curves[supplier].at_least(allocations[supplier]) for supplier in members]
        if len(members) == 1:
            meet = probabilities[0]
            all_short = 1.0 - meet
        else:
            rho = 0.0 if method == METHOD_INDEPENDENCE else rho_by_route.get(route)
            if rho is None:
                return outcomes, None
            meet, all_short = _group_joint_meet(probabilities, rho)
        outcomes.append(
            RouteOutcome(
                route=route,
                members=tuple(members),
                joint_meet_probability=meet,
                all_short_probability=all_short,
            )
        )
        joint *= meet
    return outcomes, joint


def _expected_delivered(
    allocations: Mapping[str, int],
    curves: Mapping[str, QuantityCurve],
) -> float:
    total = 0.0
    for supplier, quantity in allocations.items():
        total += curves[supplier].expected_delivered(quantity)
    return total


@dataclass(frozen=True, slots=True)
class CohortRecord:
    """同一天内各供应商的截止日前交付量；``None`` 表示观察不足、不可判定。"""

    placed_at: date
    delivered: dict[str, int | None]


def cohort_records(
    dataset: Dataset,
    category: str,
    horizon: int,
    as_of: date,
    target_quantity: int,
) -> list[CohortRecord]:
    by_day: dict[date, dict[str, Timeline]] = {}
    for timeline in build_timelines(dataset, as_of, category=category):
        if timeline.quantity < target_quantity:
            continue
        by_day.setdefault(timeline.placed_at, {})[timeline.supplier] = timeline
    records: list[CohortRecord] = []
    for placed_at, members in sorted(by_day.items()):
        delivered: dict[str, int | None] = {}
        for supplier, timeline in members.items():
            readable = (
                timeline.observed_age >= horizon
                or timeline.completion_day(timeline.quantity) is not None
            )
            delivered[supplier] = timeline.delivered_quantity(horizon) if readable else None
        records.append(CohortRecord(placed_at=placed_at, delivered=delivered))
    return records


def empirical_plan_metrics(
    records: Sequence[CohortRecord],
    allocations: Mapping[str, int],
) -> tuple[float | None, float | None, float | None, int, bool]:
    """在匹配组上直接数该方案的整批交齐率与期望短缺。

    返回 ``(full_rate, shortfall, expected, cohorts, supported)``。如果某个供应商
    的分配数量超过该供应商在匹配组里出现过的最大交付量，经验率不可识别，返回
    ``None`` 并把 ``supported`` 置为假，避免把“历史上从未交过这么多”误读成 0%。
    """

    if not allocations:
        return None, None, None, 0, False
    eligible = [
        record
        for record in records
        if all(record.delivered.get(supplier) is not None for supplier in allocations)
    ]
    if not eligible:
        return None, None, None, 0, False
    max_delivered = {
        supplier: max(record.delivered[supplier] for record in eligible)
        for supplier in allocations
    }
    supported = all(
        quantity <= max_delivered[supplier] for supplier, quantity in allocations.items()
    )
    if not supported:
        return None, None, None, len(eligible), False
    full = sum(
        1
        for record in eligible
        if all(record.delivered[supplier] >= quantity for supplier, quantity in allocations.items())
    ) / len(eligible)
    shortfall = sum(
        sum(
            max(quantity - record.delivered[supplier], 0)
            for supplier, quantity in allocations.items()
        )
        for record in eligible
    ) / len(eligible)
    expected = sum(
        sum(
            min(quantity, record.delivered[supplier])
            for supplier, quantity in allocations.items()
        )
        for record in eligible
    ) / len(eligible)
    return full, shortfall, expected, len(eligible), True


def evaluate_plans(
    dataset: Dataset,
    purchase: Purchase,
    *,
    as_of: date | None = None,
    rho_by_route: Mapping[str, float | None] | None = None,
    shortage_cost_per_unit: float = 20.0,
    simultaneous_disruption_cost_per_unit: float = 8.0,
    hedging_cost_per_unit: float = 6.0,
    commit_threshold: float = 0.8,
    min_training_rows: int = 10,
) -> dict[str, object]:
    as_of = as_of or dataset.observed_until
    horizon = purchase.horizon_days
    max_quantity = max(supplier.capacity for supplier in purchase.suppliers)
    curves, support = _curves_by_supplier(
        dataset, as_of, purchase.category, horizon, max_quantity, min_training_rows
    )
    routes = {supplier.supplier_id: supplier.route for supplier in purchase.suppliers}
    rho_by_route = dict(rho_by_route or {})
    target_quantity = max((order.quantity for order in dataset.orders), default=0)
    cohorts = cohort_records(dataset, purchase.category, horizon, as_of, target_quantity)
    evaluations: list[PlanEvaluation] = []
    for allocations in feasible_allocations(purchase):
        # 没有任何历史曲线的供应商：方案无法估计，直接标注为不支持，而不是让整批崩溃。
        unsupported_suppliers = [supplier for supplier in allocations if supplier not in curves]
        supplier_probabilities = {
            supplier: curves[supplier].at_least(quantity)
            for supplier, quantity in allocations.items()
            if supplier in curves
        }
        naive_average = (
            sum(supplier_probabilities.values()) / len(supplier_probabilities)
            if supplier_probabilities
            else 0.0
        )
        if unsupported_suppliers:
            plan = PlanRisk(
                allocations=dict(allocations),
                supplier_probabilities=supplier_probabilities,
                supplier_support={supplier: "no_history" for supplier in unsupported_suppliers},
                routes=routes,
                joint_probability=None,
                any_short_probability=None,
                both_short_probability=None,
                expected_delivered=None,
                expected_shortfall=None,
                shortage_loss=None,
                disruption_loss=None,
                hedging_loss=None,
                total_loss=None,
                naive_average_probability=naive_average,
                product_probability=0.0,
                route_outcomes=(),
            )
            evaluations.append(
                PlanEvaluation(
                    plan=plan,
                    joint_independence=None,
                    both_short_independence=None,
                    empirical_full=None,
                    empirical_shortfall=None,
                    empirical_cohorts=0,
                    empirical_expected_delivered=None,
                    empirical_supported=False,
                )
            )
            continue
        route_outcomes_ind, joint_ind = _route_outcomes(
            allocations, curves, routes, rho_by_route, METHOD_INDEPENDENCE
        )
        route_outcomes_cop, joint_cop = _route_outcomes(
            allocations, curves, routes, rho_by_route, METHOD_COPULA
        )
        both_short_ind = joint_both_late(
            supplier_probabilities, routes, rho_by_route, METHOD_INDEPENDENCE
        )
        both_short_cop = joint_both_late(
            supplier_probabilities, routes, rho_by_route, METHOD_COPULA
        )
        any_short_cop = (
            None if joint_cop is None else max(0.0, min(1.0, 1.0 - joint_cop))
        )
        expected_delivered = expected_shortfall = shortage_loss = None
        disruption_loss = hedging_loss = total_loss = None
        if joint_cop is not None:
            expected_delivered = _expected_delivered(allocations, curves)
            expected_shortfall = max(purchase.quantity - expected_delivered, 0.0)
            shortage_loss = expected_shortfall * shortage_cost_per_unit
            shared_routes = {
                route
                for route, members in _group_by_route(allocations, routes).items()
                if len(members) >= 2
            }
            shared_volume = sum(
                allocations[supplier]
                for supplier in allocations
                if routes[supplier] in shared_routes
            )
            shared_all_short = 1.0
            for outcome in route_outcomes_cop:
                if outcome.route in shared_routes:
                    shared_all_short *= outcome.all_short_probability
            disruption_loss = (
                simultaneous_disruption_cost_per_unit * shared_volume * shared_all_short
            )
            hedging_loss = 0.0 if joint_cop >= commit_threshold else hedging_cost_per_unit * purchase.quantity
            total_loss = shortage_loss + disruption_loss + hedging_loss
        empirical_full, empirical_shortfall, empirical_expected, empirical_n, empirical_supported = empirical_plan_metrics(
            cohorts, allocations
        )
        plan = PlanRisk(
            allocations=dict(allocations),
            supplier_probabilities=supplier_probabilities,
            supplier_support={
                supplier: curves[supplier].support(quantity)
                for supplier, quantity in allocations.items()
                if supplier in curves
            },
            routes=routes,
            joint_probability=joint_cop,
            any_short_probability=any_short_cop,
            both_short_probability=both_short_cop,
            expected_delivered=expected_delivered,
            expected_shortfall=expected_shortfall,
            shortage_loss=shortage_loss,
            disruption_loss=disruption_loss,
            hedging_loss=hedging_loss,
            total_loss=total_loss,
            naive_average_probability=naive_average,
            product_probability=math.prod(supplier_probabilities.values()),
            route_outcomes=tuple(route_outcomes_cop),
        )
        evaluations.append(
            PlanEvaluation(
                plan=plan,
                joint_independence=joint_ind,
                both_short_independence=both_short_ind,
                empirical_full=empirical_full,
                empirical_shortfall=empirical_shortfall,
                empirical_cohorts=empirical_n,
                empirical_expected_delivered=empirical_expected,
                empirical_supported=empirical_supported,
            )
        )
    return {
        "as_of": as_of.isoformat(),
        "horizon_days": horizon,
        "quantity": purchase.quantity,
        "category": purchase.category,
        "route_rho": {
            route: (None if value is None else round(value, 4))
            for route, value in rho_by_route.items()
        },
        "supplier_curves": {supplier: curve.as_dict() for supplier, curve in curves.items()},
        "supplier_support": support,
        "supported_category": purchase.category in dataset.categories,
        "evaluations": evaluations,
        "cohort_count": sum(1 for record in cohorts if all(value is not None for value in record.delivered.values())),
        "assumptions": {
            "minimum_order_quantity_enforced": True,
            "capacity_enforced": True,
            "joint_full_probability_is_not_average_or_product": True,
            "same_route_dependence_from_matched_cohorts": True,
            "cross_route_independence_assumed": True,
            "not_an_order_placement": True,
        },
    }
