"""共同线路与整批联合风险。

关键点：整批“全部交齐”的概率不是各自准时率的平均数，也不是简单乘积。
同一集货线路上的供应商共享运输冲击，失败会一起发生；不同线路之间的共同
冲击没有数据支持，默认按独立处理。这里给出三种可解释的假设：

* ``independence``：所有供应商相互独立（最简单，但会低估共同迟到）。
* ``common_route_copula``：同线路内用高斯 copula 估计正相关，线路之间独立。
* ``matched_cohort_empirical``：直接数同一天、同品类的匹配订单组，非参数但样本小。

所有相关性都来自 *历史匹配组*；没有匹配组时返回 ``None``，而不是假设一个
好看的数字。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from statistics import NormalDist
from typing import Sequence

from .data import Dataset
from .survival import Timeline, build_timelines

_STANDARD = NormalDist()
METHOD_INDEPENDENCE = "independence"
METHOD_COMMON_ROUTE = "common_route_copula"
METHOD_EMPIRICAL = "matched_cohort_empirical"
METHODS = (METHOD_INDEPENDENCE, METHOD_COMMON_ROUTE, METHOD_EMPIRICAL)


def norm_cdf(value: float) -> float:
    return _STANDARD.cdf(value)


def norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def inv_norm(probability: float) -> float:
    return _STANDARD.inv_cdf(min(max(probability, 1e-9), 1.0 - 1e-9))


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    if abs(rho) < 1e-9:
        return norm_cdf(a) * norm_cdf(b)
    rho = max(-0.999, min(0.999, rho))
    low = -10.0
    high = min(a, 10.0)
    if high <= low:
        return 0.0
    scale = math.sqrt(1.0 - rho * rho)

    def integrand(z: float) -> float:
        return norm_pdf(z) * norm_cdf((b - rho * z) / scale)

    steps = 400
    width = (high - low) / steps
    total = integrand(low) + integrand(high)
    for index in range(1, steps):
        total += (4.0 if index % 2 else 2.0) * integrand(low + index * width)
    return max(0.0, min(1.0, total * width / 3.0))


def _simulated_group_joint(probabilities: Sequence[float], rho: float, seed: int) -> float:
    """3 个及以上同线路供应商的等相关联合概率（固定种子，结果可复现）。"""

    rng = random.Random(seed)
    draws = 20000
    scale = math.sqrt(max(rho, 0.0))
    noise = math.sqrt(max(1.0 - rho, 0.0))
    successes = 0
    thresholds = [inv_norm(probability) for probability in probabilities]
    for _ in range(draws):
        common = rng.gauss(0.0, 1.0) * scale
        ok = True
        for threshold in thresholds:
            value = common + rng.gauss(0.0, 1.0) * noise
            if value < threshold:
                ok = False
                break
        successes += 1 if ok else 0
    return successes / draws


def joint_all(
    probabilities: dict[str, float],
    routes: dict[str, str],
    rho_by_route: dict[str, float | None],
    method: str,
) -> float | None:
    """P(所有供应商都达标)。"""

    if not probabilities:
        return 1.0
    if method == METHOD_INDEPENDENCE:
        result = 1.0
        for probability in probabilities.values():
            result *= probability
        return result
    if method != METHOD_COMMON_ROUTE:
        raise ValueError(f"joint_all 不支持方法 {method!r}")
    groups: dict[str, list[str]] = {}
    for supplier in probabilities:
        groups.setdefault(routes[supplier], []).append(supplier)
    result = 1.0
    for route, members in sorted(groups.items()):
        values = [probabilities[supplier] for supplier in members]
        if len(values) == 1:
            result *= values[0]
            continue
        rho = rho_by_route.get(route)
        if rho is None:
            return None
        if len(values) == 2:
            result *= bivariate_normal_cdf(inv_norm(values[0]), inv_norm(values[1]), rho)
        else:
            result *= _simulated_group_joint(values, rho, seed=20261113 + len(values))
    return result


def joint_both_late(
    probabilities: dict[str, float],
    routes: dict[str, str],
    rho_by_route: dict[str, float | None],
    method: str,
) -> float | None:
    """P(所有供应商都未达标)，即整批同时短缺。"""

    if not probabilities:
        return 0.0
    if method == METHOD_INDEPENDENCE:
        result = 1.0
        for probability in probabilities.values():
            result *= 1.0 - probability
        return result
    if method != METHOD_COMMON_ROUTE:
        raise ValueError(f"joint_both_late 不支持方法 {method!r}")
    flipped = {supplier: 1.0 - probability for supplier, probability in probabilities.items()}
    return joint_all(flipped, routes, rho_by_route, METHOD_COMMON_ROUTE)


def rho_uncertainty(route: str, rho: float | None, cohorts: int) -> dict[str, object]:
    """对相关系数的可靠性给出明确标注，而不是只给一个点值。"""

    if rho is None:
        return {"route": route, "rho": None, "reliable": False, "note": "没有可用的匹配组"}
    notes: list[str] = []
    reliable = cohorts >= 10
    if cohorts < 10:
        notes.append(f"匹配组只有 {cohorts} 组，相关系数不稳定。")
    if abs(rho) >= 0.9:
        notes.append(
            "估计值接近共同单调边界（rho≈1）：这意味着历史数据几乎无法区分"
            "“同一冲击”和“完全正相关”，尾部同时失败的风险可能仍被低估。"
        )
    if not notes:
        notes.append(f"匹配组 {cohorts} 组；相关系数来自联合结果拟合，仍有抽样误差。")
    return {
        "route": route,
        "rho": round(rho, 4),
        "matched_cohorts": cohorts,
        "reliable": reliable,
        "near_comonotonic_boundary": abs(rho) >= 0.9,
        "note": " ".join(notes),
    }


@dataclass(frozen=True, slots=True)
class PairDependence:
    supplier_a: str
    supplier_b: str
    route_a: str
    route_b: str
    same_route: bool
    cohorts: int
    both_on_time: int
    first_only_on_time: int
    second_only_on_time: int
    both_short: int
    lead_correlation: float | None
    rho: float | None
    both_short_independence: float | None
    both_short_empirical: float | None
    both_short_copula: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "suppliers": [self.supplier_a, self.supplier_b],
            "routes": [self.route_a, self.route_b],
            "same_route": self.same_route,
            "matched_cohorts": self.cohorts,
            "both_on_time": self.both_on_time,
            "first_only_on_time": self.first_only_on_time,
            "second_only_on_time": self.second_only_on_time,
            "both_short": self.both_short,
            "lead_correlation": None if self.lead_correlation is None else round(self.lead_correlation, 4),
            "rho": None if self.rho is None else round(self.rho, 4),
            "both_short_rate_independence": None if self.both_short_independence is None else round(self.both_short_independence, 4),
            "both_short_rate_empirical": None if self.both_short_empirical is None else round(self.both_short_empirical, 4),
            "both_short_rate_copula": None if self.both_short_copula is None else round(self.both_short_copula, 4),
        }


def _lead_days(timeline: Timeline, target: int) -> int | None:
    return timeline.completion_day(target)


def matched_pair_dependence(
    dataset: Dataset,
    category: str,
    horizon: int,
    target: int,
    as_of: date,
) -> list[PairDependence]:
    """按同一天、同品类的匹配订单组估计两两线路效应。"""

    by_day: dict[date, dict[str, Timeline]] = {}
    for timeline in build_timelines(dataset, as_of, category=category):
        if timeline.quantity < target:
            continue
        by_day.setdefault(timeline.placed_at, {})[timeline.supplier] = timeline
    pairs: list[PairDependence] = []
    suppliers = dataset.suppliers
    for index, supplier_a in enumerate(suppliers):
        for supplier_b in suppliers[index + 1 :]:
            rows: list[tuple[Timeline, Timeline]] = []
            for members in by_day.values():
                if supplier_a not in members or supplier_b not in members:
                    continue
                left, right = members[supplier_a], members[supplier_b]
                # 结局必须可判定：已交齐，或者已经看满截止日。
                if left.observed_age < horizon and left.completion_day(target) is None:
                    continue
                if right.observed_age < horizon and right.completion_day(target) is None:
                    continue
                rows.append((left, right))
            route_a = dataset.routes[supplier_a]
            route_b = dataset.routes[supplier_b]
            if not rows:
                pairs.append(
                    PairDependence(
                        supplier_a, supplier_b, route_a, route_b,
                        route_a == route_b, 0, 0, 0, 0, 0,
                        None, None, None, None, None,
                    )
                )
                continue
            both = first = second = neither = 0
            leads: list[tuple[int, int]] = []
            for left, right in rows:
                left_lead = _lead_days(left, target)
                right_lead = _lead_days(right, target)
                left_ok = left_lead is not None and left_lead <= horizon
                right_ok = right_lead is not None and right_lead <= horizon
                # 相关系数用所有已交齐订单的实际完成时长，而不是只看截止日前成功的。
                if left_lead is not None and right_lead is not None:
                    leads.append((left_lead, right_lead))
                if left_ok and right_ok:
                    both += 1
                elif left_ok:
                    first += 1
                elif right_ok:
                    second += 1
                else:
                    neither += 1
            n = len(rows)
            p_left = (both + first) / n
            p_right = (both + second) / n
            both_rate = both / n
            independence = p_left * p_right
            both_short_independence = (1.0 - p_left) * (1.0 - p_right)
            both_short_empirical = neither / n
            correlation = None
            if len(leads) >= 4:
                xs = [pair[0] for pair in leads]
                ys = [pair[1] for pair in leads]
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)
                variance_x = sum((value - mean_x) ** 2 for value in xs)
                variance_y = sum((value - mean_y) ** 2 for value in ys)
                if variance_x > 0 and variance_y > 0:
                    covariance = sum((x - mean_x) * (y - mean_y) for x, y in leads)
                    correlation = covariance / math.sqrt(variance_x * variance_y)
            rho = None
            if 0 < p_left < 1 and 0 < p_right < 1:
                best: tuple[float, float] | None = None
                for step in range(-95, 96):
                    candidate = step / 100.0
                    predicted = bivariate_normal_cdf(inv_norm(p_left), inv_norm(p_right), candidate)
                    error = abs(predicted - both_rate)
                    if best is None or error < best[0]:
                        best = (error, candidate)
                rho = best[1] if best is not None else None
            both_short_copula = None
            if rho is not None:
                copula_joint = bivariate_normal_cdf(inv_norm(p_left), inv_norm(p_right), rho)
                both_short_copula = max(0.0, 1.0 - p_left - p_right + copula_joint)
            pairs.append(
                PairDependence(
                    supplier_a=supplier_a,
                    supplier_b=supplier_b,
                    route_a=route_a,
                    route_b=route_b,
                    same_route=route_a == route_b,
                    cohorts=n,
                    both_on_time=both,
                    first_only_on_time=first,
                    second_only_on_time=second,
                    both_short=neither,
                    lead_correlation=correlation,
                    rho=rho,
                    both_short_independence=both_short_independence,
                    both_short_empirical=both_short_empirical,
                    both_short_copula=both_short_copula,
                )
            )
    return pairs


def route_rho(
    pairs: Sequence[PairDependence],
    min_cohorts: int = 10,
    default: float | None = 0.5,
) -> dict[str, float | None]:
    """每条线路的共同冲击相关系数。

    同一线路内取该线路所有供应商对的 rho 中位数；样本不足时退回
    ``default``，并在报告中标注为假设值而不是估计值。
    """

    grouped: dict[str, list[float]] = {}
    route_members: dict[str, set[str]] = {}
    for pair in pairs:
        route_members.setdefault(pair.route_a, set()).add(pair.supplier_a)
        route_members.setdefault(pair.route_b, set()).add(pair.supplier_b)
        if pair.same_route and pair.cohorts >= min_cohorts and pair.rho is not None:
            grouped.setdefault(pair.route_a, []).append(pair.rho)
    result: dict[str, float | None] = {}
    for route, members in route_members.items():
        if len(members) < 2:
            result[route] = 0.0
            continue
        values = grouped.get(route, [])
        if values:
            ordered = sorted(values)
            result[route] = ordered[len(ordered) // 2]
        else:
            result[route] = default
    return result


def rho_source(
    pairs: Sequence[PairDependence],
    min_cohorts: int = 10,
) -> dict[str, str]:
    grouped: dict[str, int] = {}
    for pair in pairs:
        if pair.same_route and pair.cohorts >= min_cohorts and pair.rho is not None:
            grouped[pair.route_a] = grouped.get(pair.route_a, 0) + 1
    return {
        route: ("estimated_from_matched_cohorts" if count else "assumption")
        for route, count in grouped.items()
    }


def route_rho_detail(
    pairs: Sequence[PairDependence],
    rho_by_route: Mapping[str, float | None],
) -> list[dict[str, object]]:
    """每个线路的 rho、样本量与可靠性标注。"""

    details: list[dict[str, object]] = []
    routes = sorted({pair.route_a for pair in pairs} | {pair.route_b for pair in pairs})
    for route in routes:
        members = {pair.supplier_a for pair in pairs if pair.route_a == route} | {
            pair.supplier_b for pair in pairs if pair.route_b == route
        }
        same_route_pairs = [
            pair
            for pair in pairs
            if pair.same_route and pair.route_a == route and pair.rho is not None
        ]
        cohorts = max((pair.cohorts for pair in same_route_pairs), default=0)
        if len(members) < 2:
            details.append(
                {
                    "route": route,
                    "rho": 0.0,
                    "matched_cohorts": cohorts,
                    "suppliers": sorted(members),
                    "source": "not_applicable_single_supplier",
                    "reliable": True,
                    "near_comonotonic_boundary": False,
                    "note": "该线路只有一个供应商，线路内共同冲击不适用。",
                }
            )
            continue
        entry = rho_uncertainty(route, rho_by_route.get(route), cohorts)
        entry["suppliers"] = sorted(members)
        entry["source"] = (
            "estimated_from_matched_cohorts" if same_route_pairs else "assumption"
        )
        details.append(entry)
    return details
