"""按承诺截止日估计“交齐数量”的生存分析。

估计对象是离散事件：订单在承诺截止日（``horizon`` 天）之前累计签收是否达到
目标数量。这里刻意区分四种观察：

* ``delivered``：在截止日前累计签收达到目标数量。
* ``terminated``：在截止日前终止。终止是“未送达”的竞争事件，不是送达。
* ``failed``：完整观察到截止日仍没有达到目标数量（包括截止日之后才交齐、
  以及过了截止日仍未结案）。这是对“截止日前是否交齐”这一问题的已观察失败，
  不是给订单贴上“迟到完成”的完整标签。
* ``censored``：观察时间还不足截止日，结局未知。它既不算成功也不算失败，
  只是提前离开风险集。

最后一种情况与 ``failed`` 的区别是“有没有看满截止日”，不是“有没有结案”。
所有输入都只使用 ``known_at <= as_of`` 的事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from .data import Dataset, Order

STATUS_DELIVERED = "delivered"
STATUS_TERMINATED = "terminated"
STATUS_FAILED = "failed"
STATUS_CENSORED = "censored"


@dataclass(frozen=True, slots=True)
class Timeline:
    """单笔订单在某个信息时点上的可见签收轨迹。"""

    order_id: str
    supplier: str
    route: str
    category: str
    placed_at: date
    quantity: int
    receipt_cum: tuple[tuple[int, int], ...]
    termination_day: int | None
    observed_age: int

    def observe(self, target: int, horizon: int) -> "Observation":
        if target < 1 or target > self.quantity:
            raise ValueError(f"{self.order_id} 无法观察数量 {target}（合同数量 {self.quantity}）")
        termination_day = self.termination_day
        for day, cumulative in self.receipt_cum:
            if termination_day is not None and day > termination_day:
                break
            if cumulative >= target:
                if day <= horizon:
                    return Observation(self, target, STATUS_DELIVERED, day)
                return Observation(self, target, STATUS_FAILED, horizon)
        if termination_day is not None:
            if termination_day <= horizon:
                return Observation(self, target, STATUS_TERMINATED, max(termination_day, 0))
            return Observation(self, target, STATUS_FAILED, horizon)
        if self.observed_age < horizon:
            return Observation(self, target, STATUS_CENSORED, max(self.observed_age, 0))
        return Observation(self, target, STATUS_FAILED, horizon)

    def completion_day(self, target: int) -> int | None:
        termination_day = self.termination_day
        for day, cumulative in self.receipt_cum:
            if termination_day is not None and day > termination_day:
                return None
            if cumulative >= target:
                return day
        return None

    def delivered_quantity(self, horizon: int) -> int:
        termination_day = self.termination_day
        delivered = 0
        for day, cumulative in self.receipt_cum:
            if termination_day is not None and day > termination_day:
                break
            if day <= horizon:
                delivered = cumulative
        return min(delivered, self.quantity)


@dataclass(frozen=True, slots=True)
class Observation:
    timeline: Timeline
    target: int
    status: str
    time: int

    @property
    def delivered(self) -> bool:
        return self.status == STATUS_DELIVERED


def build_timeline(order: Order, as_of: date) -> Timeline | None:
    if order.placed_at > as_of:
        return None
    if not order.visible_events(as_of):
        return None
    placed = order.placed_at
    termination = order.visible_termination(as_of)
    termination_day = (termination.at - placed).days if termination is not None else None
    cumulative = 0
    receipt_cum: list[tuple[int, int]] = []
    for receipt in order.visible_receipts(as_of):
        day = (receipt.at - placed).days
        if termination_day is not None and day > termination_day:
            continue
        cumulative += receipt.quantity or 0
        receipt_cum.append((day, cumulative))
    return Timeline(
        order_id=order.order_id,
        supplier=order.supplier,
        route=order.route,
        category=order.category,
        placed_at=placed,
        quantity=order.quantity,
        receipt_cum=tuple(receipt_cum),
        termination_day=termination_day,
        observed_age=max((as_of - placed).days, 0),
    )


def build_timelines(
    dataset: Dataset,
    as_of: date,
    *,
    supplier: str | None = None,
    category: str | None = None,
    placed_before: date | None = None,
) -> tuple[Timeline, ...]:
    result: list[Timeline] = []
    for order in dataset.orders:
        if supplier is not None and order.supplier != supplier:
            continue
        if category is not None and order.category != category:
            continue
        if placed_before is not None and order.placed_at >= placed_before:
            continue
        timeline = build_timeline(order, as_of)
        if timeline is not None:
            result.append(timeline)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SurvivalEstimate:
    completion: float
    failure: float
    open_at_horizon: float
    n: int
    delivered: int
    terminated: int
    failed: int
    censored: int

    @property
    def resolved(self) -> int:
        return self.delivered + self.terminated + self.failed


def outcomes(observations: Sequence[Observation], horizon: int) -> dict[str, int]:
    counts = {STATUS_DELIVERED: 0, STATUS_TERMINATED: 0, STATUS_FAILED: 0, STATUS_CENSORED: 0}
    for observation in observations:
        if observation.time > horizon:
            raise ValueError("观察时间超过截止日")
        counts[observation.status] += 1
    return counts


def aalen_johansen(observations: Sequence[Observation], horizon: int) -> SurvivalEstimate | None:
    """离散时间 Aalen-Johansen 累积发生率。

    ``terminated`` 与 ``failed`` 都是竞争事件；``censored`` 在离开风险集时不
    贡献事件。返回值 ``completion`` 是 P(截止日前累计签收达到目标数量)。
    """

    if not observations:
        return None
    counts = outcomes(observations, horizon)
    delivered = [0] * (horizon + 1)
    competing = [0] * (horizon + 1)
    censored = [0] * (horizon + 1)
    for observation in observations:
        t = observation.time
        if observation.status == STATUS_DELIVERED:
            delivered[t] += 1
        elif observation.status in (STATUS_TERMINATED, STATUS_FAILED):
            competing[t] += 1
        else:
            censored[t] += 1
    at_risk = len(observations)
    survival = 1.0
    cumulative = 0.0
    for t in range(horizon + 1):
        if at_risk <= 0:
            break
        d = delivered[t]
        c = competing[t]
        z = censored[t]
        if d or c:
            cumulative += survival * (d / at_risk)
            survival *= 1.0 - ((d + c) / at_risk)
        at_risk -= d + c + z
    return SurvivalEstimate(
        completion=cumulative,
        failure=1.0 - survival - cumulative,
        open_at_horizon=survival,
        n=len(observations),
        delivered=counts[STATUS_DELIVERED],
        terminated=counts[STATUS_TERMINATED],
        failed=counts[STATUS_FAILED],
        censored=counts[STATUS_CENSORED],
    )


def kaplan_meier(observations: Sequence[Observation], horizon: int) -> SurvivalEstimate | None:
    """对照基准：把终止和未完成都当作删失，只把“交齐”当作事件。

    这个估计会系统性高估截止日交齐概率，因为终止和已知失败被移出了分母。
    """

    if not observations:
        return None
    counts = outcomes(observations, horizon)
    delivered = [0] * (horizon + 1)
    leaving = [0] * (horizon + 1)
    for observation in observations:
        t = observation.time
        if observation.status == STATUS_DELIVERED:
            delivered[t] += 1
        else:
            leaving[t] += 1
    at_risk = len(observations)
    survival = 1.0
    for t in range(horizon + 1):
        if at_risk <= 0:
            break
        d = delivered[t]
        if d:
            survival *= 1.0 - d / at_risk
        at_risk -= d + leaving[t]
    completion = 1.0 - survival
    return SurvivalEstimate(
        completion=completion,
        failure=0.0,
        open_at_horizon=survival,
        n=len(observations),
        delivered=counts[STATUS_DELIVERED],
        terminated=counts[STATUS_TERMINATED],
        failed=counts[STATUS_FAILED],
        censored=counts[STATUS_CENSORED],
    )


def closed_case_rate(observations: Sequence[Observation], horizon: int) -> float | None:
    """对照基准：只在已结案订单里数交齐比例，把未结案和终止直接排除。"""

    resolved = [observation for observation in observations if observation.status != STATUS_CENSORED]
    if not resolved:
        return None
    return sum(1 for observation in resolved if observation.delivered) / len(resolved)


def all_orders_rate(observations: Sequence[Observation], horizon: int) -> float | None:
    """对照基准：分母包含未结案，未结案一律当作没有交齐。"""

    if not observations:
        return None
    return sum(1 for observation in observations if observation.delivered) / len(observations)


def closed_case_estimate(observations: Sequence[Observation], horizon: int) -> SurvivalEstimate | None:
    counts = outcomes(observations, horizon)
    rate = closed_case_rate(observations, horizon)
    if rate is None:
        return None
    return SurvivalEstimate(
        completion=rate,
        failure=1.0 - rate,
        open_at_horizon=0.0,
        n=len(observations),
        delivered=counts[STATUS_DELIVERED],
        terminated=counts[STATUS_TERMINATED],
        failed=counts[STATUS_FAILED],
        censored=counts[STATUS_CENSORED],
    )


def all_orders_estimate(observations: Sequence[Observation], horizon: int) -> SurvivalEstimate | None:
    counts = outcomes(observations, horizon)
    rate = all_orders_rate(observations, horizon)
    if rate is None:
        return None
    return SurvivalEstimate(
        completion=rate,
        failure=1.0 - rate,
        open_at_horizon=0.0,
        n=len(observations),
        delivered=counts[STATUS_DELIVERED],
        terminated=counts[STATUS_TERMINATED],
        failed=counts[STATUS_FAILED],
        censored=counts[STATUS_CENSORED],
    )


@dataclass(frozen=True, slots=True)
class QuantityCurve:
    """同一供应商在截止日前“至少交齐 k 件”的概率曲线。

    ``max_observed_quantity`` 是历史单笔合同数量覆盖到的上限。超过它时曲线
    按最后一个已观察点外推（``extrapolated=True``），这是假设而不是已验证的
    成功概率，报告中必须单独标注。
    """

    supplier: str
    category: str | None
    pooled: bool
    max_quantity: int
    max_observed_quantity: int
    horizons: tuple[int, ...]
    probabilities: tuple[float, ...]
    n_orders: int
    outcomes: dict[str, int]

    def at_least(self, quantity: int) -> float:
        if quantity < 1:
            return 1.0
        capped = min(quantity, self.max_quantity)
        return self.probabilities[capped - 1]

    def support(self, quantity: int) -> str:
        if quantity < 1:
            return "trivial"
        if quantity > self.max_observed_quantity:
            return "extrapolated_beyond_observed_scale"
        return "observed"

    def expected_delivered(self, quantity: int) -> float:
        if quantity < 1:
            return 0.0
        return sum(self.at_least(k) for k in range(1, quantity + 1))

    def marginal_distribution(self, quantity: int) -> tuple[float, ...]:
        """D 在 0..quantity 上的概率质量，满足 P(D>=k)=F(k)。"""

        curve = [self.at_least(k) for k in range(1, quantity + 1)]
        masses = [0.0] * (quantity + 1)
        previous = 1.0
        for k in range(1, quantity + 1):
            masses[k - 1] = previous - curve[k - 1]
            previous = curve[k - 1]
        masses[quantity] = previous
        return tuple(max(mass, 0.0) for mass in masses)

    def as_dict(self) -> dict[str, object]:
        return {
            "supplier": self.supplier,
            "category": self.category,
            "pooled_across_categories": self.pooled,
            "n_orders": self.n_orders,
            "outcomes": self.outcomes,
            "max_quantity": self.max_quantity,
            "probabilities": {
                str(k): round(probability, 6)
                for k, probability in enumerate(self.probabilities, start=1)
            },
        }


def quantity_curve(
    timelines: Sequence[Timeline],
    horizon: int,
    max_quantity: int,
    *,
    supplier: str,
    category: str | None,
    pooled: bool,
    max_observed_quantity: int | None = None,
    estimator=aalen_johansen,
) -> QuantityCurve:
    """估计 F(k)=P(截止日前至少交齐 k 件)。

    超过历史单笔观察数量时，按最后一个已观察点保持平坦（不外推增长、也不假装
    归零）；调用方必须通过 ``support()`` 标注这是假设，而不是已验证的概率。
    """

    observed_max = max_observed_quantity
    if observed_max is None:
        observed_max = max((timeline.quantity for timeline in timelines), default=0)
    raw: list[tuple[int, float]] = []
    tail: float | None = None
    for target in range(1, max_quantity + 1):
        observations = [
            timeline.observe(target, horizon)
            for timeline in timelines
            if timeline.quantity >= target
        ]
        if not observations:
            if tail is None:
                raise ValueError(
                    f"{supplier} 没有覆盖数量 {target} 的订单，无法生成曲线"
                )
            raw.append((target, tail))
            continue
        estimate = estimator(observations, horizon)
        value = estimate.completion if estimate else 0.0
        raw.append((target, value))
        if target <= observed_max:
            tail = value
    # P(D>=k) 必须随 k 非增；逐目标估计在删失模式下会略有抖动，取运行最小值。
    probabilities: list[float] = []
    running = 1.0
    for _, value in raw:
        running = min(running, value)
        probabilities.append(running)
    summary_target = min(max(observed_max, 1), max_quantity)
    summary = outcomes(
        [timeline.observe(summary_target, horizon) for timeline in timelines if timeline.quantity >= summary_target],
        horizon,
    )
    return QuantityCurve(
        supplier=supplier,
        category=category,
        pooled=pooled,
        max_quantity=max_quantity,
        max_observed_quantity=min(observed_max or max_quantity, max_quantity),
        horizons=tuple(range(1, max_quantity + 1)),
        probabilities=tuple(probabilities),
        n_orders=len(timelines),
        outcomes=summary,
    )


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
