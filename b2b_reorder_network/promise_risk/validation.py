"""按日期向前验证、校准、模型选择与简单基准比较。

规则：

* 每个验证起点只用 ``known_at <= origin`` 的数据训练，绝不使用改约/签收未来的
  入账信息。
* 验证订单的最终标签只用 ``observed_until`` 之前可见的签收，并且要求截止日
  已经被完整观察到；看不到满截止日的样本不进入标签。
* 终止、已过截止日未结案都属于“截止日前未交齐”，不是迟到完成。
* 基准包括“已结案比例”“全样本一律未交齐”和简单 Kaplan-Meier（把终止当删失）。
* 模型选择发生在验证点上，因此报告里的验证误差对“被选中的模型”是偏乐观的；
  这一点在结果中明确写出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from .data import Dataset, Purchase
from .survival import (
    Observation,
    STATUS_DELIVERED,
    Timeline,
    aalen_johansen,
    all_orders_estimate,
    build_timelines,
    closed_case_estimate,
    kaplan_meier,
)

MODEL_HAZARD = "aalen_johansen_competing_risks"
MODEL_BASELINE_KM = "kaplan_meier_termination_censored"
MODEL_BASELINE_CLOSED = "closed_case_rate"
MODEL_BASELINE_ALL = "all_orders_assumed_short"
MODELS = (MODEL_HAZARD, MODEL_BASELINE_KM, MODEL_BASELINE_CLOSED, MODEL_BASELINE_ALL)

_ESTIMATORS = {
    MODEL_HAZARD: aalen_johansen,
    MODEL_BASELINE_KM: kaplan_meier,
    MODEL_BASELINE_CLOSED: closed_case_estimate,
    MODEL_BASELINE_ALL: all_orders_estimate,
}


def observations_at(
    dataset: Dataset,
    as_of: date,
    supplier: str,
    category: str,
    target: int,
    horizon: int,
) -> tuple[Observation, ...]:
    timelines = build_timelines(dataset, as_of, supplier=supplier, category=category, placed_before=as_of)
    return tuple(
        timeline.observe(target, horizon)
        for timeline in timelines
        if timeline.quantity >= target
    )


def estimate(
    dataset: Dataset,
    as_of: date,
    supplier: str,
    category: str,
    target: int,
    horizon: int,
    model: str,
) -> float | None:
    observations = observations_at(dataset, as_of, supplier, category, target, horizon)
    if not observations:
        return None
    result = _ESTIMATORS[model](observations, horizon)
    return None if result is None else result.completion


def final_outcome(timeline: Timeline, target: int, horizon: int) -> int | None:
    """验证标签：``1`` 截止日前交齐、``0`` 截止日前未交齐、``None`` 观察不足。"""

    observation = timeline.observe(target, horizon)
    if observation.status == "censored":
        return None
    return 1 if observation.status == STATUS_DELIVERED else 0


@dataclass(frozen=True, slots=True)
class ValidationPoint:
    origin: date
    order_id: str
    supplier: str
    category: str
    target: int
    label: int
    predictions: dict[str, float | None]
    train_rows: int

    def as_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin.isoformat(),
            "order_id": self.order_id,
            "supplier": self.supplier,
            "category": self.category,
            "target": self.target,
            "actual_on_time": self.label,
            "predictions": {
                model: None if value is None else round(value, 6)
                for model, value in self.predictions.items()
            },
            "train_rows": self.train_rows,
        }


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    model: str
    points: int
    coverage: float
    brier: float | None
    log_loss: float | None
    calibration_error: float | None
    mean_prediction: float | None
    observed_rate: float | None
    expected_cost_per_order: float | None
    realized_cost_per_decision: float | None
    commit_rate: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "points": self.points,
            "coverage": round(self.coverage, 4),
            "brier": None if self.brier is None else round(self.brier, 6),
            "log_loss": None if self.log_loss is None else round(self.log_loss, 6),
            "calibration_error": None if self.calibration_error is None else round(self.calibration_error, 6),
            "mean_prediction": None if self.mean_prediction is None else round(self.mean_prediction, 6),
            "observed_rate": None if self.observed_rate is None else round(self.observed_rate, 6),
            "expected_cost_per_order": None if self.expected_cost_per_order is None else round(self.expected_cost_per_order, 3),
            "realized_cost_per_decision": None if self.realized_cost_per_decision is None else round(self.realized_cost_per_decision, 3),
            "commit_rate": None if self.commit_rate is None else round(self.commit_rate, 4),
        }


def _calibration_error(pairs: Sequence[tuple[float, int]], bins: int = 5) -> float | None:
    """经验校准误差（ECE）：分箱内平均预测与观测频率的加权差。

    验证点很少时 5 个箱会过稀；这里同时给出样本量，并在模型选择说明中提示
    该指标不稳定。"""

    if not pairs:
        return None
    buckets: dict[int, list[tuple[float, int]]] = {}
    for probability, label in pairs:
        index = min(bins - 1, int(probability * bins))
        buckets.setdefault(index, []).append((probability, label))
    total = 0.0
    for rows in buckets.values():
        mean_prediction = sum(row[0] for row in rows) / len(rows)
        mean_label = sum(row[1] for row in rows) / len(rows)
        total += len(rows) / len(pairs) * abs(mean_prediction - mean_label)
    return total


def decision_costs(
    probability: float,
    label: int,
    quantity: int,
    *,
    commit_threshold: float,
    shortage_cost: float,
    hedge_cost: float,
) -> tuple[float, float, bool]:
    """返回 (事前期望成本, 事后实际成本, 是否承诺)。

    事前期望成本是“承诺 vs 准备替代资源”两种行动里更便宜的那个，用来比较
    概率质量，而不是只比较点预测。事后成本是实际做出该行动后的损失。
    """

    commit_expected = (1.0 - probability) * shortage_cost * quantity
    hedge_expected = hedge_cost * quantity
    commit = commit_expected <= hedge_expected and probability >= 0.0
    # 同时尊重采购主管的显式承诺阈值：低于阈值不承诺。
    commit = commit and probability >= commit_threshold
    expected = commit_expected if commit else hedge_expected
    realized = (0.0 if label else shortage_cost * quantity) if commit else hedge_cost * quantity
    return expected, realized, commit


def _metrics(
    model: str,
    points: Sequence[ValidationPoint],
    quantity: int,
    decision_threshold: float,
    shortage_cost: float,
    hedge_cost: float,
) -> ModelMetrics:
    values = [
        (point.predictions[model], point.label)
        for point in points
        if point.predictions.get(model) is not None
    ]
    values = [(probability, label) for probability, label in values if probability is not None]
    if not values:
        return ModelMetrics(model, 0, 0.0, None, None, None, None, None, None, None, None)
    brier = sum((probability - label) ** 2 for probability, label in values) / len(values)
    log_loss = -sum(
        label * math.log(max(probability, 1e-9)) + (1 - label) * math.log(max(1 - probability, 1e-9))
        for probability, label in values
    ) / len(values)
    costs = [
        decision_costs(
            probability,
            label,
            quantity,
            commit_threshold=decision_threshold,
            shortage_cost=shortage_cost,
            hedge_cost=hedge_cost,
        )
        for probability, label in values
    ]
    return ModelMetrics(
        model=model,
        points=len(values),
        coverage=len(values) / len(points) if points else 0.0,
        brier=brier,
        log_loss=log_loss,
        calibration_error=_calibration_error(values),
        mean_prediction=sum(probability for probability, _ in values) / len(values),
        observed_rate=sum(label for _, label in values) / len(values),
        expected_cost_per_order=sum(cost[0] for cost in costs) / len(costs),
        realized_cost_per_decision=sum(cost[1] for cost in costs) / len(costs),
        commit_rate=sum(1 for cost in costs if cost[2]) / len(costs),
    )


def validation_origins(dataset: Dataset, horizon: int) -> tuple[date, ...]:
    configured = tuple(
        origin
        for origin in dataset.validation_origins
        if (dataset.observed_until - origin).days >= horizon
        and origin > min(order.placed_at for order in dataset.orders)
    )
    return configured


def _training_rows(
    dataset: Dataset,
    origin: date,
    supplier: str,
    category: str,
    target: int,
    horizon: int,
    min_rows: int,
) -> tuple[tuple[Observation, ...], bool]:
    observations = observations_at(dataset, origin, supplier, category, target, horizon)
    if len(observations) >= min_rows:
        return observations, False
    pooled = tuple(
        timeline.observe(min(target, timeline.quantity), horizon)
        for timeline in build_timelines(dataset, origin, supplier=supplier, placed_before=origin)
    )
    return pooled, True


def _run_points(
    dataset: Dataset,
    purchase: Purchase,
    origins: Sequence[date],
    min_training_rows: int,
) -> tuple[list[ValidationPoint], int]:
    horizon = purchase.horizon_days
    points: list[ValidationPoint] = []
    skipped_insufficient_observation = 0
    for origin in origins:
        for order in dataset.orders:
            if order.placed_at != origin:
                continue
            final_timelines = build_timelines(
                dataset,
                dataset.observed_until,
                supplier=order.supplier,
                category=order.category,
                placed_before=dataset.observed_until + timedelta(days=1),
            )
            timeline = next(item for item in final_timelines if item.order_id == order.order_id)
            label = final_outcome(timeline, order.quantity, horizon)
            if label is None:
                skipped_insufficient_observation += 1
                continue
            predictions: dict[str, float | None] = {}
            train_rows = 0
            for model in MODELS:
                observations, _ = _training_rows(
                    dataset, origin, order.supplier, order.category, order.quantity, horizon, min_training_rows
                )
                train_rows = len(observations)
                result = _ESTIMATORS[model](observations, horizon)
                predictions[model] = None if result is None else result.completion
            points.append(
                ValidationPoint(
                    origin=origin,
                    order_id=order.order_id,
                    supplier=order.supplier,
                    category=order.category,
                    target=order.quantity,
                    label=label,
                    predictions=predictions,
                    train_rows=train_rows,
                )
            )
    return points, skipped_insufficient_observation


def _report(
    dataset: Dataset,
    purchase: Purchase,
    origins: Sequence[date],
    points: Sequence[ValidationPoint],
    skipped: int,
    *,
    decision_threshold: float,
    shortage_cost: float,
    hedge_cost: float,
) -> dict[str, object]:
    metrics = [
        _metrics(model, points, purchase.quantity, decision_threshold, shortage_cost, hedge_cost)
        for model in MODELS
    ]
    candidates = [
        metric
        for metric in metrics
        if metric.model in (MODEL_HAZARD, MODEL_BASELINE_KM)
        and metric.log_loss is not None
    ]
    by_model = {metric.model: metric for metric in candidates}
    selected = MODEL_HAZARD
    margin_note = None
    if MODEL_HAZARD in by_model and MODEL_BASELINE_KM in by_model:
        hazard = by_model[MODEL_HAZARD]
        baseline = by_model[MODEL_BASELINE_KM]
        improvement = (hazard.log_loss or 0.0) - (baseline.log_loss or 0.0)
        # 只有基准显著更好（>0.05 对数损失）才替换主模型，避免追逐验证噪声。
        if improvement > 0.05:
            selected = MODEL_BASELINE_KM
        margin_note = (
            f"Kaplan-Meier 与主模型的对数损失差 {abs(improvement):.4f}；"
            "差异没有超过 0.05 的实用阈值，因此保留竞争风险模型。"
            if abs(improvement) <= 0.05
            else f"Kaplan-Meier 的对数损失比主模型低 {improvement:.4f}，超过实用阈值，改用基准。"
        )
    elif MODEL_BASELINE_KM in by_model:
        selected = MODEL_BASELINE_KM
    best_log_loss = min(
        (metric.log_loss for metric in candidates if metric.log_loss is not None),
        default=None,
    )
    if margin_note is None and best_log_loss is not None:
        margin_note = f"可用候选模型的最佳对数损失 {best_log_loss:.4f}。"
    always_commit = (
        sum((0.0 if point.label else shortage_cost * purchase.quantity) for point in points) / len(points)
        if points
        else None
    )
    always_hedge = hedge_cost * purchase.quantity if points else None
    return {
        "origins": [origin.isoformat() for origin in origins],
        "points": [point.as_dict() for point in points],
        "metrics": [metric.as_dict() for metric in metrics],
        "selected_model": selected,
        "selection_pool": [MODEL_HAZARD, MODEL_BASELINE_KM],
        "selection_criterion": (
            "向前验证点上与 Kaplan-Meier 基准比较：只有基准对数损失低出 0.05 以上才替换，"
            "否则保留竞争风险主模型（避免追逐验证噪声）"
        ),
        "selection_caveat": (
            "验证点很少，模型间差异可能落在噪声范围内；被选中模型的验证误差偏乐观。"
        ),
        "selection_margin_note": margin_note,
        "reference_costs": {
            "always_commit_per_order": None if always_commit is None else round(always_commit, 3),
            "always_hedge_per_order": None if always_hedge is None else round(always_hedge, 3),
        },
        "pending_without_full_horizon": skipped,
        "assumptions": {
            "training_uses_known_at_before_origin": True,
            "labels_require_full_horizon_observation": True,
            "termination_is_not_delivery": True,
            "open_after_deadline_counts_as_not_delivered": True,
            "decision_threshold": decision_threshold,
            "shortage_cost_per_unit": shortage_cost,
            "hedge_cost_per_unit": hedge_cost,
        },
    }


def rolling_origins(dataset: Dataset, horizon: int, step_days: int = 7) -> tuple[date, ...]:
    """补充验证起点：从首个官方起点开始，每 7 天一次，直到能看满截止日。"""

    start = min(dataset.validation_origins)
    last = dataset.observed_until - timedelta(days=horizon)
    origins: list[date] = []
    current = start
    while current <= last:
        origins.append(current)
        current += timedelta(days=step_days)
    return tuple(origins)


def forward_validation(
    dataset: Dataset,
    purchase: Purchase,
    *,
    decision_threshold: float = 0.8,
    shortage_cost: float = 20.0,
    hedge_cost: float = 6.0,
    min_training_rows: int = 10,
) -> dict[str, object]:
    horizon = purchase.horizon_days
    origins = validation_origins(dataset, horizon)
    points, skipped = _run_points(dataset, purchase, origins, min_training_rows)
    report = _report(
        dataset,
        purchase,
        origins,
        points,
        skipped,
        decision_threshold=decision_threshold,
        shortage_cost=shortage_cost,
        hedge_cost=hedge_cost,
    )
    rolling = tuple(
        origin
        for origin in rolling_origins(dataset, horizon)
        if origin not in set(origins)
    )
    if rolling:
        rolling_points, rolling_skipped = _run_points(dataset, purchase, rolling, min_training_rows)
        report["supplementary_rolling"] = _report(
            dataset,
            purchase,
            rolling,
            rolling_points,
            rolling_skipped,
            decision_threshold=decision_threshold,
            shortage_cost=shortage_cost,
            hedge_cost=hedge_cost,
        )
        report["supplementary_rolling"]["note"] = (
            "额外的每周滚动起点只用于检查结论是否稳定，不替代 fixture 给定的官方验证起点。"
        )
    return report


def baseline_comparison(validation: dict[str, object]) -> list[dict[str, object]]:
    """把候选模型与简单基准并排，便于看出校准和业务损失差异。"""

    rows = []
    selected = validation.get("selected_model")
    for metric in validation.get("metrics", []):
        rows.append(
            {
                **metric,
                "is_baseline": metric["model"] != selected,
                "selected": metric["model"] == selected,
            }
        )
    return rows
