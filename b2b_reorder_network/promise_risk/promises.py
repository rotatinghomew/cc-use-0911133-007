"""承诺、改约、终止与签收的时点语义。

这里刻意把三件事分开：

* 原合同承诺：``contract.promised_at``，永远保留。
* 双方接受的改约承诺：只有 ``accepted is True`` 且当时可见的 amendment 才能改变
  生效承诺；被拒绝的改约不改变承诺。
* 终止不是送达；未结案不是准时也不是迟到，而是尚未完成的观察。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from .data import Dataset, Event, Order


STATUS_ON_TIME = "on_time_complete"
STATUS_LATE = "late_complete"
STATUS_TERMINATED = "terminated"
STATUS_OVERDUE_UNRESOLVED = "overdue_unresolved"
STATUS_OPEN_NOT_DUE = "open_not_due"

PROMISE_BASIS_ORIGINAL = "original"
PROMISE_BASIS_ACCEPTED = "accepted"


def original_promise(order: Order) -> date:
    assert order.contract.promised_at is not None
    return order.contract.promised_at


def accepted_amendment(order: Order, as_of: date) -> Event | None:
    candidates = [
        event
        for event in order.visible_events(as_of)
        if event.kind == "amendment" and event.accepted is True and event.promised_at is not None
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda event: (event.at, event.known_at, event.index))[-1]


def effective_promise(order: Order, basis: str, as_of: date) -> date:
    if basis == PROMISE_BASIS_ORIGINAL:
        return original_promise(order)
    if basis != PROMISE_BASIS_ACCEPTED:
        raise ValueError(f"未知承诺口径: {basis!r}")
    amendment = accepted_amendment(order, as_of)
    if amendment is not None and amendment.promised_at is not None:
        return amendment.promised_at
    return original_promise(order)


@dataclass(frozen=True, slots=True)
class Progress:
    target: int
    received: int
    completion: Event | None
    termination: Event | None
    received_in_full: bool


def progress_at(order: Order, as_of: date, target: int) -> Progress:
    """由当时可见事件计算到 ``target`` 数量为止的进度。"""

    termination = order.visible_termination(as_of)
    termination_at = termination.at if termination is not None else None
    received = 0
    completion: Event | None = None
    for receipt in order.visible_receipts(as_of):
        if termination_at is not None and receipt.at > termination_at:
            continue
        received += receipt.quantity or 0
        if completion is None and received >= target:
            completion = receipt
            break
    return Progress(
        target=target,
        received=received,
        completion=completion,
        termination=termination,
        received_in_full=completion is not None,
    )


@dataclass(frozen=True, slots=True)
class PromiseLabel:
    order_id: str
    supplier: str
    route: str
    category: str
    basis: str
    promised_at: date
    status: str
    completed_at: date | None
    terminated_at: date | None
    resolution_lag_days: int | None
    received_quantity: int
    ordered_quantity: int
    as_of: date
    accepted_amendments: int
    rejected_amendments: int

    @property
    def resolved(self) -> bool:
        return self.status not in {STATUS_OVERDUE_UNRESOLVED, STATUS_OPEN_NOT_DUE}

    @property
    def delivered(self) -> bool:
        return self.status in {STATUS_ON_TIME, STATUS_LATE}

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "supplier": self.supplier,
            "route": self.route,
            "category": self.category,
            "promise_basis": self.basis,
            "promised_at": self.promised_at.isoformat(),
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "terminated_at": self.terminated_at.isoformat() if self.terminated_at else None,
            "resolution_lag_days": self.resolution_lag_days,
            "received_quantity": self.received_quantity,
            "ordered_quantity": self.ordered_quantity,
            "as_of": self.as_of.isoformat(),
            "accepted_amendments": self.accepted_amendments,
            "rejected_amendments": self.rejected_amendments,
        }


def label_order(order: Order, basis: str, as_of: date) -> PromiseLabel:
    """对整单做出 *当时可见* 的承诺标签。

    ``overdue_unresolved`` / ``open_not_due`` 都是未完成观察：前者是已过承诺日
    但仍没有送达或终止，后者是还没到承诺日。它们不能被当成准时或迟到。
    """

    promised_at = effective_promise(order, basis, as_of)
    progress = progress_at(order, as_of, order.quantity)
    terminated_at = progress.termination.at if progress.termination is not None else None
    completed_at = progress.completion.at if progress.completion is not None else None
    received = sum(
        receipt.quantity or 0
        for receipt in order.visible_receipts(as_of)
        if terminated_at is None or receipt.at <= terminated_at
    )
    if completed_at is not None:
        status = STATUS_ON_TIME if completed_at <= promised_at else STATUS_LATE
        lag: int | None = (completed_at - promised_at).days
    elif terminated_at is not None:
        status = STATUS_TERMINATED
        lag = None
    elif as_of > promised_at:
        status = STATUS_OVERDUE_UNRESOLVED
        lag = None
    else:
        status = STATUS_OPEN_NOT_DUE
        lag = None
    amendments = [event for event in order.visible_events(as_of) if event.kind == "amendment"]
    return PromiseLabel(
        order_id=order.order_id,
        supplier=order.supplier,
        route=order.route,
        category=order.category,
        basis=basis,
        promised_at=promised_at,
        status=status,
        completed_at=completed_at,
        terminated_at=terminated_at,
        resolution_lag_days=lag,
        received_quantity=received,
        ordered_quantity=order.quantity,
        as_of=as_of,
        accepted_amendments=sum(1 for event in amendments if event.accepted is True),
        rejected_amendments=sum(1 for event in amendments if event.accepted is False),
    )


def label_summary(labels: list[PromiseLabel]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.status] = counts.get(label.status, 0) + 1
    return {
        "orders": len(labels),
        "status_counts": dict(sorted(counts.items())),
        "unresolved": sum(1 for label in labels if not label.resolved),
        "terminated": counts.get(STATUS_TERMINATED, 0),
        "delivered": sum(1 for label in labels if label.delivered),
    }
