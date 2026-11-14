"""批量履约资料的解析与信息时点检查。

所有分析都沿着 *当时可见* 的信息走：事件只有在 ``known_at`` 之后才能进入
样本。``at`` 是业务发生时间，``known_at`` 是入账/可见时间，两者不能混用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

EVENT_KINDS = {"contract", "amendment", "terminated", "received"}


class FixtureError(ValueError):
    """输入资料不满足分析约定。"""


def _require(value: Any, field_name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise FixtureError(f"{field_name} 不能为空")
    return value


def _as_date(value: Any, field_name: str) -> date:
    if not isinstance(value, str):
        raise FixtureError(f"{field_name} 必须是 ISO 日期字符串，实际为 {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise FixtureError(f"{field_name} 不是有效日期：{value!r}") from exc


def _as_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureError(f"{field_name} 必须是整数，实际为 {value!r}")
    if value < minimum:
        raise FixtureError(f"{field_name} 不得小于 {minimum}，实际为 {value}")
    return value


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"{field_name} 必须是非空字符串")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    at: date
    known_at: date
    promised_at: date | None = None
    quantity: int | None = None
    accepted: bool | None = None
    index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "at": self.at.isoformat(),
            "known_at": self.known_at.isoformat(),
            "promised_at": self.promised_at.isoformat() if self.promised_at else None,
            "quantity": self.quantity,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    supplier: str
    route: str
    category: str
    placed_at: date
    events: tuple[Event, ...]
    contract: Event

    @property
    def quantity(self) -> int:
        assert self.contract.quantity is not None
        return self.contract.quantity

    def visible_events(self, as_of: date) -> tuple[Event, ...]:
        return tuple(event for event in self.events if event.known_at <= as_of)

    def visible_receipts(self, as_of: date) -> tuple[Event, ...]:
        receipts = [event for event in self.visible_events(as_of) if event.kind == "received"]
        return tuple(sorted(receipts, key=lambda event: (event.at, event.known_at, event.index)))

    def visible_termination(self, as_of: date) -> Event | None:
        events = [event for event in self.visible_events(as_of) if event.kind == "terminated"]
        if not events:
            return None
        return sorted(events, key=lambda event: (event.at, event.known_at, event.index))[0]


@dataclass(frozen=True, slots=True)
class PurchaseSupplier:
    supplier_id: str
    minimum: int
    capacity: int
    route: str


@dataclass(frozen=True, slots=True)
class Purchase:
    placed_at: date
    deadline: date
    quantity: int
    category: str
    suppliers: tuple[PurchaseSupplier, ...]

    @property
    def horizon_days(self) -> int:
        return (self.deadline - self.placed_at).days


@dataclass(frozen=True, slots=True)
class Dataset:
    schema_version: int
    seed: int | None
    observed_until: date
    orders: tuple[Order, ...]
    validation_origins: tuple[date, ...]
    purchase: Purchase
    unsupported_purchase: Mapping[str, Any] | None = None
    source: str = ""
    quality_notes: tuple[str, ...] = ()

    @property
    def suppliers(self) -> tuple[str, ...]:
        return tuple(sorted({order.supplier for order in self.orders}))

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({order.category for order in self.orders}))

    @property
    def routes(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for order in self.orders:
            result.setdefault(order.supplier, order.route)
        return dict(sorted(result.items()))

    def orders_for(self, supplier: str | None = None, category: str | None = None) -> tuple[Order, ...]:
        return tuple(
            order
            for order in self.orders
            if (supplier is None or order.supplier == supplier)
            and (category is None or order.category == category)
        )

    def max_contract_quantity(self, supplier: str | None = None, category: str | None = None) -> int:
        quantities = [order.quantity for order in self.orders_for(supplier, category)]
        return max(quantities) if quantities else 0


def _parse_event(raw: Mapping[str, Any], order_id: str, index: int) -> Event:
    kind = _as_str(raw.get("kind"), f"{order_id}.events[{index}].kind")
    if kind not in EVENT_KINDS:
        raise FixtureError(f"{order_id}.events[{index}].kind 未知：{kind!r}")
    at = _as_date(raw.get("at"), f"{order_id}.events[{index}].at")
    known_at = _as_date(raw.get("known_at"), f"{order_id}.events[{index}].known_at")
    if known_at < at:
        raise FixtureError(
            f"{order_id}.events[{index}] 的 known_at 早于 at，信息时点不成立"
        )
    promised_at = None
    if raw.get("promised_at") is not None:
        promised_at = _as_date(raw["promised_at"], f"{order_id}.events[{index}].promised_at")
    quantity = None
    if raw.get("quantity") is not None:
        quantity = _as_int(raw["quantity"], f"{order_id}.events[{index}].quantity", minimum=1)
    accepted = None
    if "accepted" in raw:
        if not isinstance(raw["accepted"], bool):
            raise FixtureError(f"{order_id}.events[{index}].accepted 必须是布尔值")
        accepted = bool(raw["accepted"])
    if kind == "contract" and (promised_at is None or quantity is None):
        raise FixtureError(f"{order_id} 的 contract 必须包含 promised_at 与 quantity")
    if kind == "amendment" and (promised_at is None or accepted is None):
        raise FixtureError(f"{order_id} 的 amendment 必须包含 promised_at 与 accepted")
    if kind == "received" and quantity is None:
        raise FixtureError(f"{order_id} 的 received 必须包含 quantity")
    return Event(
        kind=kind,
        at=at,
        known_at=known_at,
        promised_at=promised_at,
        quantity=quantity,
        accepted=accepted,
        index=index,
    )


def _parse_order(raw: Mapping[str, Any], index: int) -> Order:
    order_id = _as_str(raw.get("order_id"), f"orders[{index}].order_id")
    supplier = _as_str(raw.get("supplier"), f"{order_id}.supplier")
    route = _as_str(raw.get("route"), f"{order_id}.route")
    category = _as_str(raw.get("category"), f"{order_id}.category")
    placed_at = _as_date(raw.get("placed_at"), f"{order_id}.placed_at")
    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise FixtureError(f"{order_id}.events 必须是非空数组")
    events = tuple(
        _parse_event(event, order_id, event_index)
        for event_index, event in enumerate(raw_events)
        if isinstance(event, Mapping)
    )
    if len(events) != len(raw_events):
        raise FixtureError(f"{order_id}.events 含有非对象元素")
    contracts = [event for event in events if event.kind == "contract"]
    if len(contracts) != 1:
        raise FixtureError(f"{order_id} 必须且只能有一个 contract")
    contract = contracts[0]
    if contract.at != placed_at:
        raise FixtureError(f"{order_id} 的 placed_at 与 contract.at 不一致")
    received = sum(event.quantity or 0 for event in events if event.kind == "received")
    if received > (contract.quantity or 0):
        raise FixtureError(f"{order_id} 的签收总量 {received} 超过合同数量 {contract.quantity}")
    return Order(
        order_id=order_id,
        supplier=supplier,
        route=route,
        category=category,
        placed_at=placed_at,
        events=events,
        contract=contract,
    )


def _parse_purchase(raw: Mapping[str, Any], categories: set[str], routes: Mapping[str, str]) -> Purchase:
    if not isinstance(raw, Mapping):
        raise FixtureError("purchase 必须是对象")
    placed_at = _as_date(raw.get("placed_at"), "purchase.placed_at")
    deadline = _as_date(raw.get("deadline"), "purchase.deadline")
    if deadline <= placed_at:
        raise FixtureError("purchase.deadline 必须晚于 placed_at")
    quantity = _as_int(raw.get("quantity"), "purchase.quantity", minimum=1)
    category = _as_str(raw.get("category"), "purchase.category")
    if category not in categories:
        # 允许出现未覆盖品类；分析层会明确保留意见，而不是在这里伪造支持。
        pass
    raw_suppliers = raw.get("suppliers")
    if not isinstance(raw_suppliers, list) or not raw_suppliers:
        raise FixtureError("purchase.suppliers 必须是非空数组")
    suppliers: list[PurchaseSupplier] = []
    for index, item in enumerate(raw_suppliers):
        if not isinstance(item, Mapping):
            raise FixtureError(f"purchase.suppliers[{index}] 必须是对象")
        supplier_id = _as_str(item.get("id"), f"purchase.suppliers[{index}].id")
        minimum = _as_int(item.get("minimum"), f"purchase.suppliers[{index}].minimum", minimum=1)
        capacity = _as_int(item.get("capacity"), f"purchase.suppliers[{index}].capacity", minimum=1)
        route = _as_str(item.get("route"), f"purchase.suppliers[{index}].route")
        if minimum > capacity:
            raise FixtureError(f"{supplier_id} 的最小起订量大于产能")
        if supplier_id in routes and routes[supplier_id] != route:
            raise FixtureError(f"{supplier_id} 的采购线路 {route!r} 与历史线路 {routes[supplier_id]!r} 不一致")
        suppliers.append(PurchaseSupplier(supplier_id=supplier_id, minimum=minimum, capacity=capacity, route=route))
    return Purchase(
        placed_at=placed_at,
        deadline=deadline,
        quantity=quantity,
        category=category,
        suppliers=tuple(suppliers),
    )


def load_dataset(path: str | Path) -> Dataset:
    source = str(path)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise FixtureError(f"无法读取 {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{source} 不是有效 JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise FixtureError("资料顶层必须是对象")
    schema_version = _as_int(raw.get("schema_version", 1), "schema_version", minimum=1)
    observed_until = _as_date(raw.get("observed_until"), "observed_until")
    raw_orders = raw.get("orders")
    if not isinstance(raw_orders, list) or not raw_orders:
        raise FixtureError("orders 必须是非空数组")
    orders = tuple(_parse_order(item, index) for index, item in enumerate(raw_orders))
    if len({order.order_id for order in orders}) != len(orders):
        raise FixtureError("order_id 存在重复")
    categories = {order.category for order in orders}
    routes: dict[str, str] = {}
    for order in orders:
        routes.setdefault(order.supplier, order.route)
    purchase = _parse_purchase(raw.get("purchase"), categories, routes)
    origins_raw = raw.get("validation_origins")
    if not isinstance(origins_raw, list) or not origins_raw:
        raise FixtureError("validation_origins 必须是非空数组")
    origins = tuple(sorted({_as_date(item, "validation_origins[]") for item in origins_raw}))
    unsupported = raw.get("unsupported_purchase")
    if unsupported is not None and not isinstance(unsupported, Mapping):
        raise FixtureError("unsupported_purchase 必须是对象或 null")
    quality_notes: list[str] = []
    future_events = [
        (order.order_id, event.kind, event.known_at.isoformat())
        for order in orders
        for event in order.events
        if event.known_at > observed_until
    ]
    if future_events:
        quality_notes.append(
            f"有 {len(future_events)} 个事件的 known_at 晚于 observed_until，分析中会被截断"
        )
    return Dataset(
        schema_version=schema_version,
        seed=raw.get("seed") if isinstance(raw.get("seed"), int) else None,
        observed_until=observed_until,
        orders=orders,
        validation_origins=origins,
        purchase=purchase,
        unsupported_purchase=unsupported,
        source=source,
        quality_notes=tuple(quality_notes),
    )
