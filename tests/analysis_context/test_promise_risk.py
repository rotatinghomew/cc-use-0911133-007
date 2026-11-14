from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from b2b_reorder_network.promise_risk.analysis import analyze_dataset
from b2b_reorder_network.promise_risk.data import Dataset, Event, Order, Purchase, PurchaseSupplier, load_dataset
from b2b_reorder_network.promise_risk.dependence import (
    METHOD_COMMON_ROUTE,
    METHOD_INDEPENDENCE,
    bivariate_normal_cdf,
    inv_norm,
    joint_all,
    joint_both_late,
)
from b2b_reorder_network.promise_risk.plans import (
    empirical_plan_metrics,
    evaluate_plans,
    feasible_allocations,
)
from b2b_reorder_network.promise_risk.promises import (
    PROMISE_BASIS_ACCEPTED,
    PROMISE_BASIS_ORIGINAL,
    STATUS_LATE,
    STATUS_ON_TIME,
    STATUS_OPEN_NOT_DUE,
    STATUS_OVERDUE_UNRESOLVED,
    STATUS_TERMINATED,
    label_order,
)
from b2b_reorder_network.promise_risk.survival import (
    STATUS_CENSORED,
    STATUS_DELIVERED,
    STATUS_FAILED,
    aalen_johansen,
    all_orders_rate,
    build_timeline,
    build_timelines,
    closed_case_rate,
    kaplan_meier,
)
from b2b_reorder_network.promise_risk.validation import forward_validation


ROOT = Path(__file__).resolve().parents[2]


def make_order(
    order_id: str,
    supplier: str,
    placed_at: str,
    promised_at: str,
    events: list[dict],
    *,
    route: str = "shared-port",
    category: str = "textile",
    quantity: int = 40,
) -> Order:
    raw_events = [
        {"kind": "contract", "at": placed_at, "known_at": placed_at, "promised_at": promised_at, "quantity": quantity}
    ] + events
    parsed = tuple(
        Event(
            kind=raw["kind"],
            at=date.fromisoformat(raw["at"]),
            known_at=date.fromisoformat(raw["known_at"]),
            promised_at=date.fromisoformat(raw["promised_at"]) if raw.get("promised_at") else None,
            quantity=raw.get("quantity"),
            accepted=raw.get("accepted"),
            index=index + 1,
        )
        for index, raw in enumerate(raw_events[1:])
    )
    contract = Event(
        kind="contract",
        at=date.fromisoformat(placed_at),
        known_at=date.fromisoformat(placed_at),
        promised_at=date.fromisoformat(promised_at),
        quantity=quantity,
        index=0,
    )
    return Order(
        order_id=order_id,
        supplier=supplier,
        route=route,
        category=category,
        placed_at=date.fromisoformat(placed_at),
        events=(contract,) + parsed,
        contract=contract,
    )


def make_dataset(orders: list[Order], purchase: Purchase, observed_until: str = "2026-03-01") -> Dataset:
    return Dataset(
        schema_version=1,
        seed=None,
        observed_until=date.fromisoformat(observed_until),
        orders=tuple(orders),
        validation_origins=(date(2026, 2, 1),),
        purchase=purchase,
        unsupported_purchase={"category": "new-category", "quantity": 1000},
        source="synthetic",
    )


class PromiseSemanticsTests(unittest.TestCase):
    def test_last_extension_does_not_erase_original_promise(self):
        order = make_order(
            "o1",
            "s1",
            "2026-01-01",
            "2026-01-08",
            [
                {"kind": "amendment", "accepted": True, "promised_at": "2026-01-15", "at": "2026-01-07", "known_at": "2026-01-07"},
                {"kind": "received", "quantity": 40, "at": "2026-01-12", "known_at": "2026-01-13"},
            ],
        )
        original = label_order(order, PROMISE_BASIS_ORIGINAL, date(2026, 1, 20))
        accepted = label_order(order, PROMISE_BASIS_ACCEPTED, date(2026, 1, 20))
        self.assertEqual(original.promised_at, date(2026, 1, 8))
        self.assertEqual(original.status, STATUS_LATE)
        self.assertEqual(accepted.promised_at, date(2026, 1, 15))
        self.assertEqual(accepted.status, STATUS_ON_TIME)

    def test_rejected_amendment_does_not_change_promise(self):
        order = make_order(
            "o2",
            "s1",
            "2026-01-01",
            "2026-01-08",
            [
                {"kind": "amendment", "accepted": False, "promised_at": "2026-01-20", "at": "2026-01-06", "known_at": "2026-01-06"},
                {"kind": "received", "quantity": 40, "at": "2026-01-12", "known_at": "2026-01-13"},
            ],
        )
        accepted = label_order(order, PROMISE_BASIS_ACCEPTED, date(2026, 1, 20))
        self.assertEqual(accepted.promised_at, date(2026, 1, 8))
        self.assertEqual(accepted.status, STATUS_LATE)

    def test_termination_is_not_delivery(self):
        order = make_order(
            "o3",
            "s1",
            "2026-01-01",
            "2026-01-08",
            [{"kind": "terminated", "at": "2026-01-05", "known_at": "2026-01-05"}],
        )
        label = label_order(order, PROMISE_BASIS_ORIGINAL, date(2026, 1, 20))
        self.assertEqual(label.status, STATUS_TERMINATED)
        self.assertFalse(label.delivered)
        timeline = build_timeline(order, date(2026, 1, 20))
        observation = timeline.observe(40, 10)
        self.assertEqual(observation.status, STATUS_TERMINATED)
        self.assertFalse(observation.delivered)

    def test_unclosed_order_is_incomplete_observation(self):
        order = make_order("o4", "s1", "2026-01-01", "2026-01-08", [])
        before_due = label_order(order, PROMISE_BASIS_ORIGINAL, date(2026, 1, 5))
        after_due = label_order(order, PROMISE_BASIS_ORIGINAL, date(2026, 1, 20))
        self.assertEqual(before_due.status, STATUS_OPEN_NOT_DUE)
        self.assertEqual(after_due.status, STATUS_OVERDUE_UNRESOLVED)
        self.assertFalse(before_due.resolved)
        self.assertFalse(after_due.resolved)
        self.assertNotIn(before_due.status, {STATUS_ON_TIME, STATUS_LATE})
        self.assertNotIn(after_due.status, {STATUS_ON_TIME, STATUS_LATE})

    def test_events_are_invisible_before_known_at(self):
        order = make_order(
            "o5",
            "s1",
            "2026-01-01",
            "2026-01-08",
            [{"kind": "received", "quantity": 40, "at": "2026-01-07", "known_at": "2026-01-09"}],
        )
        early = label_order(order, PROMISE_BASIS_ORIGINAL, date(2026, 1, 8))
        late = label_order(order, PROMISE_BASIS_ORIGINAL, date(2026, 1, 10))
        self.assertNotIn(early.status, {STATUS_ON_TIME, STATUS_LATE})
        self.assertEqual(late.status, STATUS_ON_TIME)


class SurvivalTests(unittest.TestCase):
    def test_open_order_before_horizon_is_censored_not_failure(self):
        order = make_order("s1", "s1", "2026-01-01", "2026-01-08", [])
        timeline = build_timeline(order, date(2026, 1, 5))
        self.assertEqual(timeline.observe(40, 10).status, STATUS_CENSORED)
        timeline = build_timeline(order, date(2026, 1, 11))
        self.assertEqual(timeline.observe(40, 10).status, STATUS_FAILED)

    def test_partial_receipt_before_deadline_is_still_short(self):
        order = make_order(
            "s2",
            "s1",
            "2026-01-01",
            "2026-01-08",
            [
                {"kind": "received", "quantity": 15, "at": "2026-01-06", "known_at": "2026-01-07"},
                {"kind": "received", "quantity": 10, "at": "2026-01-09", "known_at": "2026-01-10"},
                {"kind": "received", "quantity": 15, "at": "2026-01-14", "known_at": "2026-01-15"},
            ],
        )
        timeline = build_timeline(order, date(2026, 1, 20))
        full = timeline.observe(40, 10)
        self.assertEqual(full.status, STATUS_FAILED)
        self.assertEqual(timeline.delivered_quantity(10), 25)
        by_20 = timeline.observe(20, 10)
        self.assertEqual(by_20.status, STATUS_DELIVERED)
        self.assertEqual(by_20.time, 8)

    def test_aalen_johansen_is_bounded_and_matches_hand_calculation(self):
        # 4 笔：2 笔第 3 天交齐，1 笔第 5 天终止，1 笔观察 8 天未满截止日（删失）。
        orders = [
            make_order("a", "s1", "2026-01-01", "2026-01-08", [{"kind": "received", "quantity": 40, "at": "2026-01-04", "known_at": "2026-01-04"}]),
            make_order("b", "s1", "2026-01-01", "2026-01-08", [{"kind": "received", "quantity": 40, "at": "2026-01-04", "known_at": "2026-01-04"}]),
            make_order("c", "s1", "2026-01-01", "2026-01-08", [{"kind": "terminated", "at": "2026-01-06", "known_at": "2026-01-06"}]),
            make_order("d", "s1", "2026-01-01", "2026-01-08", []),
        ]
        observations = [
            build_timeline(order, date(2026, 1, 9)).observe(40, 10) for order in orders
        ]
        estimate = aalen_johansen(observations, 10)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.completion, 0.5)
        self.assertAlmostEqual(estimate.failure, 0.25)
        self.assertAlmostEqual(estimate.open_at_horizon, 0.25)
        self.assertEqual(estimate.censored, 1)
        self.assertLessEqual(estimate.completion + estimate.failure + estimate.open_at_horizon, 1.0 + 1e-9)

    def test_kaplan_meier_overstates_when_termination_precedes_delivery(self):
        # 终止发生在其余交付之前；把它当删失的 KM 会更高估交齐概率。
        orders = [
            make_order("a", "s1", "2026-01-01", "2026-01-08", [{"kind": "terminated", "at": "2026-01-02", "known_at": "2026-01-02"}]),
            make_order("b", "s1", "2026-01-01", "2026-01-08", [{"kind": "received", "quantity": 40, "at": "2026-01-06", "known_at": "2026-01-06"}]),
            make_order("c", "s1", "2026-01-01", "2026-01-08", [{"kind": "received", "quantity": 40, "at": "2026-01-09", "known_at": "2026-01-09"}]),
        ]
        observations = [
            build_timeline(order, date(2026, 1, 20)).observe(40, 10) for order in orders
        ]
        hazard = aalen_johansen(observations, 10)
        km = kaplan_meier(observations, 10)
        self.assertAlmostEqual(hazard.completion, 2 / 3)
        self.assertAlmostEqual(hazard.failure, 1 / 3)
        self.assertAlmostEqual(km.completion, 1.0)
        self.assertGreater(km.completion, hazard.completion)
        self.assertAlmostEqual(closed_case_rate(observations, 10), 2 / 3)
        self.assertAlmostEqual(all_orders_rate(observations, 10), 2 / 3)

    def test_quantity_curve_is_monotone_and_marks_extrapolation(self):
        dataset = load_dataset(ROOT / "fixtures/promise_risk.json")
        from b2b_reorder_network.promise_risk.survival import quantity_curve

        timelines = build_timelines(dataset, dataset.observed_until, supplier="s1", category="textile")
        curve = quantity_curve(timelines, dataset.purchase.horizon_days, 60, supplier="s1", category="textile", pooled=False)
        values = [curve.at_least(quantity) for quantity in range(1, 61)]
        for left, right in zip(values, values[1:]):
            self.assertLessEqual(right, left + 1e-12)
        self.assertEqual(curve.support(40), "observed")
        self.assertIn("extrapolated", curve.support(41))


class DependenceTests(unittest.TestCase):
    def test_positive_correlation_increases_both_joint_success_and_both_failure(self):
        probabilities = {"a": 0.8, "b": 0.75}
        routes = {"a": "shared", "b": "shared"}
        rho = {"shared": 0.8}
        independent_success = joint_all(probabilities, routes, rho, METHOD_INDEPENDENCE)
        correlated_success = joint_all(probabilities, routes, rho, METHOD_COMMON_ROUTE)
        independent_both_short = joint_both_late(probabilities, routes, rho, METHOD_INDEPENDENCE)
        correlated_both_short = joint_both_late(probabilities, routes, rho, METHOD_COMMON_ROUTE)
        self.assertGreater(correlated_success, independent_success)
        self.assertGreater(correlated_both_short, independent_both_short)

    def test_joint_is_not_average_or_product(self):
        probabilities = {"a": 0.9, "b": 0.6}
        routes = {"a": "shared", "b": "shared"}
        independent = joint_all(probabilities, routes, {"shared": 0.0}, METHOD_INDEPENDENCE)
        average = sum(probabilities.values()) / len(probabilities)
        product = math.prod(probabilities.values())
        self.assertAlmostEqual(independent, product)
        self.assertNotAlmostEqual(independent, average)

    def test_copula_cdf_matches_known_values(self):
        self.assertAlmostEqual(bivariate_normal_cdf(0, 0, 0.0), 0.25, places=6)
        self.assertAlmostEqual(bivariate_normal_cdf(0, 0, 0.5), 1 / 3, places=3)
        self.assertGreater(bivariate_normal_cdf(0, 0, 0.9), bivariate_normal_cdf(0, 0, 0.5))
        self.assertAlmostEqual(inv_norm(0.5), 0.0, places=9)

    def test_shared_route_example_in_report(self):
        dataset = load_dataset(ROOT / "fixtures/promise_risk.json")
        report = analyze_dataset(dataset)
        dependence = report["dependence"]
        self.assertEqual(dependence["rho_source"]["shared-port"], "estimated_from_matched_cohorts")
        pair = dependence["shared_route_example"]["pair"]
        self.assertTrue(pair["same_route"])
        self.assertGreater(pair["both_short_rate_empirical"], pair["both_short_rate_independence"])
        self.assertIsNotNone(dependence["shared_route_example"]["late_cohort"])


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset(ROOT / "fixtures/promise_risk.json")

    def test_feasible_allocations_respect_minimum_and_capacity(self):
        purchase = self.dataset.purchase
        allocations = feasible_allocations(purchase)
        self.assertTrue(allocations)
        for allocation in allocations:
            self.assertEqual(sum(allocation.values()), purchase.quantity)
            for supplier in purchase.suppliers:
                quantity = allocation.get(supplier.supplier_id, 0)
                if quantity:
                    self.assertGreaterEqual(quantity, supplier.minimum)
                self.assertLessEqual(quantity, supplier.capacity)

    def test_whole_order_is_infeasible_under_capacity(self):
        result = evaluate_plans(
            self.dataset,
            self.dataset.purchase,
            rho_by_route={"shared-port": 0.8, "land-route": 0.0},
        )
        for evaluation in result["evaluations"]:
            self.assertLessEqual(len(evaluation.plan.allocations), 3)

    def test_joint_probability_is_not_naive_average(self):
        result = evaluate_plans(
            self.dataset,
            self.dataset.purchase,
            rho_by_route={"shared-port": 0.8, "land-route": 0.0},
        )
        for evaluation in result["evaluations"]:
            if len(evaluation.plan.allocations) < 2:
                continue
            joint = evaluation.plan.joint_probability
            average = evaluation.plan.naive_average_probability
            self.assertIsNotNone(joint)
            self.assertNotAlmostEqual(joint, average)
            self.assertGreaterEqual(joint, 0.0)
            self.assertLessEqual(joint, 1.0)

    def test_shared_route_plan_has_higher_simultaneous_shortage(self):
        result = evaluate_plans(
            self.dataset,
            self.dataset.purchase,
            rho_by_route={"shared-port": 0.8, "land-route": 0.0},
        )
        shared = next(
            evaluation
            for evaluation in result["evaluations"]
            if set(evaluation.plan.allocations) == {"s1", "s2"}
        )
        cross = next(
            evaluation
            for evaluation in result["evaluations"]
            if set(evaluation.plan.allocations) == {"s1", "s3"}
            and evaluation.plan.allocations.get("s1") == shared.plan.allocations.get("s1")
        )
        self.assertGreater(
            shared.plan.both_short_probability,
            cross.plan.both_short_probability * 3,
        )
        self.assertGreater(
            shared.plan.both_short_probability,
            shared.both_short_independence * 2,
        )

    def test_empirical_cohorts_refuse_unobserved_scale(self):
        result = evaluate_plans(
            self.dataset,
            self.dataset.purchase,
            rho_by_route={"shared-port": 0.8, "land-route": 0.0},
        )
        oversized = next(
            evaluation
            for evaluation in result["evaluations"]
            if evaluation.plan.allocations.get("s3", 0) >= 60
        )
        self.assertFalse(oversized.empirical_supported)
        self.assertIsNone(oversized.empirical_full)

    def test_supplier_without_history_is_not_given_a_probability(self):
        purchase = replace(
            self.dataset.purchase,
            suppliers=self.dataset.purchase.suppliers
            + (PurchaseSupplier("s9", 20, 60, "new-route"),),
        )
        result = evaluate_plans(
            self.dataset,
            purchase,
            rho_by_route={"shared-port": 0.8, "land-route": 0.0, "new-route": 0.0},
        )
        involving_new = [
            evaluation
            for evaluation in result["evaluations"]
            if "s9" in evaluation.plan.allocations
        ]
        self.assertTrue(involving_new)
        for evaluation in involving_new:
            self.assertIsNone(evaluation.plan.joint_probability)
            self.assertEqual(evaluation.plan.supplier_support["s9"], "no_history")
        untouched = [
            evaluation
            for evaluation in result["evaluations"]
            if "s9" not in evaluation.plan.allocations
        ]
        self.assertTrue(any(evaluation.plan.joint_probability is not None for evaluation in untouched))

    def test_recommendation_changes_are_explained(self):
        report = analyze_dataset(self.dataset)
        recommendation = report["plans"]["primary_recommendation"]
        changes = recommendation["changes"]
        self.assertTrue(any(change["recommendation_changes"] for change in changes))
        changed = next(change for change in changes if change["recommendation_changes"])
        self.assertIn("explanation", changed)
        self.assertIn("comparison", changed)

    def test_recommendation_does_not_place_an_order(self):
        report = analyze_dataset(self.dataset)
        self.assertTrue(report["assumptions"])
        text = " ".join(report["assumptions"])
        self.assertIn("不自动下单", text)
        self.assertTrue(report["plans"]["primary_recommendation"]["status"], "supported")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset(ROOT / "fixtures/promise_risk.json")

    def test_forward_validation_uses_only_visible_information(self):
        result = forward_validation(self.dataset, self.dataset.purchase)
        assumptions = result["assumptions"]
        self.assertTrue(assumptions["training_uses_known_at_before_origin"])
        self.assertTrue(assumptions["labels_require_full_horizon_observation"])
        self.assertTrue(assumptions["termination_is_not_delivery"])
        self.assertTrue(assumptions["open_after_deadline_counts_as_not_delivered"])
        self.assertEqual(result["origins"], ["2026-08-05", "2026-08-19", "2026-09-02"])

    def test_validation_reports_baselines_and_selected_model(self):
        result = forward_validation(self.dataset, self.dataset.purchase)
        models = {metric["model"] for metric in result["metrics"]}
        self.assertIn("aalen_johansen_competing_risks", models)
        self.assertIn("kaplan_meier_termination_censored", models)
        self.assertIn("closed_case_rate", models)
        self.assertIn("all_orders_assumed_short", models)
        self.assertIn(result["selected_model"], result["selection_pool"])
        self.assertIsNotNone(result["reference_costs"]["always_hedge_per_order"])

    def test_training_at_origin_ignores_future_known_events(self):
        from b2b_reorder_network.promise_risk.validation import observations_at

        origin = date(2026, 8, 5)
        observations = observations_at(self.dataset, origin, "s1", "textile", 40, 10)
        self.assertTrue(observations)
        for observation in observations:
            self.assertLessEqual(observation.time, 10)
        timelines = build_timelines(self.dataset, origin, supplier="s1", category="textile")
        for timeline in timelines:
            self.assertTrue(all(day >= 0 for day, _ in timeline.receipt_cum))


class UnsupportedAndCliTests(unittest.TestCase):
    def test_uncovered_category_returns_reservation_not_probability(self):
        dataset = load_dataset(ROOT / "fixtures/promise_risk.json")
        purchase = replace(dataset.purchase, category="new-category", quantity=1000)
        report = analyze_dataset(replace(dataset, purchase=purchase))
        self.assertFalse(report["supported"])
        self.assertEqual(report["reservation"]["status"], "insufficient_support")
        self.assertNotIn("plans", report)
        self.assertNotIn("quantity_curves", report)
        text = report["reservation"]["statement"]
        self.assertIn("保留意见", text)

    def test_cli_writes_report_summary_and_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "b2b_reorder_network.promise_risk",
                    str(ROOT / "fixtures/promise_risk.json"),
                    "--output",
                    directory,
                    "--quiet",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["supported"])
            self.assertEqual(payload["recommendation"], {"s1": 40, "s3": 40})
            for name in ("report.json", "summary.md", "supplier_commitments.csv", "plan_comparison.csv"):
                self.assertTrue((Path(directory) / name).exists(), name)
            summary = (Path(directory) / "summary.md").read_text(encoding="utf-8")
            self.assertIn("整批全部交齐", summary)
            self.assertIn("推荐为何变化", summary)
            self.assertIn("保留意见", summary)
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertIn("promise_basis", report)
            self.assertIn("unclosed_examples", report)


if __name__ == "__main__":
    unittest.main()
