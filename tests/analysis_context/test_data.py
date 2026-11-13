from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DataContextTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "fixtures/promise_risk.json").read_text(encoding="utf-8"))

    def test_receipts_never_exceed_ordered_quantity(self):
        for order in self.data["orders"]:
            received = sum(row["quantity"] for row in order["events"] if row["kind"] == "received")
            self.assertLessEqual(received, order["events"][0]["quantity"])
        self.assertEqual(len(self.data["orders"]), len({row["order_id"] for row in self.data["orders"]}))

    def test_open_orders_shared_routes_and_observation_horizon(self):
        self.assertTrue(any(not any(row["kind"] in {"received", "terminated"} for row in order["events"]) for order in self.data["orders"]))
        suppliers = {row["supplier"] for row in self.data["orders"] if row["route"] == "shared-port"}
        self.assertGreater(len(suppliers), 1)
        self.assertTrue(all(row["known_at"] <= self.data["observed_until"] for order in self.data["orders"] for row in order["events"]))

