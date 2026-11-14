"""交付承诺风险分析。

这个子包只读取本地批量资料并生成分析结果，不修改现有返单流程，也不会下单。
入口：``python3 -m b2b_reorder_network.promise_risk <fixture> --output <dir>``。
"""

from __future__ import annotations

from .data import Dataset, FixtureError, load_dataset
from .analysis import analyze_dataset

__all__ = ["Dataset", "FixtureError", "load_dataset", "analyze_dataset"]
