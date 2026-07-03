from __future__ import annotations

from core.perf_monitor import PerfMonitor


def test_perf_monitor_records_aggregate_metrics():
    monitor = PerfMonitor()
    monitor.record("display.plot", 10.0)
    monitor.record("display.plot", 30.0)
    snap = monitor.snapshot()["display.plot"]
    assert snap["count"] == 2
    assert snap["avg_ms"] == 20.0
    assert snap["max_ms"] == 30.0
    assert snap["last_ms"] == 30.0


def test_perf_monitor_span_records_duration():
    monitor = PerfMonitor()
    with monitor.span("unit"):
        sum(range(10))
    snap = monitor.snapshot()["unit"]
    assert snap["count"] == 1
    assert snap["last_ms"] >= 0.0
