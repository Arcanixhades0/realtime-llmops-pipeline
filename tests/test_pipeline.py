import json
import os

os.environ["LLM_PROVIDER"] = "fake"
os.environ["FAKE_LLM_LATENCY"] = "0"

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app, state  # noqa: E402
from src.drift import DriftMonitor  # noqa: E402
from src.llm import ModelManager  # noqa: E402
from src.producer import make_event  # noqa: E402
from src.worker import parse, point_id  # noqa: E402


# ---------------- Drift ----------------

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_drift_low_for_similar_queries_high_for_different():
    monitor = DriftMonitor(reference=[unit([1, 0, 0])] * 5, min_samples=3)
    for _ in range(3):
        score = monitor.update(unit([1, 0.05, 0]))
    assert score < 0.01

    far = DriftMonitor(reference=[unit([1, 0, 0])] * 5, min_samples=3)
    for _ in range(3):
        score = far.update(unit([0, 1, 0]))
    assert score > 0.9


def test_drift_waits_for_enough_samples():
    assert DriftMonitor(reference=[unit([1, 0])], min_samples=5).update(unit([1, 0])) is None


# ---------------- Model manager ----------------

def test_rollback_and_restore():
    m = ModelManager()
    assert m.active == "primary"
    assert m.rollback("HighLLMLatency") is True
    assert m.active == "fallback"
    assert m.rollback("HighLLMLatency") is False   # already rolled back
    m.restore()
    assert m.active == "primary"


def test_chaos_errors_hit_primary_only():
    m = ModelManager()
    m.chaos = {"latency_s": 0, "error_rate": 1.0}
    with pytest.raises(RuntimeError):
        asyncio.run(m.generate("hi"))
    m.rollback("HighErrorRate")
    text, model = asyncio.run(m.generate("hi"))
    assert model == m.models["fallback"]


# ---------------- Kafka message handling ----------------

class Msg:
    def __init__(self, value):
        self._value = value

    def error(self):
        return None

    def value(self):
        return self._value


class FakeDLQ:
    def __init__(self):
        self.sent = []

    def produce(self, topic, value, headers=None):
        self.sent.append(value)


def test_bad_messages_go_to_dead_letter_queue():
    dlq = FakeDLQ()
    good = json.dumps({"id": "1", "text": "hello", "ts": 0}).encode()
    events = parse([Msg(good), Msg(b"not json"), Msg(json.dumps({"id": "2", "text": ""}).encode())], dlq)
    assert len(events) == 1
    assert len(dlq.sent) == 2


def test_event_ids_are_deterministic():
    item = {"text": "x" * 300, "source": "a.pdf", "page": 3}
    assert make_event(item)["id"] == make_event(item)["id"]
    assert point_id("abc") == point_id("abc")


# ---------------- Alertmanager webhook ----------------

def test_webhook_rolls_back_only_on_rollback_alerts():
    client = TestClient(app)          # no "with": skips startup, so no Qdrant needed
    state["models"] = ModelManager()

    warning = {"alerts": [{"status": "firing", "labels": {"alertname": "EmbeddingDrift", "severity": "warning"}}]}
    assert client.post("/admin/alerts", json=warning).json()["rolled_back"] == []

    critical = {"alerts": [{"status": "firing", "labels": {"alertname": "HighLLMLatency", "action": "rollback"}}]}
    assert client.post("/admin/alerts", json=critical).json()["rolled_back"] == ["HighLLMLatency"]
    assert state["models"].active == "fallback"
