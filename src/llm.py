"""
Serves LLM calls through a primary model with a fallback ready.

`chaos` lets us inject failures (extra latency, random errors) into the primary
model on purpose, to prove the monitoring + automatic rollback actually work.
"""
import asyncio
import random
import time

from src import metrics
from src.config import FAKE_LLM_LATENCY, FALLBACK_MODEL, LLM_PROVIDER, PRIMARY_MODEL


class ModelManager:

    def __init__(self):
        self.models = {"primary": PRIMARY_MODEL, "fallback": FALLBACK_MODEL}
        self.active = "primary"
        self.chaos = {"latency_s": 0.0, "error_rate": 0.0}
        self.history = []
        self._clients = {}
        self._publish_active()

    @property
    def active_model(self):
        return self.models[self.active]

    def _publish_active(self):
        for role, name in self.models.items():
            metrics.ACTIVE_MODEL.labels(name).set(1 if role == self.active else 0)

    def rollback(self, reason):
        if self.active == "fallback":
            return False
        self.active = "fallback"
        self.history.append({"event": "rollback", "reason": reason, "at": time.time()})
        metrics.ROLLBACKS.labels(reason).inc()
        self._publish_active()
        return True

    def restore(self):
        self.active = "primary"
        self.history.append({"event": "restore", "at": time.time()})
        self._publish_active()

    def _client(self, model):
        if model not in self._clients:
            from langchain_groq import ChatGroq
            extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            self._clients[model] = ChatGroq(model=model, temperature=0, max_retries=3, **extra)
        return self._clients[model]

    async def generate(self, prompt):
        model = self.active_model

        # Injected failures only hit the primary, like a real provider incident would
        if self.active == "primary":
            if self.chaos["latency_s"]:
                await asyncio.sleep(self.chaos["latency_s"])
            if random.random() < self.chaos["error_rate"]:
                raise RuntimeError("injected failure")

        if LLM_PROVIDER == "fake":
            await asyncio.sleep(FAKE_LLM_LATENCY * random.uniform(0.7, 1.3))
            text = "Based on the retrieved documents: " + prompt[-120:]
            usage = {"input_tokens": len(prompt) // 4, "output_tokens": 40}
        else:
            reply = await self._client(model).ainvoke(prompt)
            text = reply.content
            usage = reply.usage_metadata or {}

        metrics.TOKENS.labels(model, "prompt").inc(usage.get("input_tokens", 0))
        metrics.TOKENS.labels(model, "completion").inc(usage.get("output_tokens", 0))

        return text, model
