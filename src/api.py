"""
Query API: question -> embed -> Qdrant -> LLM, fully instrumented.

  POST /ask            {"question": "...", "k": 5}
  GET  /metrics        Prometheus scrapes this
  POST /admin/alerts   Alertmanager calls this when an alert fires -> automatic rollback
  POST /admin/chaos    inject latency/errors into the primary model (failure drills)
  POST /admin/restore  switch back to the primary model
  GET  /admin/state

Run:  uvicorn src.api:app --port 8000
"""
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from src import metrics
from src.config import COLLECTION, QDRANT_URL
from src.drift import DriftMonitor
from src.embedder import EmbedBatcher, get_model
from src.llm import ModelManager

PROMPT = """Answer using only the context. If the answer is not there, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

state = {}


async def load_reference(qdrant, drift):
    """Sample stored document vectors to build the drift reference centroid."""
    if not await qdrant.collection_exists(COLLECTION):
        return
    points, _ = await qdrant.scroll(COLLECTION, limit=1000, with_vectors=True, with_payload=False)
    if points:
        drift.set_reference([p.vector for p in points])


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(get_model)          # load the embedding model before taking traffic
    # pool_size: reuse up to 100 connections (the default opened a new one per request);
    # timeout: the 5s default timed out requests that were only waiting in a queue
    state["qdrant"] = AsyncQdrantClient(url=QDRANT_URL, timeout=30, pool_size=100)
    state["batcher"] = EmbedBatcher()
    state["batcher"].start()
    state["models"] = ModelManager()
    state["drift"] = DriftMonitor()
    await load_reference(state["qdrant"], state["drift"])
    yield
    await state["batcher"].stop()
    await state["qdrant"].close()


app = FastAPI(title="Real-Time RAG", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class Chaos(BaseModel):
    latency_s: float = Field(default=0, ge=0, le=30)
    error_rate: float = Field(default=0, ge=0, le=1)


@app.post("/ask")
async def ask(body: Ask):
    models = state["models"]
    model = models.active_model
    metrics.REQUESTS.labels(model).inc()
    start = time.perf_counter()

    try:
        with metrics.RETRIEVAL_SECONDS.time():
            with metrics.EMBED_SECONDS.time():
                # Batched with other concurrent requests, computed in a worker thread
                vector = await state["batcher"].embed(body.question)
            with metrics.SEARCH_SECONDS.time():
                hits = (await state["qdrant"].query_points(COLLECTION, query=vector.tolist(), limit=body.k)).points
    except Exception as e:
        metrics.REQUEST_ERRORS.labels(model, "retrieval").inc()
        raise HTTPException(status_code=503, detail=f"retrieval failed: {type(e).__name__}: {e}")

    state["drift"].update(vector)
    context = "\n\n".join(h.payload["text"] for h in hits)

    try:
        llm_start = time.perf_counter()
        answer, model = await models.generate(PROMPT.format(context=context, question=body.question))
        metrics.LLM_SECONDS.labels(model).observe(time.perf_counter() - llm_start)
    except Exception as e:
        metrics.REQUEST_ERRORS.labels(model, "llm").inc()
        raise HTTPException(status_code=502, detail=f"llm failed: {e}")

    seconds = time.perf_counter() - start
    metrics.REQUEST_SECONDS.labels(model).observe(seconds)

    return {
        "answer": answer,
        "model": model,
        "seconds": round(seconds, 3),
        "sources": [{"source": h.payload.get("source"), "page": h.payload.get("page"), "score": h.score} for h in hits],
    }


@app.get("/health")
async def health():
    count = (await state["qdrant"].count(COLLECTION)).count if await state["qdrant"].collection_exists(COLLECTION) else 0
    return {"status": "ok", "documents": count, "active_model": state["models"].active_model}


@app.post("/admin/alerts")
async def alertmanager_webhook(payload: dict):
    """
    Alertmanager posts here. Alerts labelled action="rollback" switch traffic
    to the fallback model immediately, without waiting for a human.
    """
    rolled_back = []
    for alert in payload.get("alerts", []):
        labels = alert.get("labels", {})
        if alert.get("status") == "firing" and labels.get("action") == "rollback":
            if state["models"].rollback(labels.get("alertname", "unknown")):
                rolled_back.append(labels.get("alertname"))
    return {"rolled_back": rolled_back, "active_model": state["models"].active_model}


@app.post("/admin/chaos")
async def chaos(body: Chaos):
    state["models"].chaos = body.model_dump()
    return state["models"].chaos


@app.post("/admin/restore")
async def restore():
    state["models"].restore()
    state["models"].chaos = {"latency_s": 0.0, "error_rate": 0.0}
    return {"active_model": state["models"].active_model}


@app.get("/admin/state")
async def admin_state():
    m = state["models"]
    return {"active": m.active, "model": m.active_model, "chaos": m.chaos, "history": m.history}
