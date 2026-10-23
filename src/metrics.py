"""
Every Prometheus metric in one place.

The three metric types:
  Counter    only goes up (total requests, total tokens). Prometheus computes rates from it.
  Histogram  counts observations into buckets (latency). Lets Prometheus compute p50/p95/p99.
  Gauge      a value that goes up and down (drift score, which model is active).
"""
from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13)

# ---------------- Ingestion (worker) ----------------
EVENTS_CONSUMED = Counter("ingest_events_total", "Events consumed from Kafka", ["status"])
EMBED_BATCH_SECONDS = Histogram("ingest_embed_batch_seconds", "Time to embed one batch", buckets=LATENCY_BUCKETS)
INGEST_LAG_SECONDS = Histogram(
    "ingest_lag_seconds", "Time from event creation to being searchable in Qdrant",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

# ---------------- Query API ----------------
REQUEST_SECONDS = Histogram("rag_request_seconds", "End-to-end /ask latency", ["model"], buckets=LATENCY_BUCKETS)
RETRIEVAL_SECONDS = Histogram("rag_retrieval_seconds", "Embedding + Qdrant search latency", buckets=LATENCY_BUCKETS)
# Split timings: without these we could only see "retrieval is slow", not WHICH half
EMBED_SECONDS = Histogram("rag_embed_seconds", "Query embedding latency (incl. batching wait)", buckets=LATENCY_BUCKETS)
SEARCH_SECONDS = Histogram("rag_search_seconds", "Qdrant search latency", buckets=LATENCY_BUCKETS)
LLM_SECONDS = Histogram("rag_llm_seconds", "LLM generation latency", ["model"], buckets=LATENCY_BUCKETS)
REQUEST_ERRORS = Counter("rag_errors_total", "Failed /ask requests", ["model", "stage"])
REQUESTS = Counter("rag_requests_total", "All /ask requests", ["model"])
TOKENS = Counter("llm_tokens_total", "LLM tokens used", ["model", "kind"])

QUERY_DRIFT = Gauge("embedding_drift_score", "1 - cosine(recent query centroid, reference centroid)")
ACTIVE_MODEL = Gauge("llm_active_model", "1 for the model currently serving traffic", ["model"])
ROLLBACKS = Counter("llm_rollbacks_total", "Automatic rollbacks to the fallback model", ["reason"])
