# Real-Time AI Pipeline with LLMOps Monitoring

Streaming RAG: documents arrive on Kafka, get embedded and indexed in Qdrant within seconds, and are
immediately searchable through an instrumented query API. Prometheus + Grafana watch latency, errors,
tokens and embedding drift; critical alerts trigger an automatic rollback to a fallback model.

```
 producer ──▶ Kafka "documents" (3 partitions) ──▶ worker(s) ──▶ Qdrant
                    │                                │  embed batch, upsert, THEN commit offset
                    └──▶ "documents.dlq" ◀───────────┘  (bad messages)

 user ──▶ FastAPI /ask ──▶ embed ──▶ Qdrant search ──▶ LLM (primary | fallback)
              │
          /metrics ◀── Prometheus (5s) ──▶ alert rules ──▶ Alertmanager ──▶ POST /admin/alerts
              │                                                               └─▶ rollback to fallback
           Grafana dashboard: latency p50/p95/p99, errors, tokens/min, drift, ingest rate, ingest lag
```

## Run

```bash
pip install -r requirements.txt
docker compose up -d                              # Kafka, Qdrant, Prometheus, Alertmanager, Grafana

python -m src.worker                              # terminal 1 (run 2-3 of these to scale)
uvicorn src.api:app --port 8000                   # terminal 2
python -m src.producer --rate 200 --seconds 120   # terminal 3

curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is LoRA?\"}"
```
Grafana: http://localhost:3000 → LLMOps folder.

## Measure

```bash
python -m eval.load_test --concurrency 500 --requests 2000   # → reports/load_test_500.json
python -m eval.failure_drill                                 # → reports/failure_drill.json
python -m pytest -q
```

## Results

Measured on a laptop (CPU-only embeddings). Raw data in `reports/`.

| Test | Result |
|---|---|
| Kafka ingest | **~60,000 events/min** (12k events in 12s) |
| Embedding worker | ~2,100 events/min per worker. 3 workers on one machine were *slower* (CPU oversubscription + partition skew, see `ingestion_throughput.md`) |
| Idempotency | 12,000 replayed events → 1,013 unique points in Qdrant |
| 500 concurrent `/ask` | **99.2% success**, 52 req/s (simulated LLM). Was 58.5% before micro-batching + Qdrant connection pooling |
| Failure drill (+3s injected latency) | **automatic rollback in 27.9s**, no human action (`failure_drill.json`) |

At 500 concurrent requests, p95 latency is ~19s: each request needs ~1.3s of work, and the server
handles ~52/s, so the rest is queueing (Little's Law: 500 / 52 ≈ 9.6s average wait). More capacity,
not code changes, fixes that.

## Design decisions (interview answers)

- **At-least-once + idempotent writes:** offsets committed only after Qdrant confirms; point ids derived from event ids, so redelivery overwrites instead of duplicating. (Exactly-once is expensive; idempotency gets the same result.)
- **Dead-letter queue:** a malformed message would otherwise be retried forever and block its partition.
- **Partition key = source document:** keeps one document's paragraphs in order; 3 partitions = up to 3 workers in parallel.
- **Histograms, not averages:** averages hide the slow tail. p95/p99 come from histogram buckets.
- **Drift = 1 − cos(recent query centroid, document centroid):** rises when users ask about things the index doesn't cover. Warning only; no automated action, because it needs a human.
- **Auto-rollback only on clear signals** (p95 latency, error rate) with `for: 10s` so one slow request doesn't flip models.
- **Chaos injection:** you can't claim recovery works until you break it on purpose and time it.
