"""
Ingestion worker: Kafka -> embed -> Qdrant.

Reliability decisions (each one is a good interview answer):
  - Offsets are committed only AFTER Qdrant has the vectors. If the worker crashes
    mid-batch, Kafka redelivers the batch ("at-least-once").
  - Point ids come from the event id, so a redelivered batch overwrites the same
    points instead of duplicating them (idempotent writes).
  - Messages that can never succeed (bad JSON, empty text) go to a dead-letter topic
    instead of blocking the partition forever.
  - Run several workers with the same group id: Kafka splits the 3 partitions between them.

Run:  python -m src.worker
Metrics: http://localhost:8001/metrics
"""
import json
import signal
import time
import uuid

from confluent_kafka import Consumer, Producer
from prometheus_client import start_http_server
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import metrics
from src.config import (
    COLLECTION, CONSUMER_GROUP, DLQ_TOPIC, EMBEDDING_DIM, KAFKA_BOOTSTRAP,
    QDRANT_URL, TOPIC, WORKER_BATCH_SIZE, WORKER_METRICS_PORT,
)
from src.embedder import embed

running = True


def stop(*_):
    global running
    running = False


def ensure_collection(client):
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))


def point_id(event_id):
    # Qdrant ids must be ints or UUIDs; derive a stable UUID from our string id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, event_id))


def parse(messages, dlq):
    events = []
    for msg in messages:
        if msg.error():
            metrics.EVENTS_CONSUMED.labels("kafka_error").inc()
            continue
        try:
            event = json.loads(msg.value())
            if not event.get("text"):
                raise ValueError("empty text")
            events.append(event)
        except (ValueError, TypeError) as e:
            dlq.produce(DLQ_TOPIC, value=msg.value(), headers={"error": str(e)})
            metrics.EVENTS_CONSUMED.labels("dead_letter").inc()
    return events


def upsert_with_retry(client, points, attempts=5):
    for attempt in range(attempts):
        try:
            client.upsert(COLLECTION, points=points, wait=True)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            # Exponential backoff: 0.5s, 1s, 2s, 4s
            time.sleep(0.5 * 2 ** attempt)


def main():
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    start_http_server(WORKER_METRICS_PORT)

    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant)

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,     # we commit manually, after Qdrant confirms
    })
    consumer.subscribe([TOPIC])
    dlq = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    print(f"Worker consuming '{TOPIC}' -> Qdrant '{COLLECTION}' (metrics on :{WORKER_METRICS_PORT})")

    while running:
        messages = consumer.consume(num_messages=WORKER_BATCH_SIZE, timeout=1.0)
        if not messages:
            continue

        events = parse(messages, dlq)

        if events:
            with metrics.EMBED_BATCH_SECONDS.time():
                vectors = embed([e["text"] for e in events])

            points = [
                PointStruct(
                    id=point_id(e["id"]),
                    vector=v.tolist(),
                    payload={"text": e["text"], "source": e.get("source"), "page": e.get("page"), "ts": e["ts"]},
                )
                for e, v in zip(events, vectors)
            ]
            upsert_with_retry(qdrant, points)

            now = time.time()
            for e in events:
                metrics.INGEST_LAG_SECONDS.observe(now - e["ts"])
            metrics.EVENTS_CONSUMED.labels("ok").inc(len(events))

        dlq.poll(0)

        # Commit only now: everything in this batch is safely in Qdrant (or the DLQ).
        # A batch of pure errors (e.g. topic not created yet) has no offsets to commit,
        # and committing it raises _NO_OFFSET, which crashed the first version of this worker.
        if any(not m.error() for m in messages):
            consumer.commit(asynchronous=False)

    consumer.close()
    dlq.flush()


if __name__ == "__main__":
    main()
