# Ingestion throughput (laptop: CPU-only embeddings, all-MiniLM-L6-v2, full-page events ~1,500 chars)

| Setup | Result | Notes |
|---|---|---|
| Producer → Kafka | **59,951 events/min** (12k events in 12s) | Kafka is not the bottleneck |
| 1 worker draining a 12k backlog | **2,102 events/min** | Embedding on CPU is the bottleneck. RAG eval was also using the CPU. |
| 3 workers draining a 12k backlog | 1,198 events/min (per worker: 4,908 / 1,056 / 6,035) | **Slower than 1 worker** |
| Unique documents in Qdrant after 12k events | 1,013 | Replayed events overwrote the same points (idempotent upserts) |

## Why 3 workers were slower

1. **CPU oversubscription.** Each worker's PyTorch uses all cores by default, so three processes on one
   machine contend for the same cores. Fix: `torch.set_num_threads(cores // workers)`, or (in production)
   one worker per machine/container with CPU limits, or a GPU.
2. **Partition skew.** The message key is the source PDF, and there are only 3 PDFs: two keys hashed to
   the same partition, so one worker got ~4x less work. Fix: a higher-cardinality key (e.g. document id)
   when ordering per source isn't required.

Scaling out only helps when the bottleneck resource scales with it.
