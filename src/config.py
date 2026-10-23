import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "documents"
DLQ_TOPIC = "documents.dlq"          # dead-letter queue for messages that can't be processed
CONSUMER_GROUP = "embedders"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "live_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Source documents for the simulated live feed: reuse the RAG project's PDFs
FEED_DIR = Path(os.getenv("FEED_DIR", ROOT.parent / "Production_RAG_Pipeline" / "data"))

# LLM: primary model, and the fallback we roll back to when the primary misbehaves.
# "fake" simulates an LLM (with adjustable latency) for load tests and failure drills.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "fake")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "qwen/qwen3.8-27b")
FAKE_LLM_LATENCY = float(os.getenv("FAKE_LLM_LATENCY", "0.3"))

WORKER_METRICS_PORT = int(os.getenv("WORKER_METRICS_PORT", "8001"))
WORKER_BATCH_SIZE = 64
