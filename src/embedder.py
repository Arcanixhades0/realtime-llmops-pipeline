import asyncio
import threading

from src.config import EMBEDDING_MODEL

_model = None
_lock = threading.Lock()


def get_model():
    # Load once per process; the lock stops concurrent first requests from loading it twice
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts):
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64)


class EmbedBatcher:
    """
    Micro-batching for query embeddings.

    Without it, 500 concurrent requests made 500 separate encode() calls that
    fought over the CPU (average retrieval 7.6s under load). Embedding 64 texts
    in ONE call costs barely more than embedding one, so we wait up to
    `max_wait_ms` for other requests to arrive and embed them together.
    The same idea is how GPU inference servers get their throughput.
    """

    def __init__(self, max_batch=64, max_wait_ms=10):
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = asyncio.Queue()
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()

    async def embed(self, text):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def _run(self):
        while True:
            batch = [await self.queue.get()]
            deadline = asyncio.get_running_loop().time() + self.max_wait

            # Keep collecting until the batch is full or the wait is over
            while len(batch) < self.max_batch:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break

            texts = [text for text, _ in batch]
            try:
                vectors = await asyncio.to_thread(embed, texts)
                for (_, future), vector in zip(batch, vectors):
                    future.set_result(vector)
            except Exception as e:
                for _, future in batch:
                    future.set_exception(e)
