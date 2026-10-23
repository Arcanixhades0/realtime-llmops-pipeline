"""
Fires N concurrent /ask requests and reports latency percentiles.

  python -m eval.load_test --concurrency 500 --requests 2000
"""
import argparse
import asyncio
import json
import random
import time

import httpx
import numpy as np

from src.config import ROOT

QUESTIONS = [
    "What is LoRA and why is it efficient?",
    "How does a diffusion model generate images?",
    "What is retrieval augmented generation?",
    "Explain the attention mechanism in transformers.",
    "What is gradient descent?",
    "How do you evaluate an LLM application?",
    "What is a vector database used for?",
    "What does the encoder in a VAE do?",
]


async def worker(client, queue, results):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        start = time.perf_counter()
        try:
            r = await client.post("/ask", json={"question": random.choice(QUESTIONS)})
            results.append((r.status_code, time.perf_counter() - start))
        except httpx.HTTPError:
            results.append((0, time.perf_counter() - start))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=500)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    queue = asyncio.Queue()
    for _ in range(args.requests):
        queue.put_nowait(1)

    results = []
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=60, limits=limits) as client:
        start = time.perf_counter()
        await asyncio.gather(*[worker(client, queue, results) for _ in range(args.concurrency)])
        wall = time.perf_counter() - start

    ok = [t for code, t in results if code == 200]
    report = {
        "concurrency": args.concurrency,
        "requests": len(results),
        "success_rate": len(ok) / len(results),
        "throughput_rps": round(len(results) / wall, 1),
        "p50_ms": round(float(np.percentile(ok, 50)) * 1000) if ok else None,
        "p95_ms": round(float(np.percentile(ok, 95)) * 1000) if ok else None,
        "p99_ms": round(float(np.percentile(ok, 99)) * 1000) if ok else None,
    }
    print(json.dumps(report, indent=2))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"load_test_{args.concurrency}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
