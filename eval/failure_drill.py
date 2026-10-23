"""
Failure drill: break the primary model on purpose and time the recovery.

  1. Send steady traffic.
  2. Inject 3s of extra latency into the primary model.
  3. Prometheus sees p95 > 2s -> alert fires -> Alertmanager calls /admin/alerts
     -> API switches to the fallback model.
  4. Measure:
       time_to_detect_and_rollback : injection -> rollback (fully automatic)
       time_to_recover             : injection -> requests are fast again

Needs the full stack running (docker compose up, worker, api).

  python -m eval.failure_drill
"""
import asyncio
import json
import time

import httpx

from src.config import ROOT

URL = "http://127.0.0.1:8000"
PROMETHEUS = "http://127.0.0.1:9090"
FAST_ENOUGH_S = 1.0


async def wait_for_quiet(client):
    """
    Start only when no rollback alert is pending or firing in Prometheus.
    Otherwise an alert left over from the previous drill fires again during
    warm-up and the measured "rollback time" is negative (this happened).
    """
    async with httpx.AsyncClient(base_url=PROMETHEUS, timeout=10) as prom:
        for _ in range(120):
            alerts = (await prom.get("/api/v1/alerts")).json()["data"]["alerts"]
            if not any(a["labels"].get("action") == "rollback" for a in alerts):
                return
            await asyncio.sleep(2)
    raise RuntimeError("Rollback alerts never cleared")


async def traffic(client, stop, log):
    while not stop.is_set():
        # Log when each request STARTED: a request sent before the failure was injected
        # says nothing about recovery (the first version of this drill got that wrong)
        started_at, start = time.time(), time.perf_counter()
        try:
            r = await client.post("/ask", json={"question": "What is retrieval augmented generation?"})
            log.append((started_at, time.perf_counter() - start, r.json().get("model") if r.status_code == 200 else None))
        except httpx.HTTPError:
            log.append((started_at, time.perf_counter() - start, None))
        await asyncio.sleep(0.05)


async def main():
    async with httpx.AsyncClient(base_url=URL, timeout=60) as client:
        await client.post("/admin/restore")
        print("Waiting for any old alerts to clear...")
        await wait_for_quiet(client)
        await client.post("/admin/restore")

        stop, log = asyncio.Event(), []
        tasks = [asyncio.create_task(traffic(client, stop, log)) for _ in range(20)]

        print("Warm-up 20s of healthy traffic...")
        await asyncio.sleep(20)

        if (await client.get("/admin/state")).json()["active"] != "primary":
            raise RuntimeError("Rolled back during warm-up: an alert fired before the failure was injected")

        injected_at = time.time()
        await client.post("/admin/chaos", json={"latency_s": 3.0})
        print("Injected +3s latency into the primary model. Waiting for automatic rollback...")

        rollback_at = None
        while time.time() - injected_at < 180:
            st = (await client.get("/admin/state")).json()
            if st["active"] == "fallback":
                rollback_at = next(h["at"] for h in reversed(st["history"]) if h["event"] == "rollback")
                break
            await asyncio.sleep(1)

        # Recovered = first request (started after injection) that begins a run of 10 fast requests
        await asyncio.sleep(10)
        stop.set()
        await asyncio.gather(*tasks)

        after = sorted(e for e in log if e[0] > injected_at)
        recovered_at = None
        for i in range(len(after) - 10):
            if all(lat < FAST_ENOUGH_S for _, lat, _ in after[i:i + 10]):
                recovered_at = after[i][0]
                break

        report = {
            "injected_latency_s": 3.0,
            "time_to_detect_and_rollback_s": round(rollback_at - injected_at, 1) if rollback_at else None,
            "time_to_recover_s": round(recovered_at - injected_at, 1) if recovered_at else None,
            "requests_during_drill": len(after),
            "slow_requests_during_incident": sum(1 for _, lat, _ in after if lat >= FAST_ENOUGH_S),
        }
        print(json.dumps(report, indent=2))
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports" / "failure_drill.json").write_text(json.dumps(report, indent=2))

        await client.post("/admin/restore")


if __name__ == "__main__":
    asyncio.run(main())
