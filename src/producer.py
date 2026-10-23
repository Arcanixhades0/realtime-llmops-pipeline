"""
Simulated live feed: streams paragraphs from the PDFs into Kafka at a fixed rate.

In a real system these events would be support tickets, news, logs, chat messages...

Run:  python -m src.producer --rate 200 --seconds 60
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from confluent_kafka import Producer

from src.config import FEED_DIR, KAFKA_BOOTSTRAP, TOPIC


def paragraphs(feed_dir=FEED_DIR, min_chars=200):
    from pypdf import PdfReader

    for pdf in sorted(Path(feed_dir).glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(str(pdf)).pages):
            text = page.extract_text() or ""
            for para in text.split("\n\n") if "\n\n" in text else [text]:
                para = " ".join(para.split())
                if len(para) >= min_chars:
                    yield {"text": para[:1500], "source": pdf.name, "page": page_no}


def make_event(item):
    return {
        # Deterministic id: replaying the same paragraph overwrites instead of duplicating
        "id": hashlib.sha1(f"{item['source']}|{item['page']}|{item['text'][:100]}".encode()).hexdigest(),
        "ts": time.time(),
        **item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=200, help="events per second")
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "linger.ms": 20,          # wait up to 20ms to send messages in batches (higher throughput)
        "acks": "all",            # only count a message as sent once the broker has stored it
        "enable.idempotence": True,  # retries never create duplicate messages
    })

    items = list(paragraphs())
    print(f"Loaded {len(items)} paragraphs; streaming {args.rate:.0f}/s for {args.seconds:.0f}s")

    sent, start = 0, time.perf_counter()
    interval = 1 / args.rate

    while time.perf_counter() - start < args.seconds:
        event = make_event(items[sent % len(items)])
        # Key by source, so all paragraphs of one document land in the same partition (kept in order)
        producer.produce(TOPIC, key=event["source"], value=json.dumps(event))
        producer.poll(0)
        sent += 1

        # Pace to the target rate
        sleep_for = start + sent * interval - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    producer.flush()
    elapsed = time.perf_counter() - start
    print(f"Sent {sent} events in {elapsed:.1f}s ({sent / elapsed * 60:,.0f} events/min)")


if __name__ == "__main__":
    main()
