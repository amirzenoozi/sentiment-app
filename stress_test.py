"""Async load / stress test for the deployed sentiment API.

Fires concurrent requests at a classify endpoint and reports throughput,
latency percentiles (p50/p90/p95/p99) and an error breakdown — the numbers you
need to judge how much load the service handles before latency degrades.

Examples
--------
    # 1000 requests, 20 concurrent, against the server
    poetry run python stress_test.py --host http://65.21.60.8:8022 -n 1000 -c 20

    # Run for 60s at 50 concurrent, Dutch-only (skip the translation path)
    poetry run python stress_test.py --host http://65.21.60.8:8022 -d 60 -c 50 --dutch-only

    # Test the quantized model instead
    poetry run python stress_test.py --host http://65.21.60.8:8022 --endpoint /v2/classify -n 500

Only depends on httpx (already a dependency via FastAPI).
"""
import argparse
import asyncio
import statistics
import time

import httpx

# Mixed payloads. The non-Dutch ones exercise the LibreTranslate path, which is
# slower — use --dutch-only to measure pure model latency without it.
DUTCH_REVIEWS = [
    "Een prima film, kijkt lekker weg.",
    "Wat een saaie en slechte film, zonde van mijn tijd.",
    "Absolute topfilm! Geweldig acteerwerk en mooie beelden.",
    "Het was wel oké, niets bijzonders.",
    "Ontzettend teleurstellend, ik viel bijna in slaap.",
]
NON_DUTCH_REVIEWS = [
    "This movie was surprisingly good, I really enjoyed it.",   # en
    "Ein wirklich langweiliger und schlechter Film.",           # de
]


def percentile(sorted_values, pct):
    """Nearest-rank percentile of an already-sorted list (pct in 0..100)."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(round((pct / 100) * len(sorted_values) + 0.5)) - 1))
    return sorted_values[k]


async def warmup(client, url, payload, n=5):
    """Prime the model so the first (slow) inferences don't skew the results."""
    print(f"[*] Warming up with {n} requests...")
    for _ in range(n):
        try:
            await client.post(url, json=payload, timeout=60.0)
        except Exception:
            pass


async def worker(client, url, reviews, stop, state, timeout):
    """Issue requests back-to-back until the stop condition is met.

    `state` accumulates latencies and status counts shared across workers.
    `stop()` returns True when the run should end (count or deadline reached).
    """
    i = 0
    while not stop():
        review = reviews[(state["issued"]) % len(reviews)]
        state["issued"] += 1
        i += 1
        start = time.perf_counter()
        try:
            resp = await client.post(url, json={"review": review}, timeout=timeout)
            elapsed = time.perf_counter() - start
            state["latencies"].append(elapsed)
            state["status"][resp.status_code] = state["status"].get(resp.status_code, 0) + 1
            if resp.status_code == 200:
                state["ok"] += 1
        except Exception as err:  # timeouts, connection refused, etc.
            state["latencies"].append(time.perf_counter() - start)
            key = type(err).__name__
            state["errors"][key] = state["errors"].get(key, 0) + 1


async def run(args):
    url = args.host.rstrip("/") + args.endpoint
    reviews = DUTCH_REVIEWS if args.dutch_only else (DUTCH_REVIEWS + NON_DUTCH_REVIEWS)

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        # Sanity check that the endpoint is reachable before hammering it.
        try:
            probe = await client.post(url, json={"review": reviews[0]}, timeout=60.0)
            print(f"[+] Endpoint reachable: {url} -> HTTP {probe.status_code}")
        except Exception as err:
            print(f"[-] Cannot reach {url}: {err}")
            return

        if not args.no_warmup:
            await warmup(client, url, {"review": reviews[0]})

        state = {"latencies": [], "status": {}, "errors": {}, "ok": 0, "issued": 0}

        # Stop condition: fixed duration, or fixed request count.
        if args.duration:
            deadline = time.perf_counter() + args.duration
            stop = lambda: time.perf_counter() >= deadline
            mode = f"{args.duration}s duration"
        else:
            stop = lambda: state["issued"] >= args.requests
            mode = f"{args.requests} requests"

        print(f"[*] Load: {mode}, concurrency={args.concurrency}, endpoint={args.endpoint}, "
              f"payloads={'dutch-only' if args.dutch_only else 'mixed'}")
        wall_start = time.perf_counter()
        await asyncio.gather(*[
            worker(client, url, reviews, stop, state, args.timeout)
            for _ in range(args.concurrency)
        ])
        wall = time.perf_counter() - wall_start

    report(state, wall)


def report(state, wall):
    lat = sorted(state["latencies"])
    total = len(lat)
    errors = sum(state["errors"].values())
    print("\n" + "=" * 48)
    print(f"  Total requests   : {total}")
    print(f"  Wall time        : {wall:.2f}s")
    print(f"  Throughput       : {total / wall:.1f} req/s" if wall > 0 else "")
    print(f"  Success (200)    : {state['ok']} ({100 * state['ok'] / total:.1f}%)" if total else "")
    print("  --- latency (ms) ---")
    if lat:
        print(f"  min / mean / max : {lat[0]*1000:.0f} / {statistics.mean(lat)*1000:.0f} / {lat[-1]*1000:.0f}")
        print(f"  p50 / p90 / p95  : {percentile(lat,50)*1000:.0f} / {percentile(lat,90)*1000:.0f} / {percentile(lat,95)*1000:.0f}")
        print(f"  p99              : {percentile(lat,99)*1000:.0f}")
    print("  --- status codes ---")
    for code, cnt in sorted(state["status"].items()):
        print(f"  HTTP {code}         : {cnt}")
    if errors:
        print("  --- transport errors ---")
        for name, cnt in sorted(state["errors"].items()):
            print(f"  {name}: {cnt}")
    print("=" * 48)


def main():
    p = argparse.ArgumentParser(description="Async load/stress test for the sentiment API")
    p.add_argument("--host", default="http://localhost:8022", help="Base URL of the service")
    p.add_argument("--endpoint", default="/v1/classify", help="Classify endpoint to hit")
    p.add_argument("-c", "--concurrency", type=int, default=20, help="Number of concurrent workers")
    p.add_argument("-n", "--requests", type=int, default=500, help="Total requests to send (ignored if --duration set)")
    p.add_argument("-d", "--duration", type=float, default=None, help="Run for this many seconds instead of a fixed count")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    p.add_argument("--dutch-only", action="store_true", help="Send only Dutch reviews (skip the translation path)")
    p.add_argument("--no-warmup", action="store_true", help="Skip the warmup requests")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
