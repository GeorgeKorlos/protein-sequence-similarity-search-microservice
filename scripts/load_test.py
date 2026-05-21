import json
import time
import httpx
import pathlib
import asyncio
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load testing utility for search and embedding API endpoints. "
            "Sends concurrent HTTP requests to measure latency, throughput, and stability."
        )
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (e.g., http://localhost:8000)",
    )

    parser.add_argument(
        "--n-requests",
        type=int,
        default=200,
        help="Total number of requests to send during the load test",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent workers sending requests in parallel",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["search", "embed"],
        default="search",
        help="Target API endpoint to test: 'search' for retrieval or 'embed' for embeddings",
    )

    return parser.parse_args()


def sample_sequences(n: int, endpoint: str) -> list:
    df = pd.read_csv("data/swissprot_clean.csv")

    sampled = df.sample(n=n, random_state=42)

    if endpoint == "search":
        return [{"sequence": seq} for seq in sampled["sequence"].tolist()]

    return [{"sequences": seq} for seq in sampled["sequence"].tolist()]


async def send_request(
    client: httpx.AsyncClient, url: str, payload: dict, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:

        t0 = time.perf_counter()

        try:
            response = await client.post(url, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000

            return {
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "error": None,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000

            return {"status_code": None, "latency_ms": latency_ms, "error": str(e)}


async def run(args):
    payloads = sample_sequences(n=args.n_requests, endpoint=args.endpoint)

    url = args.url + "/" + args.endpoint

    async with httpx.AsyncClient(timeout=120.0) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        wall_start = time.perf_counter()
        tasks = [
            send_request(
                client=client,
                url=url,
                payload=payload,
                semaphore=semaphore,
            )
            for payload in payloads
        ]
        results = await asyncio.gather(*tasks)

        wall_end = time.perf_counter()
        total_time = wall_end - wall_start

        successful = [r for r in results if r["error"] is None]
        failed = [r for r in results if r["error"] is not None]

        latencies = [r["latency_ms"] for r in successful]

        avg_latency = np.mean(latencies) if latencies else 0
        p50 = np.percentile(latencies, 50) if latencies else 0
        p95 = np.percentile(latencies, 95) if latencies else 0
        p99 = np.percentile(latencies, 99) if latencies else 0

        throughput = len(results) / total_time if total_time > 0 else 0

        print("\n=== Load Test Summary ===")
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Wall time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Average latency: {avg_latency:.2f} ms")
        print(f"P50 latency: {p50:.2f} ms")
        print(f"P95 latency: {p95:.2f} ms")
        print(f"P99 latency: {p99:.2f} ms")

        pathlib.Path("reports").mkdir(exist_ok=True)

        with open(
            "reports/load_test_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
