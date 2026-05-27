import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def request_health(base_url: str, timeout: float) -> tuple[int, float]:
    start = time.perf_counter()
    response = requests.get(f"{base_url}/health", timeout=timeout)
    return response.status_code, time.perf_counter() - start


def request_chat(base_url: str, timeout: float, index: int) -> tuple[int, float]:
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "question": f"压测问题 {index}：请用一句话回答当前知识库是否可用。",
            "user_id": 9000 + index,
            "knowledge_base_id": 1,
            "mode": "knowledge",
        },
        timeout=timeout,
    )
    return response.status_code, time.perf_counter() - start


def request_chat_stream(base_url: str, timeout: float, index: int) -> tuple[int, float]:
    start = time.perf_counter()
    with requests.post(
        f"{base_url}/api/chat/stream",
        json={
            "question": f"流式压测问题 {index}：请用一句话回答当前知识库是否可用。",
            "user_id": 19000 + index,
            "knowledge_base_id": 1,
            "mode": "knowledge",
        },
        stream=True,
        timeout=timeout,
    ) as response:
        for _ in response.iter_lines(decode_unicode=True):
            pass
        return response.status_code, time.perf_counter() - start


def run_case(name: str, fn, base_url: str, requests_count: int, concurrency: int, timeout: float) -> None:
    started_at = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(fn, base_url, timeout, index)
            for index in range(requests_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - started_at
    latencies = [latency for _, latency in results]
    status_counts = {}
    for status_code, _ in results:
        status_counts[status_code] = status_counts.get(status_code, 0) + 1

    print(f"\n{name}")
    print(f"requests={requests_count} concurrency={concurrency} elapsed={elapsed:.2f}s")
    print(f"status_counts={status_counts}")
    if latencies:
        print(
            "latency_seconds="
            f"min:{min(latencies):.2f} "
            f"avg:{statistics.mean(latencies):.2f} "
            f"p95:{percentile(latencies, 95):.2f} "
            f"max:{max(latencies):.2f}"
        )


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile_value / 100) * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple load test for zxl-agent FastAPI endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    run_case("GET /health", lambda url, timeout, index: request_health(url, timeout), base_url, args.requests, args.concurrency, args.timeout)
    run_case("POST /api/chat", request_chat, base_url, args.requests, args.concurrency, args.timeout)
    run_case("POST /api/chat/stream", request_chat_stream, base_url, args.requests, args.concurrency, args.timeout)


if __name__ == "__main__":
    main()

