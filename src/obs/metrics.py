from prometheus_client import Counter, Histogram, Info

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["route", "method", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["route", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

faiss_search_seconds = Histogram(
    "faiss_search_seconds",
    "Wall time for asyncio.to_thread(searcher.search)",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

embed_batch_size = Histogram(
    "embed_batch_size",
    "Number of sequences per embed request",
    buckets=[1, 2, 4, 8, 16, 32],
)

model_inference_seconds = Histogram(
    "model_inference_seconds",
    "Wall time for asyncio.to_thread(embedder.embed)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

errors_total = Counter("errors_total", "Total errors by error code", ["error_code"])

service_build_info = Info("service_build", "Service version and build metadata")
