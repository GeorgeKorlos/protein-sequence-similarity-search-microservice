from src.obs.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    model_inference_seconds,
    faiss_search_seconds,
    embed_batch_size,
    errors_total,
    service_build_info,
)


def test_metrics_import_without_error():
    assert http_requests_total is not None
    assert http_request_duration_seconds is not None
    assert model_inference_seconds is not None
    assert faiss_search_seconds is not None
    assert embed_batch_size is not None
    assert errors_total is not None
    assert service_build_info is not None


def test_http_requests_total_increments_with_route_method_status_labels():
    http_requests_total.labels(route="/test", method="GET", status="200").inc()


def test_errors_total_increments_with_error_code_label():
    errors_total.labels(error_code="INVALID_SEQUENCE").inc()


def test_model_inference_seconds_observes_value():
    model_inference_seconds.observe(0.5)


def test_embed_batch_size_observes_value():
    embed_batch_size.observe(0.5)


def test_service_build_info_sets_service_version_model_version_index_version_python_version():
    service_build_info.info(
        {
            "service_version": "0.1.0",
            "model_version": "test",
            "index_version": "test",
            "python_version": "3.11",
        }
    )
