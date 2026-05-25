# Cloud Deployment — GCP Cloud Run

## Platform Choice

**Choice:** GCP Cloud Run, europe-west1  

### Options considered
1. GCP Cloud Run — serverless containers, per-request billing, no infra management
2. GCP Compute Engine — persistent VM, full control, always-on cost
3. AWS Lambda — function-level, 10 GB storage limit rules out model weights + index simultaneously
4. Fly.io — simpler DX, less enterprise credibility for portfolio purposes

### Rationale
Cloud Run handles containerized workloads with zero infra management. Per-request
billing suits a demo workload with infrequent queries. europe-west1 chosen for
geographic proximity.

AWS Lambda rejected: 10 GB ephemeral storage limit cannot fit ESM-2 weights (1.3 GB)
+ FAISS index (2.7 GB) simultaneously. Compute Engine rejected: always-on cost not
justified for a portfolio demo with sporadic traffic. Fly.io rejected: less recognizable
to academic reviewers than GCP.

## Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Memory | 16Gi | Measured peak at startup: 8206 MiB; 8Gi limit exceeded in testing |
| CPU | 4 vCPU | Parallelizes tokenization; ESM-2 forward pass is single-threaded |
| Concurrency | 1 | CPU inference cannot serve concurrent requests without latency degradation |
| Min instances | 0 | Cold starts accepted; demo workload |
| Max instances | 1 | Cost control; single instance sufficient |
| Timeout | 300s | Covers worst-case long sequence on CPU |
| Region | europe-west1 | Geographic proximity |

### Memory footprint (measured)

| Component | Size |
|---|---|
| ESM-2 650M (float16, CPU) | ~1.3 GB |
| FAISS IVFFlat index | ~2.7 GB |
| Corpus metadata (547K rows) | ~50 MB |
| Runtime buffers + Python overhead | ~4.1 GB |
| **Peak at startup** | **~8.2 GB** |

Initial deployment used 8Gi limit. Container was OOM-killed at 8206 MiB on two
consecutive revisions. Limit increased to 16Gi. Actual headroom at 16Gi is ~7.8 GB.

Note: float16 weights were used to reduce model footprint from ~2.6 GB (float32)
to ~1.3 GB. This did not improve inference latency — CPUs lack native float16
compute units and perform internal conversion, adding overhead.

## Index Delivery

The FAISS index (2.7 GB) is stored in GCS and downloaded to `/tmp/index` at
container startup via `scripts/entrypoint.sh` + `scripts/gcs_download.py`.
This adds ~60-90s to cold start time but keeps the Docker image lean and allows
index updates without rebuilding the image.

**GCS URI:** `gs://protein-search-497311-index/ivf/`

Alternative considered: bake index into Docker image. Rejected: image would exceed
6 GB, push times prohibitive, and index updates would require a full image rebuild.

The download script uses the Python `google-cloud-storage` SDK rather than the
`gsutil` CLI — the base `python:3.10-slim` image does not include the gcloud CLI,
and installing it would add ~400 MB to the image.

## CPU Inference Trade-off

Cloud Run standard tier is CPU-only (GPU instances available only in preview,
not in europe-west1 at time of deployment).

**Measured latency** (europe-west1, 4 vCPU, 16Gi, warm instance, 37 aa sequence):

| Metric | Value |
|---|---|
| p50 | 23.4s |
| p95 | 24.8s |
| min | 23.0s |
| max | 24.8s |

**GPU baseline** (RTX 3090, benchmarked during corpus embedding): ~10-15 ms/sequence

CPU is approximately **1500-2500x slower** than GPU for single-sequence ESM-2 inference.
For a portfolio demonstration this is acceptable — the service is functionally correct
and all endpoints return valid responses. For production use, Cloud Run GPU (preview)
or a persistent GPU instance would be required to meet interactive latency SLAs.

## Environment Variables

Set via `--set-env-vars` at deploy time. No secrets required — no API keys, no
database credentials. `.env` is not used in Cloud Run.

| Variable | Value |
|---|---|
| `model_tag` | `facebook/esm2_t33_650M_UR50D` |
| `device` | `cpu` |
| `corpus_path` | `/app/data/swissprot_clean.csv` |
| `ids_path` | `/app/data/swissprot_ids.txt` |
| `max_batch_size` | `32` |
| `max_payload_size` | `50000` |
| `max_top_k` | `50` |
| `GCS_INDEX_URI` | `gs://protein-search-497311-index/ivf` |
| `INDEX_PATH` | `/tmp/index` |

## `/metrics` Access

Publicly accessible on the Cloud Run deployment. Acceptable for a portfolio
demonstration — no sensitive data is exposed in Prometheus metrics. In a production
deployment, `/metrics` should be IAM-gated using a service account with
`roles/run.invoker`, or exposed only via a VPC-internal Prometheus scraper.

## Endpoint Verification

**Service URL:** https://protein-search-699950260063.europe-west1.run.app

| Endpoint | Expected | Verified |
|---|---|---|
| `GET /health` | 200, all version fields | ✓ |
| `GET /ready` | 200, `ready: true` | ✓ |
| `POST /search` | 200, top-k results | ✓ |
| `POST /embed` | 200, (1, 1280) shape | ✓ |
| `GET /metrics` | 200, Prometheus text | ✓ |
