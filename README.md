# Protein Sequence Similarity Search Microservice

Fast semantic similarity search over 547K UniProt proteins using ESM-2 embeddings and FAISS indexing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼────┐            ┌──────▼───┐
    │ /embed │            │ /search  │
    └───┬────┘            └──────┬───┘
        │                        │
        │    ┌──────────────┐    │
        └───►│ ESM2-650M    │◄───┘
             │ Embedder     │
             └──────┬───────┘
                    │
                    ▼
              ┌────────────┐
              │ FAISS      │
              │ Index      │
              └──────┬─────┘
                     │
              ┌──────▼──────┐
              │ CorpusStore │
              │ (UniProt)   │
              └─────────────┘
```

For detailed architecture, see [notebooks/02_embedding_analysis.ipynb](notebooks/02_embedding_analysis.ipynb).

## Quickstart

```bash
# Index must be built first — see RUNBOOK.md
docker-compose -f infra/docker-compose.yaml up
```

Service runs on `http://localhost:8000` with OpenAPI docs at `/docs`.

## API Endpoints

| Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/embed` | POST | Generate embeddings for protein sequences | `{"sequences": ["MKVL..."], ...}` | `{"embeddings": [...], "model_version": "...", "request_id": "..."}` |
| `/search` | POST | Find similar sequences in corpus | `{"sequence": "MKVL...", "top_k": 10}` | `{"results": [{rank, accession, score, organism, keywords, go_terms}, ...], "query_length": N}` |
| `/health` | GET | Service health status | — | `{"status": "ok", "service_version": "...", "model_version": "...", ...}` |
| `/ready` | GET | Deployment readiness (model & index loaded) | — | `{"ready": true, "model_loaded": true, "index_loaded": true}` |
| `/metrics` | GET | Prometheus metrics (inference time, QPS, errors) | — | Prometheus text format |

## Benchmark Results

Index performance on 500 queries (ESM-2 embeddings, k=10 recall):

| Index Type | Configuration | Recall@10 | QPS (Mean) | Build Time | Size |
|---|---|---|---|---|---|
| **Flat** | — | 1.0000 | 23.49 | 1.26s | 2667 MB |
| **IVF** | nprobe=5 | 0.9733 | 88.09 | 4.57s | 2672 MB |
| **IVF** | nprobe=10 | 0.9906 | 45.42 | 4.51s | 2672 MB |
| **HNSW** | efSearch=64 | 0.9564 | 1677.19 | 372.35s | 2809 MB |

See [reports/benchmark_results.md](reports/benchmark_results.md) for full comparison matrix.

## P5 Integration

These embeddings and corpus metadata are consumed by P5 for retrieval-augmented drug-target interaction prediction and embedding ablation studies; see [reports/p5_embedding_handoff.md](reports/p5_embedding_handoff.md) for data provenance and loading instructions.
