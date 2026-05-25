# Benchmark Results

| Index Type | Params | Recall@k | QPS Mean | QPS Median | QPS Std | Build Time (s) | Index Size (MB) |
|---|---|---|---|---|---|---|---|
| IndexFlatIP | `{}` | 1.0000 | 23.49 | 23.43 | 0.11 | 1.26 | 2667.02 |
| IndexIVFFlat | `{'nlist': 100, 'nprobe': 1}` | 0.7914 | 475.25 | 474.31 | 17.09 | 4.84 | 2671.67 |
| IndexIVFFlat | `{'nlist': 100, 'nprobe': 5}` | 0.9733 | 88.09 | 88.42 | 0.59 | 4.57 | 2671.67 |
| IndexIVFFlat | `{'nlist': 100, 'nprobe': 10}` | 0.9906 | 45.42 | 45.09 | 0.54 | 4.51 | 2671.67 |
| IndexIVFFlat | `{'nlist': 100, 'nprobe': 20}` | 0.9954 | 23.52 | 23.38 | 0.31 | 4.66 | 2671.67 |
| IndexIVFFlat | `{'nlist': 100, 'nprobe': 50}` | 0.9960 | 10.68 | 10.63 | 0.11 | 4.71 | 2671.67 |
| IndexHNSWFlat | `{'M': 32, 'efConstruction': 200, 'efSearch': 16}` | 0.9279 | 6571.89 | 5097.79 | 2101.73 | 366.02 | 2808.77 |
| IndexHNSWFlat | `{'M': 32, 'efConstruction': 200, 'efSearch': 32}` | 0.9480 | 3337.52 | 3333.34 | 26.88 | 364.34 | 2808.77 |
| IndexHNSWFlat | `{'M': 32, 'efConstruction': 200, 'efSearch': 64}` | 0.9564 | 1677.19 | 1675.02 | 8.38 | 372.35 | 2808.77 |
| IndexHNSWFlat | `{'M': 32, 'efConstruction': 200, 'efSearch': 128}` | 0.9655 | 1073.75 | 1106.24 | 52.90 | 372.27 | 2808.77 |
## Cloud Run CPU Inference Latency

**Environment**: GCP Cloud Run, europe-west1, 4 vCPU, 16Gi RAM, CPU-only (float16)
**Date**: 2026-05-25
**Sequence**: MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP (37 aa)
**Requests**: 10 sequential, warm instance

| Metric | Value |
|---|---|
| p50 | 23.4s |
| p95 | 24.8s |
| min | 23.0s |
| max | 24.8s |

vs GPU baseline (RTX 3090, benchmarked during corpus embedding): ~10-15ms per sequence.
CPU is approximately 1500-2500x slower than GPU for single-sequence inference.

Note: float16 on CPU does not improve latency vs float32 — most CPUs lack native
fp16 compute units and perform internal conversion, adding overhead.
