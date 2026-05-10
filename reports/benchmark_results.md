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