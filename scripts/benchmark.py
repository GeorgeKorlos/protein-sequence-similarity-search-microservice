import os
import time
import json
import argparse
import tempfile
import numpy as np
from pathlib import Path
from src.core.index_manager import IndexManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/swissprot_embeddings.npy")
    )
    parser.add_argument("--ids", type=Path, default=Path("data/swissprot_ids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/"))
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def compute_recall(retrieved, ground_truth, index_ids):
    recalls = []
    for i, row in enumerate(retrieved):
        predicted = {index_ids[int(pos)] for pos in row}
        recalls.append(len(predicted & ground_truth[i]) / len(ground_truth[i]))
    return float(np.mean(recalls))


def measure_qps(index, query_embeddings, k, n_queries):
    runs = []
    for _ in range(3):
        start = time.time()
        index.search(query_embeddings, k)
        runs.append(n_queries / (time.time() - start))
    return float(np.mean(runs)), float(np.median(runs)), float(np.std(runs))


def measure_index_size(index_manager):
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index"
        index_manager.save(index_path)
        return float(os.path.getsize(index_path / "index.faiss") / (1024 * 1024))


def benchmark_index(
    index_type,
    params,
    index_embeddings,
    query_embeddings,
    ground_truth,
    index_ids,
    k,
    n_queries,
    ef=None,
):
    manager = IndexManager(index_type, 1280, params)
    start = time.time()
    manager.build(index_embeddings)
    build_time = time.time() - start
    if ef is not None:
        manager.index.hnsw.efSearch = ef  # type: ignore
    _, indices = manager.index.search(query_embeddings, k)  # type: ignore
    recall = compute_recall(indices, ground_truth, index_ids)
    qps_mean, qps_median, qps_std = measure_qps(
        manager.index, query_embeddings, k, n_queries
    )
    out_params = dict(params)
    if ef is not None:
        out_params["efSearch"] = ef
    return {
        "index_type": index_type,
        "params": out_params,
        "recall_at_k": recall,
        "qps_mean": qps_mean,
        "qps_median": qps_median,
        "qps_std": qps_std,
        "build_time_s": build_time,
        "index_size_mb": measure_index_size(manager),
    }


def main():
    args = parse_args()
    np.random.seed(42)

    embeddings = np.load(args.embeddings, mmap_mode="r")
    with open(args.ids) as f:
        ids = np.array([line.strip() for line in f if line.strip()])
    assert len(ids) == len(embeddings)

    query_set = np.random.choice(len(embeddings), args.n_queries, replace=False)
    index_set = np.setdiff1d(np.arange(len(embeddings)), query_set)
    query_embeddings = np.ascontiguousarray(embeddings[query_set], dtype=np.float32)
    index_embeddings = embeddings[index_set]
    index_ids = ids[index_set]

    results = []

    # FlatIP — exact ground truth
    flat = IndexManager("IndexFlatIP", 1280, {})
    start = time.time()
    flat.build(index_embeddings)
    flat_build_time = time.time() - start
    _, gt_indices = flat.index.search(query_embeddings, args.k)  # type: ignore
    ground_truth = {
        i: {index_ids[int(pos)] for pos in row} for i, row in enumerate(gt_indices)
    }
    flat_qps = measure_qps(flat.index, query_embeddings, args.k, args.n_queries)
    results.append(
        {
            "index_type": "IndexFlatIP",
            "params": {},
            "recall_at_k": 1.0,
            "qps_mean": flat_qps[0],
            "qps_median": flat_qps[1],
            "qps_std": flat_qps[2],
            "build_time_s": flat_build_time,
            "index_size_mb": measure_index_size(flat),
        }
    )

    for nprobe in [1, 5, 10, 20, 50]:
        results.append(
            benchmark_index(
                "IndexIVFFlat",
                {"nlist": 100, "nprobe": nprobe},
                index_embeddings,
                query_embeddings,
                ground_truth,
                index_ids,
                args.k,
                args.n_queries,
            )
        )

    for ef in [16, 32, 64, 128]:
        results.append(
            benchmark_index(
                "IndexHNSWFlat",
                {"M": 32, "efConstruction": 200},
                index_embeddings,
                query_embeddings,
                ground_truth,
                index_ids,
                args.k,
                args.n_queries,
                ef=ef,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Benchmark Results",
        "",
        "| Index Type | Params | Recall@k | QPS Mean | QPS Median | QPS Std | Build Time (s) | Index Size (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['index_type']} | `{r['params']}` | {r['recall_at_k']:.4f} | "
            f"{r['qps_mean']:.2f} | {r['qps_median']:.2f} | {r['qps_std']:.2f} | "
            f"{r['build_time_s']:.2f} | {r['index_size_mb']:.2f} |"
        )
    (args.output_dir / "benchmark_results.md").write_text("\n".join(lines))

    print(
        f"\nBenchmark complete — {embeddings.shape[0]} embeddings, "
        f"{len(query_set)} queries, k={args.k}"
    )
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
