import os
import time
import argparse
import tempfile
import numpy as np
from pathlib import Path
from src.core.index_manager import IndexManager


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/swissprot_embeddings.npy"),
        help="Path to the embeddings NPY file (default: data/swissprot_embeddings.npy)",
    )

    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("data/swissprot_ids.txt"),
        help="Path to the id file (default: data/swissprot_ids.txt)",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/swissprot_clean.csv"),
        help="Path to the input CSV file (default: data/swissprot_clean.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/"),
        help="Directory to the benchmark results (default: reports/)",
    )

    parser.add_argument(
        "--n-queries",
        type=int,
        default=1,
        help="How many vectors to hold out as queries (default: 1)",
    )

    parser.add_argument(
        "--k", type=int, default=5, help="top-k for search (default: 5)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(42)

    embeddings = np.load(args.embeddings, mmap_mode="r")

    with open(args.ids, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    ids = np.array(lines)

    query_set = np.random.choice(len(embeddings), args.n_queries, replace=False)
    index_set = np.setdiff1d(np.arange(len(embeddings)), query_set)

    query_embeddings = embeddings[query_set]
    query_ids = ids[query_set]

    index_embeddings = embeddings[index_set]
    index_ids = ids[index_set]

    ground_truth_index = IndexManager(index_type="IndexFlatIP", dim=1280, params={})
    ground_truth_index.build(index_embeddings)

    gt_distances, gt_indices = ground_truth_index.index.search(query_embeddings, args.k)  # type: ignore

    ground_truth = {}

    for i, row in enumerate(gt_indices):

        accessions = {index_ids[int(pos)] for pos in row}
        ground_truth[i] = accessions

    assert len(ground_truth) == len(query_embeddings)

    results = []

    start_time = time.time()
    ground_truth_index.build(index_embeddings)
    build_time = time.time() - start_time

    search_times = []
    qps_runs = []

    for _ in range(3):
        start_time = time.time()

        distances, indices = ground_truth_index.index.search(query_embeddings, args.k)  # type: ignore

        wall_time = time.time() - start_time

        search_times.append(wall_time)
        qps = args.n_queries / wall_time
        qps_runs.append(qps)

    qps_mean = np.mean(qps_runs)
    qps_median = np.median(qps_runs)
    qps_std = np.std(qps_runs)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index"

        ground_truth_index.save(index_path)

        faiss_file = index_path / "index.faiss"

        size_bytes = os.path.getsize(faiss_file)
        size_mb = size_bytes / (1024 * 1024)

        print(f"Index size: {size_mb:.2f} MB")

    results.append(
        {
            "index_type": "IndexFlatIP",
            "build_time": build_time,
            "qps_mean": qps_mean,
            "qps_median": qps_median,
            "qps_std": qps_std,
            "index_size_mb": size_mb,
            "recall_at_k": 1.0,
        }
    )

    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Query set: {len(query_set)}")
    print(f"Index set: {len(index_set)}")
    print(f"k: {args.k}")
    print(f"Output dir: {args.output_dir}")
    assert len(lines) == len(embeddings)


if __name__ == "__main__":
    main()
