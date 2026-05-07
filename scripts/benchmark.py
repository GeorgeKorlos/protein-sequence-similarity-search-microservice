import argparse
import numpy as np
from pathlib import Path
from src.core.corpus_store import CorpusStore


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

    embeddings = np.load(args.embeddings)
    with open(args.ids, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    corpus = CorpusStore(args.corpus, args.ids)
    query_set = np.random.choice(len(embeddings), args.n_queries)
    index_set = np.setdiff1d(np.arange(len(embeddings)), query_set)

    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Corpus size: {len(corpus)}")
    print(f"Query set: {len(query_set)}")
    print(f"Index set: {len(index_set)}")
    print(f"k: {args.k}")
    print(f"Output dir: {args.output_dir}")
    assert len(lines) == len(embeddings)


if __name__ == "__main__":
    main()
