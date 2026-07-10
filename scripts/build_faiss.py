import os
import json
import time
import logging
import argparse
import numpy as np
import faiss

from pathlib import Path
from src.core.index_manager import IndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/build_faiss.log")],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build and save a production FAISS index.")
    p.add_argument(
        "--embeddings", type=Path, default=Path("data/swissprot_embeddings.npy")
    )
    p.add_argument("--ids", type=Path, default=Path("data/swissprot_ids.txt"))
    p.add_argument(
        "--embedding-config", type=Path, default=Path("data/embedding_config.json")
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/indexes/ivf"))
    p.add_argument(
        "--index-type",
        default="IndexIVFFlat",
        choices=["IndexFlatIP", "IndexIVFFlat", "IndexHNSWFlat"],
    )
    p.add_argument("--nlist", type=int, default=100)
    p.add_argument("--nprobe", type=int, default=10)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    return p.parse_args()


def validate_embeddings(embeddings, ids):
    log.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    log.info(f"IDs count: {len(ids)}")
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
    assert len(ids) == len(
        embeddings
    ), f"ID count {len(ids)} != embedding count {len(embeddings)}"

    sample_idx = np.random.choice(
        len(embeddings), min(100, len(embeddings)), replace=False
    )
    norms = np.linalg.norm(embeddings[sample_idx], axis=1)
    max_dev = np.max(np.abs(norms - 1.0))
    assert max_dev < 1e-3, f"Not L2-normalized. Max norm deviation: {max_dev:.6f}"
    log.info(f"L2 norm check passed — max deviation: {max_dev:.6f}")


def build_params(args):
    if args.index_type == "IndexIVFFlat":
        return {"nlist": args.nlist, "nprobe": args.nprobe}
    if args.index_type == "IndexHNSWFlat":
        return {"M": args.m, "efConstruction": args.ef_construction}
    return {}


def verify_self_search(index, embeddings, nprobe):
    log.info("Running self-search verification...")
    if nprobe is not None:
        index.nprobe = nprobe
    probe_idx = min(100, len(embeddings) - 1)
    query = np.ascontiguousarray(
        embeddings[probe_idx : probe_idx + 1], dtype=np.float32
    )
    scores, indices = index.search(query, 10)
    top1_idx, top1_score = int(indices[0][0]), float(scores[0][0])
    assert (
        top1_idx == probe_idx
    ), f"Self-search failed: expected {probe_idx}, got {top1_idx}"
    assert top1_score > 0.999, f"Self-search score too low: {top1_score:.6f}"
    log.info(f"Self-search passed — rank 1 idx: {top1_idx}, score: {top1_score:.6f}")


def spot_check_recall(index, embeddings, nprobe, n_queries=100, k=10):
    log.info(f"Running recall spot-check ({n_queries} queries, k={k})...")
    np.random.seed(42)
    corpus = np.ascontiguousarray(embeddings, dtype=np.float32)
    query_idx = np.random.choice(len(corpus), n_queries, replace=False)
    query_emb = corpus[query_idx]

    flat = faiss.IndexFlatIP(corpus.shape[1])
    flat.add(corpus)  # type: ignore
    _, gt_indices = flat.search(query_emb, k)  # type: ignore

    if nprobe is not None:
        index.nprobe = nprobe
    _, prod_indices = index.search(query_emb, k)

    recalls = [
        len(set(gt_indices[i]) & set(prod_indices[i])) / k for i in range(n_queries)
    ]
    recall = float(np.mean(recalls))
    log.info(f"Recall spot-check: {recall:.4f}")
    return recall


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("build_faiss.py — Production Index Build")
    log.info("=" * 60)
    log.info(f"Index type : {args.index_type}")
    log.info(f"Output dir : {args.output_dir}")

    with open(args.embedding_config) as f:
        emb_config = json.load(f)
    model_version = emb_config["model_version"]
    corpus_version = emb_config["corpus_version"]
    log.info(f"Model version  : {model_version}")
    log.info(f"Corpus version : {corpus_version}")

    log.info(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    log.info(f"Loading IDs from {args.ids}...")
    with open(args.ids) as f:
        ids = np.array([line.strip() for line in f if line.strip()])

    validate_embeddings(embeddings, ids)

    params = build_params(args)
    log.info(f"Building {args.index_type} with params: {params}")

    manager = IndexManager(
        index_type=args.index_type, dim=embeddings.shape[1], params=params
    )
    manager.model_version = model_version
    manager.corpus_version = corpus_version

    embeddings_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)
    start = time.time()
    manager.build(embeddings_contiguous)
    build_time = time.time() - start
    log.info(f"Build complete — {manager.index.ntotal} vectors, {build_time:.2f}s")  # type: ignore

    log.info(f"Saving index to {args.output_dir}...")
    manager.save(args.output_dir)

    meta_path = args.output_dir / "index_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["build_duration_seconds"] = round(build_time, 2)
    meta["production"] = True
    meta["index_file"] = str(args.output_dir / "index.faiss")
    meta["n_vectors"] = int(manager.index.ntotal)  # type: ignore
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("index_meta.json written")

    nprobe = params.get("nprobe") if args.index_type == "IndexIVFFlat" else None
    verify_self_search(manager.index, embeddings_contiguous, nprobe)
    if args.index_type != "IndexFlatIP":
        recall = spot_check_recall(manager.index, embeddings_contiguous, nprobe)
        log.info(f"Spot-check recall: {recall:.4f}")

    index_size_mb = os.path.getsize(args.output_dir / "index.faiss") / (1024 * 1024)
    log.info("=" * 60)
    log.info("Build summary")
    log.info(f"  Index type   : {args.index_type}")
    log.info(f"  Params       : {params}")
    log.info(f"  Vectors      : {manager.index.ntotal:,}")  # type: ignore
    log.info(f"  Build time   : {build_time:.2f}s")
    log.info(f"  Index size   : {index_size_mb:.1f} MB")
    log.info(f"  Output       : {args.output_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
