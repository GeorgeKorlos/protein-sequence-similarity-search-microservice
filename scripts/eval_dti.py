import ast
import json
import faiss
import hashlib
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from src.core.embedder import ESM2Embedder, PhysiochemicalEmbedder, RandomEmbedder

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/swissprot_clean.csv"),
        help="Path to the input CSV file (default: data/swissprot_clean.csv)",
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/swissprot_embeddings.npy"),
        help="Path to the embeddings file (default: data/swissprot_embeddings.npy)",
    )

    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("data/swissprot_ids.txt"),
        help="Path to the ids text file (default: data/swissprot_ids.txt)",
    )

    parser.add_argument(
        "--embedder",
        type=str,
        choices=["esm2", "physiochemical", "random", "all"],
        default="all",
        help="Which embedder to use (default: all)",
    )

    parser.add_argument(
        "--n-queries",
        type=int,
        default=500,
        help="Number of query samples used for evaluation (default: 500)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors retrieved per query for metric computation (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/"),
        help="Directory to the DTI evaluation results (default: reports/)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def load_corpus(corpus_path: Path) -> pd.DataFrame:
    df = pd.read_csv(corpus_path)
    df["go_terms"] = df["go_terms"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df = df[df["go_terms"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    return df.reset_index(drop=True)


def get_corpus_embeddings(
    embedder_name: str,
    embedder,
    df: pd.DataFrame,
    embeddings_path: Path,
    ids_path: Path,
) -> np.ndarray:
    if embedder_name == "esm2":
        embeddings = np.load(embeddings_path, mmap_mode="r")
        with open(ids_path, "r") as f:
            all_ids = [line.strip() for line in f if line.strip()]
        id_to_idx = {uid: i for i, uid in enumerate(all_ids)}
        indices = [id_to_idx[uid] for uid in df["id"] if uid in id_to_idx]
        embs = np.array(embeddings[indices], dtype=np.float32)
        return embs, None  # type: ignore
    else:
        sequences = df["sequence"].tolist()
        dim = embedder.embedding_dim
        index = faiss.IndexFlatIP(dim)
        batch_size = 10000
        for i in range(0, len(sequences), batch_size):
            batch = embedder.embed(sequences[i : i + batch_size])
            index.add(batch)  # type: ignore
        return None, index  # type: ignore


def build_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)  # type: ignore
    return index


def evaluate(embedder_name, embedder, df, corpus_embs, index, n_queries, top_k, rng):
    query_indices = rng.choice(len(df), size=n_queries, replace=False)

    aurocs, mrrs, hit_at_k_list = [], [], []
    n_skipped = 0

    n_skipped_all_positive = 0
    n_skipped_all_negative = 0
    n_skipped_mixed_invalid = 0

    n_pure_positive = 0
    n_pure_negative = 0

    for qi in query_indices:
        query_id = df["id"].iloc[qi]
        query_go = set(df["go_terms"].iloc[qi])

        if embedder_name == "esm2":
            q_emb = corpus_embs[qi : qi + 1]
        else:
            q_emb = embedder.embed([df["sequence"].iloc[qi]])

        D, I = index.search(q_emb, top_k + 1)
        D, I = D[0], I[0]

        mask = [df["id"].iloc[i] != query_id for i in I]
        D, I = D[mask][:top_k], I[mask][:top_k]

        y_true = [1 if set(df["go_terms"].iloc[i]) & query_go else 0 for i in I]
        y_score = D.tolist()

        hit_at_k = 1.0 if any(y_true) else 0.0
        hit_at_k_list.append(hit_at_k)

        rr = 0.0
        for rank, label in enumerate(y_true, start=1):
            if label == 1:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        try:
            auroc = roc_auc_score(y_true, y_score)
            if np.isnan(auroc):
                raise ValueError
            aurocs.append(auroc)

        except ValueError:
            n_skipped += 1

            if all(t == 1 for t in y_true):
                n_skipped_all_positive += 1
                n_pure_positive += 1

            elif all(t == 0 for t in y_true):
                n_skipped_all_negative += 1
                n_pure_negative += 1

            else:
                n_skipped_mixed_invalid += 1

            aurocs.append(None)

    valid_aurocs = [a for a in aurocs if a is not None]

    return {
        "mean_auroc": float(np.mean(valid_aurocs)) if len(valid_aurocs) > 0 else None,
        "mean_mrr": float(np.mean(mrrs)),
        "hit_at_k": float(np.mean(hit_at_k_list)),
        "n_queries_scored": len(valid_aurocs),
        "n_queries_skipped": n_skipped,
        "n_skipped_all_positive": n_skipped_all_positive,
        "n_skipped_all_negative": n_skipped_all_negative,
        "n_skipped_mixed_invalid": n_skipped_mixed_invalid,
        "pct_pure_positive": n_pure_positive / len(query_indices),
        "pct_pure_negative": n_pure_negative / len(query_indices),
    }


def main():

    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    rng = np.random.default_rng(args.seed)

    df = load_corpus(corpus_path=args.corpus)
    logger.info("Corpus loaded: %d proteins with GO terms", len(df))

    embedders = {
        "esm2": None,
        "physiochemical": PhysiochemicalEmbedder(),
        "random": RandomEmbedder(seed=args.seed),
    }

    results = {}
    for name in ["esm2", "physiochemical", "random"]:
        if args.embedder != "all" and args.embedder != name:
            continue
        logger.info("Evaluating embedder: %s", name)
        embedder = embedders[name]
        corpus_embs, index = get_corpus_embeddings(
            name, embedder, df, args.embeddings, args.ids
        )
        if index is None:
            index = build_index(corpus_embs)
        metrics = evaluate(
            name, embedder, df, corpus_embs, index, args.n_queries, args.top_k, rng
        )

        metrics["model_version"] = (
            embedder.model_version if embedder else "esm2_t33_650M_UR50D"
        )
        results[name] = metrics
        logger.info(
            "%s: AUROC=%.4f MRR=%.4f", name, metrics["mean_auroc"], metrics["mean_mrr"]
        )
    with open(args.corpus, "rb") as f:
        corpus_sha256 = hashlib.sha256(f.read()).hexdigest()

    output = {
        "task": "GO-term functional similarity proxy",
        "corpus_version": corpus_sha256,
        "n_queries": args.n_queries,
        "top_k": args.top_k,
        "seed": args.seed,
        "results": results,
    }

    out_path = Path(args.output_dir) / "eval_dti_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)

    print(f"\n{'Embedder':<20} {'AUROC':>8} {'MRR':>8} {'Scored':>8} {'Skipped':>8}")
    print("-" * 56)
    for name, m in results.items():
        print(
            f"{name:<20} {m['mean_auroc']:>8.4f} {m['mean_mrr']:>8.4f} "
            f"{m['n_queries_scored']:>8} {m['n_queries_skipped']:>8}"
        )


if __name__ == "__main__":
    main()
