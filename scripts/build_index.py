import time
import logging
import argparse
import numpy as np
from pathlib import Path
from src.core.embedder import ESM2Embedder
from src.core.corpus_store import CorpusStore


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    log_file = log_dir / "build_index.log"
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed protein sequences from a corpus using a transformer"
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
        default=Path("data/"),
        help="Directory to save embeddings (default: data/)",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace model tag (default: facebook/esm2_t33_650M_UR50D)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help="Hard cap on sequences per batch (default: 256)",
    )
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=24.0,
        help="Total GPU VRAM in GB (default: 24.0)",
    )
    parser.add_argument(
        "--vram-safety",
        type=float,
        default=0.85,
        help="Fraction of usable VRAM after model weights (default: 0.85)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/"),
        help="Directory for log files (default: logs/)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit rows loaded from corpus (for debugging)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output files"
    )

    return parser.parse_args()


_BYTES_PER_ELEM = 2  # fp16
LONG_SEQ_THRESHOLD = 800


def _batch_memory_bytes(
    n: int, max_len: int, num_heads: int, num_layers: int, hidden_dim: int
) -> int:
    attn = n * num_heads * (max_len**2) * _BYTES_PER_ELEM
    hidden = n * max_len * num_layers * hidden_dim * _BYTES_PER_ELEM
    return attn + hidden


def _max_batch_size_for_len(
    max_len: int,
    hard_cap: int,
    budget: int,
    num_heads: int,
    num_layers: int,
    hidden_dim: int,
) -> int:
    lo, hi = 1, hard_cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if (
            _batch_memory_bytes(mid, max_len, num_heads, num_layers, hidden_dim)
            <= budget
        ):
            lo = mid
        else:
            hi = mid - 1
    return lo


def build_dynamic_batches(
    sequences: list[str],
    ids: list[str],
    max_batch_size: int,
    vram_budget: int,
    num_heads: int,
    num_layers: int,
    hidden_dim: int,
) -> list[tuple]:
    batches = []
    current_seqs, current_ids, current_max_len = [], [], 0

    for seq, sid in zip(sequences, ids):
        seq_len = len(seq)

        if seq_len > LONG_SEQ_THRESHOLD:
            if current_seqs:
                batches.append((current_seqs, current_ids))
                current_seqs, current_ids, current_max_len = [], [], 0
            batches.append(([seq], [sid]))
            continue

        new_max_len = max(current_max_len, seq_len)
        cap = _max_batch_size_for_len(
            new_max_len, max_batch_size, vram_budget, num_heads, num_layers, hidden_dim
        )

        flush = (
            len(current_seqs) >= cap
            or _batch_memory_bytes(
                len(current_seqs) + 1, new_max_len, num_heads, num_layers, hidden_dim
            )
            > vram_budget
        )

        if flush and current_seqs:
            batches.append((current_seqs, current_ids))
            current_seqs, current_ids, current_max_len = [], [], 0
            new_max_len = seq_len

        current_seqs.append(seq)
        current_ids.append(sid)
        current_max_len = new_max_len

    if current_seqs:
        batches.append((current_seqs, current_ids))

    return batches


def main():
    args = parse_args()
    logger = setup_logging(args.log_dir)

    store = CorpusStore(args.corpus, args.nrows)
    logger.info("Corpus loaded from: %s", args.corpus)
    logger.info("Corpus size: %d", len(store))
    logger.info("Corpus version: %s", store.corpus_version)

    model = ESM2Embedder(model_tag=args.model_tag, device=args.device, debug=False)
    logger.info("Loaded model: %s", model.MODEL_TAG)
    logger.info("Model version: %s", model.model_version)

    cfg = model.model.config
    num_heads = cfg.num_attention_heads
    num_layers = cfg.num_hidden_layers
    hidden_dim = cfg.hidden_size
    embedding_dim = model.embedding_dim
    model_weight_bytes = sum(p.numel() * 2 for p in model.model.parameters())

    logger.info(
        "Model config: heads=%d layers=%d dim=%d weights=%.0f MB",
        num_heads,
        num_layers,
        hidden_dim,
        model_weight_bytes / 1024**2,
    )

    all_sequences = store.get_all_sequences()
    all_ids = store.get_all_ids()

    pairs = sorted(zip(all_sequences, all_ids), key=lambda x: len(x[0]))
    all_sequences, all_ids = zip(*pairs)
    all_sequences, all_ids = list(all_sequences), list(all_ids)

    vram_budget = int((args.vram_gb * 1024**3 - model_weight_bytes) * args.vram_safety)
    logger.info("VRAM budget: %.2f MB", vram_budget / 1024**2)

    batches = build_dynamic_batches(
        all_sequences,
        all_ids,
        args.max_batch_size,
        vram_budget,
        num_heads,
        num_layers,
        hidden_dim,
    )
    logger.info(
        "Dynamic batching: %d sequences → %d batches (avg %.1f seq/batch)",
        len(store),
        len(batches),
        len(store) / len(batches),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "swissprot_embeddings.npy"
    ids_path = args.output_dir / "swissprot_ids.txt"
    failed_path = args.log_dir / "failed_sequences.txt"

    sequences_done = 0
    batches_to_skip = 0

    if args.resume and ids_path.exists():
        with open(ids_path) as f:
            sequences_done = sum(1 for line in f if line.strip())

        counted = 0
        for i, (seqs, _) in enumerate(batches):
            if counted + len(seqs) <= sequences_done:
                counted += len(seqs)
                batches_to_skip = i + 1
            else:
                break

        logger.info(
            "Resuming from sequence %d (skipping %d batches)",
            sequences_done,
            batches_to_skip,
        )
        batches = batches[batches_to_skip:]
        memmap_arr = np.lib.format.open_memmap(
            filename=output_path,
            mode="r+",
            dtype="float32",
            shape=(len(store), embedding_dim),
        )
    else:
        memmap_arr = np.lib.format.open_memmap(
            filename=output_path,
            mode="w+",
            dtype="float32",
            shape=(len(store), embedding_dim),
        )

    total = len(store)
    start_time = time.time()
    log_every = max(1, len(batches) // 200)
    ids_file_mode = "a" if args.resume else "w"
    sequences_this_run = 0

    with open(ids_path, ids_file_mode) as f_ids, open(failed_path, "w") as f_failed:
        for batch_idx, (batch_seqs, batch_ids) in enumerate(batches):
            batch_start_time = time.time()

            try:
                tokens = model.tokenize(batch_seqs)
                input_ids = tokens["input_ids"].to(model.device, non_blocking=True)
                attention_mask = tokens["attention_mask"].to(
                    model.device, non_blocking=True
                )

                embeddings = model.embed_tokenized(input_ids, attention_mask)
                memmap_arr[sequences_done : sequences_done + len(batch_seqs)] = (
                    embeddings
                )

                if batch_idx % 50 == 0:
                    memmap_arr.flush()

                f_ids.write("\n".join(batch_ids) + "\n")
                sequences_this_run += len(batch_seqs)
                sequences_done += len(batch_seqs)

                if batch_idx % log_every == 0:
                    batch_latency_ms = (time.time() - batch_start_time) * 1000
                    elapsed = time.time() - start_time
                    eta = (
                        (elapsed / sequences_this_run) * (total - sequences_done)
                        if sequences_this_run
                        else 0
                    )
                    batch_max_len = max(len(s) for s in batch_seqs)
                    logger.info(
                        "Processed %d/%d | batch=%d seqs | max_len=%d | latency=%.2f ms | elapsed=%.2fs | eta=%.2fs",
                        sequences_done,
                        total,
                        len(batch_seqs),
                        batch_max_len,
                        batch_latency_ms,
                        elapsed,
                        eta,
                    )

            except Exception as e:
                logger.error(
                    "Batch failed at seq %d. IDs: %s. Error: %s",
                    sequences_done,
                    batch_ids,
                    str(e),
                )
                f_failed.write("\n".join(batch_ids) + "\n")
                sequences_this_run += len(batch_seqs)
                sequences_done += len(batch_seqs)

    total_time = time.time() - start_time
    sequences_per_sec = total / total_time if total_time > 0 else 0.0
    logger.info(
        "Completed embedding %d sequences in %.2fs (%.2f seq/s)",
        total,
        total_time,
        sequences_per_sec,
    )


if __name__ == "__main__":
    main()
