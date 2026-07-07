import os
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

    FLUSH_EVERY = 50

    done_ids: set[str] = set()
    write_cursor = 0

    if args.resume and ids_path.exists():
        with open(ids_path) as f:
            done_ids = {line.strip() for line in f if line.strip()}
        write_cursor = len(done_ids)
        logger.info("Resuming: %d sequences already embedded", write_cursor)
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
    embedded_this_run = 0
    id_buffer: list[str] = []
    ids_mode = "a" if args.resume else "w"
    failed_mode = "a" if args.resume else "w"

    def _commit(f_ids):
        memmap_arr.flush()
        if id_buffer:
            f_ids.write("".join(id_buffer))
            f_ids.flush()
            os.fsync(f_ids.fileno())
            id_buffer.clear()

    with open(ids_path, ids_mode) as f_ids, open(failed_path, failed_mode) as f_failed:
        for batch_idx, (batch_seqs, batch_ids) in enumerate(batches):
            if done_ids and all(bid in done_ids for bid in batch_ids):
                continue

            batch_start_time = time.time()
            n = len(batch_seqs)

            try:
                tokens = model.tokenize(batch_seqs)
                input_ids = tokens["input_ids"].to(model.device, non_blocking=True)
                attention_mask = tokens["attention_mask"].to(
                    model.device, non_blocking=True
                )
                embeddings = model.embed_tokenized(input_ids, attention_mask)

                memmap_arr[write_cursor : write_cursor + n] = embeddings
                id_buffer.append("\n".join(batch_ids) + "\n")
                write_cursor += n
                embedded_this_run += n

                if batch_idx % FLUSH_EVERY == 0:
                    _commit(f_ids)

                if batch_idx % log_every == 0:
                    batch_latency_ms = (time.time() - batch_start_time) * 1000
                    elapsed = time.time() - start_time
                    eta = (
                        (elapsed / embedded_this_run) * (total - write_cursor)
                        if embedded_this_run
                        else 0
                    )
                    logger.info(
                        "Written %d/%d | batch=%d seqs | max_len=%d | latency=%.2f ms | elapsed=%.2fs | eta=%.2fs",
                        write_cursor,
                        total,
                        n,
                        max(len(s) for s in batch_seqs),
                        batch_latency_ms,
                        elapsed,
                        eta,
                    )

            except Exception as e:
                logger.error(
                    "Batch failed at row %d. IDs: %s. Error: %s",
                    write_cursor,
                    batch_ids,
                    str(e),
                )
                f_failed.write("\n".join(batch_ids) + "\n")
                f_failed.flush()

        _commit(f_ids)

    if write_cursor < total:
        logger.warning(
            "Only %d/%d embedded; truncating output to valid rows", write_cursor, total
        )
        del memmap_arr
        valid = np.array(np.load(output_path, mmap_mode="r")[:write_cursor])
        np.save(output_path, valid)


if __name__ == "__main__":
    main()
