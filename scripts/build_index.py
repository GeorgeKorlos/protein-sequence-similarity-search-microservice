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

    log_file = log_dir / 'build_index.log'
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Embed protein sequences from a corpus using a transformers")

    parser.add_argument("--corpus", type=Path, default=Path("data/swissprot_clean.csv"), help="Path to the input CSV file containing sequences (default: data/swissprot_clean.csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/"), help="Directory to save embeddings (default: data/)")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of sequences processed per batch (default: 8)")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default="cuda", help="Compute device to use: 'cpu' or 'cuda' (default: cuda)")
    parser.add_argument("--log-dir",type=Path, default=Path("logs/"), help="Directory for log files (default: logs/)")
    parser.add_argument("--nrows", type=int, default=None, help="Limit number of rows loaded from corpus (for debugging / sampling)")

    return parser.parse_args()


def main():
    
    args = parse_args()

    logger = setup_logging(args.log_dir)

    store = CorpusStore(args.corpus, args.nrows)

    logger.info(f"Corpus loaded from: {args.corpus}")
    logger.info(f"Corpus size: {len(store)}")
    logger.info(f"Corpus version: {store.corpus_version}")

    model = ESM2Embedder(device=args.device, debug=False)

    logger.info(f"Loaded model: {model.MODEL_TAG}")
    logger.info(f"Model version: {model.model_version}")

    all_sequences = store.get_all_sequences()
    all_ids = store.get_all_ids()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "swissprot_embeddings.npy"

    memmap_arr = np.lib.format.open_memmap(filename=output_path, mode='w+', dtype='float32', shape=(len(store), 640))

    ids_path = args.output_dir / "swissprot_ids.txt"
    failed_path = args.log_dir / "failed_sequences.txt"

    with open(ids_path, "w") as f_ids, open(failed_path, "w") as f_failed:

        total = len(store)
        start_time = time.time()

        for i in range(0, total, args.batch_size):

            batch_start_time = time.time()
            batch_sequences =  all_sequences[i:i+args.batch_size]
            batch_ids = all_ids[i:i+args.batch_size]
            
            try:
                embeddings = model.embed(batch_sequences)
                memmap_arr[i:i+len(batch_sequences)] = embeddings

                for seq_id in batch_ids:
                    f_ids.write(f"{seq_id}\n")
                
                batch_latency_ms = (time.time() - batch_start_time) * 1000
                sequences_done = i + len(batch_sequences)
                elapsed = time.time() - start_time
                estimated_remaining = (elapsed / sequences_done) * (total - sequences_done)

                logger.info("Processed %d/%d | batch latency=%.2f ms | elapsed=%.2fs | eta=%.2fs", sequences_done, total, batch_latency_ms, elapsed, estimated_remaining)
                
            except Exception as e:
                logger.error("Batch failed at index %d. IDs: %s. Error: %s", i, batch_ids, str(e))
                for seq_id in batch_ids:
                    f_failed.write(f"{seq_id}\n")
        
        total_time = time.time() - start_time
        sequences_per_sec = total / total_time if total_time > 0 else 0.0

        logger.info("Completed embedding %d sequences in %.2fs (%.2f seq/s)", total, total_time, sequences_per_sec)
                
if __name__ == '__main__':
    main()
