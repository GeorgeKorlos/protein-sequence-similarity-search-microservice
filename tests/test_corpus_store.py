from src.core.corpus_store import CorpusStore
from pathlib import Path
import re

DATA_PATH = Path(__file__).parent.parent / "data" / "swissprot_clean.csv"

store = CorpusStore(csv_path=DATA_PATH, nrows=10)


def test_length():
    assert len(store) == 10


def test_get_sequence():
    seq = store.get_sequence(0)
    assert isinstance(seq, str)
    assert seq != ""


def test_get_metadata():
    meta = store.get_metadata(0)
    assert isinstance(meta, dict)
    assert {"id", "organism", "keywords", "go_terms"} <= meta.keys()


def test_corpus_version():
    version = store.corpus_version
    assert re.fullmatch(r"[0-9a-f]{64}", version)


def test_get_all_sequences():
    seqs = store.get_all_sequences()
    assert len(seqs) == 10
    assert all(isinstance(s, str) and s != "" for s in seqs)


def test_get_all_ids():
    ids = store.get_all_ids()
    assert len(ids) == 10
    assert all(isinstance(i, str) and i != "" for i in ids)
