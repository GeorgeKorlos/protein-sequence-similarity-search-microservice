from fastapi import Request
from src.core.searcher import Searcher
from src.core.embedder import ESM2Embedder
from src.core.corpus_store import CorpusStore
from src.core.validator import SequenceValidator


def get_embedder(request: Request) -> ESM2Embedder:
    return request.app.state.embedder


def get_searcher(request: Request) -> Searcher:
    return request.app.state.searcher


def get_corpus(request: Request) -> CorpusStore:
    return request.app.state.corpus


def get_validator(request: Request) -> SequenceValidator:
    return request.app.state.validator


def get_request_id(request: Request) -> str:
    return str(request.state.request_id)
