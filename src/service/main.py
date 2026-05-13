from fastapi import FastAPI
from src.core.searcher import Searcher
from src.service.config import settings
from contextlib import asynccontextmanager
from src.core.embedder import ESM2Embedder
from src.core.corpus_store import CorpusStore
from src.core.index_manager import IndexManager
from src.core.validator import SequenceValidator
from src.service.middleware import TimingMiddleware, RequestIDMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.corpus = CorpusStore(csv_path=settings.corpus_path)
    app.state.embedder = ESM2Embedder(
        model_tag=settings.model_tag, device=settings.device
    )
    app.state.index_manager = IndexManager(index_type="", dim=0, params=None)
    app.state.index_manager.load(settings.index_path)
    app.state.validator = SequenceValidator(
        max_batch_size=settings.max_batch_size,
        max_payload_size=settings.max_payload_size,
    )
    app.state.searcher = Searcher(
        corpus_store=app.state.corpus,
        embedder=app.state.embedder,
        index_manager=app.state.index_manager,
    )

    app.state.model_loaded = True
    app.state.index_loaded = True

    yield


app = FastAPI(lifespan=lifespan)
app.middleware(TimingMiddleware)  # type: ignore
app.middleware(RequestIDMiddleware)  # type: ignore
