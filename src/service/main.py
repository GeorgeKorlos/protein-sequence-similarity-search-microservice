import logging
from fastapi import FastAPI, Request
from src.service.routes import router
from src.core.searcher import Searcher
from src.service.config import settings
from contextlib import asynccontextmanager
from src.core.embedder import ESM2Embedder
from fastapi.responses import JSONResponse
from src.service.schemas import ErrorResponse
from src.core.corpus_store import CorpusStore
from src.core.index_manager import IndexManager
from src.core.validator import SequenceValidator
from src.core.exceptions import ServiceException
from src.service.middleware import TimingMiddleware, RequestIDMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings

    app.state.corpus = CorpusStore(csv_path=settings.corpus_path)

    app.state.embedder = ESM2Embedder(
        model_tag=settings.model_tag, device=settings.device
    )
    try:
        app.state.embedder.embed(["ACDE"])
        app.state.model_loaded = True
    except Exception as e:
        app.state.model_loaded = False
        logger.error("Model failed to load: %s", e)

    app.state.index_manager = IndexManager(index_type="", dim=0, params=None)
    app.state.index_manager.load(settings.index_path)
    if app.state.index_manager.index.ntotal > 0:  # type: ignore
        app.state.index_loaded = True
    else:
        app.state.index_loaded = False
        logger.error("Index not loaded")

    app.state.validator = SequenceValidator(
        max_batch_size=settings.max_batch_size,
        max_payload_size=settings.max_payload_size,
    )
    app.state.searcher = Searcher(
        corpus_store=app.state.corpus,
        embedder=app.state.embedder,
        index_manager=app.state.index_manager,
    )

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(TimingMiddleware)  # type: ignore
app.add_middleware(RequestIDMiddleware)  # type: ignore


@app.exception_handler(ServiceException)
async def service_exception_handler(
    request: Request, exc: ServiceException
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    error_response = ErrorResponse(
        error_code=exc.error_code, message=exc.message, request_id=request_id
    )

    return JSONResponse(
        status_code=exc.http_status, content=error_response.model_dump()
    )


app.include_router(router=router)
