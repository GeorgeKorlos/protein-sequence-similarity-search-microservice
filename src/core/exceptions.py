class ServiceException(Exception):
    def __init__(self, error_code: str, message: str, http_status: int) -> None:
        self.error_code = error_code
        self.message = message
        self.http_status = http_status
        super().__init__(message)


class InvalidSequenceException(ServiceException):
    def __init__(
        self,
        message: str = "Invalid sequence.",
    ) -> None:
        super().__init__(
            error_code="INVALID_SEQUENCE",
            message=message,
            http_status=422,
        )


class SequenceTooLongException(ServiceException):
    def __init__(
        self,
        message: str = "Sequence exceeds maximum length.",
    ) -> None:
        super().__init__(
            error_code="SEQUENCE_TOO_LONG",
            message=message,
            http_status=422,
        )


class BatchTooLargeException(ServiceException):
    def __init__(
        self,
        message: str = "Batch size exceeds limit.",
    ) -> None:
        super().__init__(error_code="BATCH_TOO_LARGE", message=message, http_status=422)


class PayloadTooLargeException(ServiceException):
    def __init__(
        self,
        message: str = "Payload too large.",
    ) -> None:
        super().__init__(
            error_code="PAYLOAD_TOO_LARGE", message=message, http_status=413
        )


class ModelNotLoadedException(ServiceException):
    def __init__(self, message: str = "Model is not loaded") -> None:
        super().__init__(
            error_code="MODEL_NOT_LOADED", message=message, http_status=503
        )


class IndexNotReadyException(ServiceException):
    def __init__(
        self,
        message: str = "Index is not ready.",
    ) -> None:
        super().__init__(
            error_code="INDEX_NOT_READY",
            message=message,
            http_status=503,
        )


class EmbeddingFailedException(ServiceException):
    def __init__(
        self,
        message: str = "Embedding generation failed.",
    ) -> None:
        super().__init__(
            error_code="EMBEDDING_FAILED",
            message=message,
            http_status=500,
        )


class SearchFailedException(ServiceException):
    def __init__(
        self,
        message: str = "Search failed.",
    ) -> None:
        super().__init__(
            error_code="SEARCH_FAILED",
            message=message,
            http_status=500,
        )


class InvalidRequestException(ServiceException):
    def __init__(
        self,
        message: str = "Invalid request.",
    ) -> None:
        super().__init__(
            error_code="INVALID_REQUEST",
            message=message,
            http_status=400,
        )


class ValidationError(Exception):

    def __init__(self, error_code: str, message: str | None = None):
        self.error_code = error_code
        self.message = message or error_code
        super().__init__(self.message)


class SequenceValidationError(ValidationError):
    """Raised when sequence validation fails."""


INVALID_SEQUENCE = "INVALID_SEQUENCE"
SEQUENCE_TOO_LONG = "SEQUENCE_TOO_LONG"
BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
