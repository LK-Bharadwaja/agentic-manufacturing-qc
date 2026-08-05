"""Typed errors so callers (notably the API) can map failures to status codes."""


class SurfaceRoughnessError(Exception):
    """Base class for all project-specific errors."""


class InvalidInputError(SurfaceRoughnessError):
    """Uploaded data is unusable — unreadable file, wrong type, or too large."""


class MissingChannelError(InvalidInputError):
    """None of the expected vibration channel columns were present."""

    def __init__(self, expected: list[str], found: list[str]):
        self.expected = expected
        self.found = found
        super().__init__(
            f"No vibration channel columns found. Expected any of {expected}; "
            f"file had columns: {found[:10]}"
        )


class FeatureExtractionError(SurfaceRoughnessError):
    """Feature extraction failed on otherwise well-formed input."""


class ModelNotAvailableError(SurfaceRoughnessError):
    """A model's artifacts or its runtime dependency are missing."""

    def __init__(self, model: str, reason: str):
        self.model = model
        self.reason = reason
        super().__init__(f"{model} unavailable: {reason}")
