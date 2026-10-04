"""Pydantic request/response models — these drive the Swagger docs at /docs."""


from typing import Any

from pydantic import BaseModel, Field

from modules.feature_extraction import FEATURE_NAMES


class ModelPrediction(BaseModel):
    model: str = Field(..., description="Model name", examples=["MLR"])
    ra: float | None = Field(
        None, description="Predicted surface roughness in µm", examples=[2.31]
    )
    available: bool = Field(..., description="Whether this model could run")
    detail: str | None = Field(
        None, description="Why the model was unavailable, if it was"
    )


class PredictionResponse(BaseModel):
    filename: str | None = Field(None, description="Uploaded file name, if any")
    channels_found: list[str] = Field(
        ..., description="Expected channel columns present in the input"
    )
    predictions: list[ModelPrediction]
    spread_um: float | None = Field(
        None, description="Max minus min Ra across models that produced a value"
    )
    category: str | None = Field(
        None, description="Surface quality band from the fuzzy prediction"
    )
    features: dict[str, float] = Field(
        ..., description=f"The {len(FEATURE_NAMES)} extracted vibration features"
    )
    latency_ms: float


class FeaturesRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=(
            f"All {len(FEATURE_NAMES)} feature values keyed by name, "
            "e.g. 'Channel 1 RMS'"
        ),
    )


class ModelStatus(BaseModel):
    model: str
    ready: bool
    detail: str | None = None


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatus]
    all_ready: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


class ToolCallLog(BaseModel):
    tool: str = Field(..., description="Name of the tool the agent invoked")
    input: dict[str, Any] = Field(..., description="Arguments the agent passed")
    output: str = Field(..., description="The tool's return value")
    reasoning: str | None = Field(
        None, description="The agent's own text preceding this call, if any"
    )


class AgentPredictResponse(BaseModel):
    filename: str | None = Field(None, description="Uploaded file name, if any")
    ra: float | None = Field(
        None, description="The agent's final trusted Ra (µm), parsed from its report"
    )
    models_used: list[str] = Field(
        ..., description="Prediction tool names the agent called, in call order"
    )
    tool_calls: list[ToolCallLog] = Field(
        ..., description="Full reasoning trace: every tool call the agent made"
    )
    noise_level: str | None = Field(
        None, description="Raw get_signal_noise_level output, if the agent called it"
    )
    report: str = Field(..., description="The agent's final free-text QC report")
    rag_consulted: bool = Field(..., description="Whether the agent called query_rag")


class RagAskRequest(BaseModel):
    question: str = Field(..., description="A question about the project methodology")


class RagAskResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
