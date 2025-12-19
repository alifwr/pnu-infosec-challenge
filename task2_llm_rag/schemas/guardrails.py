from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    is_input_safe: bool = Field(
        description="Whether the request is rejected due to safety concerns (e.g. PII requests)."
    )
