"""API request ও response schemas — Pydantic দিয়ে type validation।"""
from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    """Machine sensor readings।"""
    air_temperature: float = Field(
        ..., ge=290.0, le=310.0,
        description="Air temperature in Kelvin (290–310 K)",
    )
    process_temperature: float = Field(
        ..., ge=300.0, le=320.0,
        description="Process temperature in Kelvin",
    )
    rotational_speed: float = Field(
        ..., ge=1000.0, le=3000.0,
        description="Rotational speed in RPM",
    )
    torque: float = Field(
        ..., ge=0.0, le=100.0,
        description="Torque in Nm",
    )
    tool_wear: float = Field(
        ..., ge=0.0, le=300.0,
        description="Tool wear in minutes",
    )
    type: Literal["L", "M", "H"] = Field(
        ..., description="Machine quality variant: L (Low), M (Medium), H (High)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0,
                "type": "M"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result।"""
    prediction: int = Field(..., description="0 = No Failure, 1 = Failure")
    probability: float = Field(..., description="Failure probability (0.0–1.0)")
    failure_type: str = Field(..., description="Type of predicted failure")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Risk assessment level"
    )
    cached: bool = Field(..., description="Was this result served from cache?")


class HealthResponse(BaseModel):
    """API health status।"""
    status: str
    redis: bool