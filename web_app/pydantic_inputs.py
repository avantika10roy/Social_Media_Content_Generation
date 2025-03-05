# Dependencies
from pydantic import Field
from typing import Optional
from pydantic import BaseModel

# Text Generation Input Class
class TextGenerationRequest(BaseModel):
    occasion: str                = Field(..., example="Product Launch")
    brief: str                   = Field(..., example="Announcing our AI-powered automation tool!")
    platform: str                = Field(..., example="LinkedIn")
    target_audience: str         = Field(..., example="Tech Entrepreneurs")
    tone: str                    = Field(..., example="Professional")
    extra_details: Optional[str] = Field(None, example="Include the benefits of AI-driven automation.")
    max_length: int              = Field(256, example=640)
    temperature: float           = Field(0.7, example=0.5)
    top_p: float                 = Field(0.9, example=0.85)


# Image Generation Input Class
class ImageGenerationRequest(BaseModel):
    occasion: str                = Field(..., example="Product Launch")
    brief: str                   = Field(..., example="Announcing our AI-powered automation tool!")
    platform: str                = Field(..., example="LinkedIn")
    target_audience: str         = Field(..., example="Tech Entrepreneurs")
    tone: str                    = Field(..., example="Professional")
    extra_details: Optional[str] = Field(None, example="Include the benefits of AI-driven automation.")
    max_length: int              = Field(256, example=640)
    temperature: float           = Field(0.7, example=0.5)
    top_p: float                 = Field(0.9, example=0.85)


