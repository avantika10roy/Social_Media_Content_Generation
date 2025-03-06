# Dependencies
from pydantic import Field
from typing import Optional
from pydantic import BaseModel


class TextGenerationResponse(BaseModel):
    generated_text: str

