from pydantic import BaseModel

class GenerateRequest(BaseModel):
    theme: str
    n: int = 1

class GenerateResponse(BaseModel):
    quotes: list[str]