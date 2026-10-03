from typing import List, Optional
from pydantic import BaseModel, Field

class LLMAnswerSchema(BaseModel):
    found: bool = Field(description="Чи знайдено відповідь у наданих фрагментах")
    answer: str = Field(description="Текст відповіді клієнту зі сносками [1], [2]")
    sources: List[int] = Field(default_factory=list, description="Номери використаних джерел")

class DocumentSourceItem(BaseModel):
    id: int
    title: str
    section: str
    date: Optional[str] = None
    file_name: str
    score: float

class RAGResponse(BaseModel):
    answer: str
    found: bool
    sources: List[DocumentSourceItem]
    retrieved_chunks: List[dict]
    search_time_ms: float
    generation_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    model_name: str
    warning: Optional[str] = None