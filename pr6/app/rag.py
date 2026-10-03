import time
from typing import Dict, Any, List
from app.retrieval import filter_and_format_chunks
from app.llm import generate_answer
from app.schema import RAGResponse, DocumentSourceItem
from app.index import semantic_search

class RAGPipeline:
    def __init__(
        self,
        search_top_k: int = 8,
        model_top_k: int = 4,
        similarity_threshold: float = 0.35,
        model_name: str = "gpt-4o-mini",
        token_budget: int = 1500
    ):
        self.search_top_k = search_top_k
        self.model_top_k = model_top_k
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.token_budget = token_budget

    def run(self, question: str) -> RAGResponse:
        start_search = time.perf_counter()
        raw_results = semantic_search(question, top_k=self.search_top_k)
        search_time_ms = (time.perf_counter() - start_search) * 1000.0

        chunks, context_text = filter_and_format_chunks(
            raw_results=raw_results,
            top_k_for_model=self.model_top_k,
            similarity_threshold=self.similarity_threshold,
            max_token_budget=self.token_budget
        )

        if not chunks:
            return RAGResponse(
                answer="На жаль, у базі знань магазину немає інформації за вашим запитом.",
                found=False,
                sources=[],
                retrieved_chunks=[],
                search_time_ms=search_time_ms,
                generation_time_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                model_name=self.model_name,
                warning="Пошук не знайшов відповідних документів вище заданого порогу схожості."
            )

        llm_res, usage, gen_time_ms = generate_answer(
            question=question,
            context=context_text,
            model_name=self.model_name
        )

        valid_indices = {c["context_id"] for c in chunks}
        warning_msg = None

        if llm_res.found and not llm_res.sources:
            llm_res.found = False
            warning_msg = "Модель повідомила про знахідку, але не вказала жодного джерела."

        invalid_sources = [s for s in llm_res.sources if s not in valid_indices]
        if invalid_sources:
            llm_res.sources = [s for s in llm_res.sources if s in valid_indices]
            warning_msg = f"Виявлено неіснуючі посилання: {invalid_sources}."
            if not llm_res.sources:
                llm_res.found = False

        cited_sources: List[DocumentSourceItem] = []
        if llm_res.found:
            for s_id in sorted(list(set(llm_res.sources))):
                chunk = next(c for c in chunks if c["context_id"] == s_id)
                meta = chunk.get("metadata", {})
                cited_sources.append(
                    DocumentSourceItem(
                        id=s_id,
                        title=meta.get("title", "Документ"),
                        section=meta.get("section", "Розділ"),
                        date=meta.get("date"),
                        file_name=meta.get("file_name", ""),
                        score=chunk.get("score", 0.0)
                    )
                )

        return RAGResponse(
            answer=llm_res.answer,
            found=llm_res.found,
            sources=cited_sources,
            retrieved_chunks=chunks,
            search_time_ms=search_time_ms,
            generation_time_ms=gen_time_ms,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            model_name=self.model_name,
            warning=warning_msg
        )