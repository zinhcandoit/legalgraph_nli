import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from be.src.config.config import config
from ..logger import logger
from ..models.embedding import EmbeddingModel
from ..models.llm import GeminiModel, get_llm
from ..models.reranker import RerankerModel
from ..processing.query_rewrite import rewrite_query
from ..retrieval.graph_retriever import GraphRetriever
from ..schemas import NLIVerificationResult, RetrievedChunk, compute_sha256

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

NLI_SERVICE_URL = os.getenv("NLI_SERVICE_URL", "http://localhost:8001/predict")


DIRECT_PROMPT_TEMPLATE = PromptTemplate.from_template("Trả lời ngắn gọn câu hỏi sau: {query}")

RAG_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Bạn là chuyên gia trợ lý pháp lý thông minh.
Hãy dựa vào các căn cứ và điều khoản pháp luật được trích dẫn dưới đây để trả lời câu hỏi chính xác và trích dẫn điều khoản cụ thể.
Nếu các trích đoạn dưới đây không chứa thông tin trả lời, hãy tự tin trả lời là "Không có chứng cứ xác minh".

CĂN CỨ VÀ NGỮ CẢNH PHÁP LÝ:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

CÂU TRẢ LỜI:"""
)


class RAGPipeline:
    def __init__(self):
        logger.info("🚀 Đang khởi tạo LangChain RAG & NLI Pipeline...")
        self.embedding = EmbeddingModel()
        self.retriever = GraphRetriever(embedding_model=self.embedding)
        self.reranker = RerankerModel()
        self.llm_model = get_llm()
        self.llm = self.llm_model.get()
        self.output_parser = StrOutputParser()

        # Chuỗi LangChain LCEL chuẩn
        self.direct_chain = DIRECT_PROMPT_TEMPLATE | self.llm | self.output_parser
        self.rag_chain = RAG_PROMPT_TEMPLATE | self.llm | self.output_parser

        logger.info("✅ LangChain RAG Pipeline đã sẵn sàng phục vụ!")

    def _format_docs_for_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "Không tìm thấy trích đoạn pháp lý liên quan trong cơ sở dữ liệu."
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            blocks.append(f"[Trích đoạn {i} ({chunk.source_type})]:\n{chunk.text}")
        return "\n\n".join(blocks)

    def _verify_nli(self, query: str, answer: str) -> Optional[NLIVerificationResult]:
        if not answer or not answer.strip() or not query or not query.strip():
            return None

        clean_query = query.strip()
        clean_ans = answer.strip()

        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.post(
                    NLI_SERVICE_URL,
                    json={
                        "specific_question": clean_query,
                        "legal_document": clean_ans,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    pred = data.get("prediction", {})
                    label_id = pred.get("label_id", 0)
                    label = pred.get("label", "UNKNOWN")
                    confidence = float(pred.get("confidence", 0.0))
                    probs = pred.get("probabilities", {})

                    is_valid = (label_id == 1 or "ENTAILMENT" in label.upper() or "WIN" in label.upper())
                    note = (
                        "Đánh giá NLI: Hợp lệ (Entailment/Win)."
                        if is_valid
                        else "Cảnh báo NLI: Mâu thuẫn/Không thỏa mãn (Contradiction/Lose)."
                    )
                    return NLIVerificationResult(
                        is_valid=is_valid,
                        label=label,
                        confidence=confidence,
                        probabilities=probs,
                        note=note,
                    )
        except Exception as e:
            logger.warning(f"Lỗi kết nối NLI Service: {e}")

        return None

    def run(
        self,
        query: str,
        rag: bool = True,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        input_sha256 = compute_sha256(f"{query.strip()}_rag={rag}")
        query_clean = query.strip()
        rewritten_query = None
        retrieved_chunks: List[RetrievedChunk] = []

        # 1. w/o RAG: Direct LLM -> NLI Verification (Hypothesis: Answer, Premise: Query)
        if not rag:
            try:
                answer = str(self.direct_chain.invoke({"query": query_clean})).strip()
            except Exception as e:
                logger.error(f"Lỗi Direct LLM: {e}")
                answer = f"Lỗi tạo câu trả lời: {str(e)}"

            # NLI kiểm chứng: 2 input (Hypothesis: answer, Premise: query)
            nli_verification = self._verify_nli(query=query_clean, answer=answer)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "request_id": request_id,
                "timestamp": timestamp,
                "query": query_clean,
                "rewritten_query": None,
                "rag_used": False,
                "retrieval_mode": None,
                "answer": answer,
                "nli_verification": nli_verification,
                "retrieved_chunks": [],
                "total_chunks": 0,
                "latency_ms": latency_ms,
                "input_sha256": input_sha256,
                "session_id": session_id,
                "metadata": metadata,
            }

        # 2. w/ RAG: Query -> Rewriter (tùy config) -> GraphDB -> Chunks -> Reranker -> LCEL Chain
        if config.retrieval.rewrite_query:
            rewritten_query = rewrite_query(query_clean, llm=self.llm_model)
        else:
            rewritten_query = query_clean

        search_k = config.retrieval.search_k
        effective_top_k = top_k if top_k is not None else config.retrieval.rerank_top_k

        candidate_docs = self.retriever.retrieve(query=rewritten_query, top_k=search_k)
        ranked_results = self.reranker.rerank(
            query=rewritten_query,
            documents=candidate_docs,
            top_k=effective_top_k,
        )

        for doc, score in ranked_results:
            retrieved_chunks.append(
                RetrievedChunk(
                    id=doc.get("id"),
                    text=doc.get("text", ""),
                    score=round(float(score), 4),
                    source_type=doc.get("source_type", "text_unit"),
                    metadata=doc.get("metadata"),
                )
            )

        context_str = self._format_docs_for_context(retrieved_chunks)
        try:
            answer = str(self.rag_chain.invoke({"context": context_str, "query": rewritten_query})).strip()
        except Exception as e:
            logger.error(f"Lỗi RAG Generation: {e}")
            answer = f"Lỗi tạo câu trả lời tổng hợp: {str(e)}"

        nli_verification = self._verify_nli(query=query_clean, answer=answer)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "request_id": request_id,
            "timestamp": timestamp,
            "query": query_clean,
            "rewritten_query": rewritten_query,
            "rag_used": True,
            "retrieval_mode": self.retriever.mode,
            "answer": answer,
            "nli_verification": nli_verification,
            "retrieved_chunks": retrieved_chunks,
            "total_chunks": len(retrieved_chunks),
            "latency_ms": latency_ms,
            "input_sha256": input_sha256,
            "session_id": session_id,
            "metadata": metadata,
        }


_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
