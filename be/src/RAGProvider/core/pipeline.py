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
from ..processing.hyde import generate_hypothetical_law
from ..processing.query_rewrite import rewrite_query
from ..retrieval.graph_retriever import GraphRetriever
from ..schemas import NLIVerificationResult, RetrievedChunk, compute_sha256

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

NLI_SERVICE_URL = os.getenv("NLI_SERVICE_URL", "http://localhost:8001/predict")
NLI_BATCH_URL = os.getenv("NLI_BATCH_URL", "http://localhost:8001/predict_batch")

RAG_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Bạn là trợ lý luật thông minh.
Dựa vào các căn cứ và điều khoản pháp luật được trích dẫn để trả lời câu hỏi sau cùng trích dẫn luật cụ thể.
Lưu ý: Nếu trích dẫn luật bị bỏ trống hoặc không hỗ trợ việc trả lời câu hỏi. Hãy tự tin và CHỈ được trả về "Không có chứng cứ xác minh".
TRÍCH DẪN LUẬT:
{context}

CÂU HỎI:
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

        # Chuỗi LangChain LCEL: max_tokens x 2 khi không sử dụng RAG (Direct LLM)
        base_max_tokens = getattr(self.llm_model, "max_output_tokens", None) or getattr(config.models, "max_new_tokens", 2048)
        direct_max_tokens = base_max_tokens * 2
        self.direct_llm_model = GeminiModel(
            model=self.llm_model.model_name,
            temperature=self.llm_model.temperature,
            max_output_tokens=direct_max_tokens,
            api_key=self.llm_model.api_key,
        )
        self.direct_llm = self.direct_llm_model.get()

        # Thống nhất 1 RAG_PROMPT_TEMPLATE cho cả Direct LLM lẫn RAG
        self.direct_chain = RAG_PROMPT_TEMPLATE | self.direct_llm | self.output_parser
        self.rag_chain = RAG_PROMPT_TEMPLATE | self.llm | self.output_parser

        logger.info(f"✅ LangChain RAG Pipeline đã sẵn sàng phục vụ! (Direct max_tokens={direct_max_tokens}, RAG max_tokens={base_max_tokens})")

    def _format_docs_for_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "Không tìm thấy trích đoạn pháp lý liên quan trong cơ sở dữ liệu."
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            blocks.append(f"[Trích đoạn {i} ({chunk.source_type})]:\n{chunk.text}")
        return "\n\n".join(blocks)

    def _verify_chunks_nli(self, query: str, chunks: List[RetrievedChunk]) -> tuple[List[RetrievedChunk], Optional[NLIVerificationResult]]:
        """Kiểm chứng logic NLI trực tiếp đối với các retrieved chunks."""
        if not chunks or not query or not query.strip():
            return chunks, None

        clean_query = query.strip()
        items_payload = [
            {"specific_question": clean_query, "legal_document": chunk.text, "id": str(idx)}
            for idx, chunk in enumerate(chunks)
        ]

        predictions_map: Dict[int, Any] = {}
        # 1. Ưu tiên gọi batch predict
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(NLI_BATCH_URL, json={"items": items_payload})
                if res.status_code == 200:
                    data = res.json()
                    preds = data.get("predictions", [])
                    for i, pred in enumerate(preds):
                        predictions_map[i] = pred
        except Exception as e:
            logger.warning(f"Lỗi batch NLI ({e}), chuyển sang fallback gọi đơn lẻ...")

        # 2. Fallback gọi đơn lẻ nếu batch không phản hồi
        if not predictions_map:
            try:
                with httpx.Client(timeout=6.0) as client:
                    for idx, chunk in enumerate(chunks):
                        res = client.post(
                            NLI_SERVICE_URL,
                            json={"specific_question": clean_query, "legal_document": chunk.text},
                        )
                        if res.status_code == 200:
                            predictions_map[idx] = res.json().get("prediction", {})
            except Exception as e:
                logger.warning(f"Lỗi kết nối NLI Service: {e}")

        # Gán nhãn NLI cho từng retrieved chunk
        valid_count = 0
        for idx, chunk in enumerate(chunks):
            pred = predictions_map.get(idx)
            if pred:
                label_id = pred.get("label_id", 0)
                label = pred.get("label", "UNKNOWN")
                confidence = float(pred.get("confidence", 0.0))
                probs = pred.get("probabilities", {})

                is_valid = (label_id == 1 or "ENTAILMENT" in label.upper() or "WIN" in label.upper())
                if is_valid:
                    valid_count += 1

                note = (
                    f"Trích đoạn #{idx + 1}: Hợp lệ (Entailment/Win)."
                    if is_valid
                    else f"Trích đoạn #{idx + 1}: Mâu thuẫn/Không đủ căn cứ (Contradiction/Lose)."
                )
                chunk.nli_verification = NLIVerificationResult(
                    is_valid=is_valid,
                    label=label,
                    confidence=confidence,
                    probabilities=probs,
                    note=note,
                )

        if not predictions_map:
            return chunks, None

        # Tổng hợp kết quả NLI chung
        overall_is_valid = valid_count > 0
        if overall_is_valid:
            valid_confs = [c.nli_verification.confidence for c in chunks if c.nli_verification and c.nli_verification.is_valid]
            avg_conf = round(sum(valid_confs) / len(valid_confs), 4) if valid_confs else 0.0
            overall_nli = NLIVerificationResult(
                is_valid=True,
                label=f"ENTAILMENT/WIN ({valid_count}/{len(chunks)})",
                confidence=avg_conf,
                probabilities={"ENTAILMENT/WIN": avg_conf, "CONTRADICTION/LOSE": round(1.0 - avg_conf, 4)},
                note=f"Đánh giá NLI: Có {valid_count}/{len(chunks)} trích đoạn pháp lý được xác thực hợp lệ (Entailment).",
            )
        else:
            confs = [c.nli_verification.confidence for c in chunks if c.nli_verification]
            max_conf = round(max(confs), 4) if confs else 0.0
            overall_nli = NLIVerificationResult(
                is_valid=False,
                label=f"CONTRADICTION/LOSE (0/{len(chunks)})",
                confidence=max_conf,
                probabilities={"CONTRADICTION/LOSE": max_conf, "ENTAILMENT/WIN": round(1.0 - max_conf, 4)},
                note=f"Cảnh báo NLI: Toàn bộ {len(chunks)} trích đoạn pháp lý không thỏa mãn căn cứ xác thực câu hỏi (Contradiction).",
            )

        return chunks, overall_nli

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

        # 1. w/o RAG: Tự sinh điều luật qua HyDE -> Context -> RAG_PROMPT_TEMPLATE -> NLI trên chunk
        if not rag:
            hyde_law = generate_hypothetical_law(query=query_clean, llm=self.direct_llm_model)
            if not hyde_law:
                hyde_law = "Không có điều luật trích dẫn bổ sung."

            retrieved_chunks = [
                RetrievedChunk(
                    id="hyde-direct-0",
                    text=hyde_law,
                    score=1.0,
                    source_type="hyde_law",
                    metadata={"source": "Direct LLM HyDE", "type": "hypothetical_law"},
                )
            ]

            context_str = self._format_docs_for_context(retrieved_chunks)
            try:
                answer = str(self.direct_chain.invoke({"context": context_str, "query": query_clean})).strip()
            except Exception as e:
                logger.error(f"Lỗi Direct LLM: {e}")
                answer = f"Lỗi tạo câu trả lời: {str(e)}"

            # NLI kiểm chứng các retrieved chunks (ở đây là hyde_law)
            retrieved_chunks, nli_verification = self._verify_chunks_nli(query=query_clean, chunks=retrieved_chunks)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "request_id": request_id,
                "timestamp": timestamp,
                "query": query_clean,
                "rewritten_query": None,
                "rag_used": False,
                "retrieval_mode": "direct_hyde",
                "answer": answer,
                "nli_verification": nli_verification,
                "retrieved_chunks": retrieved_chunks,
                "total_chunks": len(retrieved_chunks),
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

        # NLI kiểm chứng đối với các retrieved chunks
        retrieved_chunks, nli_verification = self._verify_chunks_nli(query=query_clean, chunks=retrieved_chunks)
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
