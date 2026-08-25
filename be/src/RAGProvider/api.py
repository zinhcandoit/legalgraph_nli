# Input: JSON
# Output: Answer in JSON
# Should logging the process

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .core.pipeline import get_rag_pipeline
from .logger import logger
from .schemas import RAGRequest, RAGResponse

# Thư mục gốc ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi chạy và nạp sẵn toàn bộ mô hình khi server bật."""
    logger.info("🚀 Đang khởi động FastAPI RAG Provider...")
    pipeline = get_rag_pipeline()
    logger.info(f"🎯 RAG Provider sẵn sàng (Retrieval Mode: {pipeline.retriever.mode})")
    yield
    logger.info("🛑 FastAPI RAG Provider đã dừng.")


app = FastAPI(
    title="RAG Provider API (GraphRAG Legal Assistant)",
    description="Microservice hỏi đáp pháp luật tiếng Việt dựa trên Graph Database và LLM, hỗ trợ chế độ w/ RAG và w/o RAG kèm NLI Verification.",
    version="1.0.0",
    lifespan=lifespan,
)

# Kích hoạt CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Kiểm tra trạng thái hệ thống RAG và các models")
def health_check():
    try:
        pipeline = get_rag_pipeline()
        llm_model_name = getattr(pipeline.llm, "model", None) or getattr(pipeline.llm_model, "model_name", "Gemini")
        
        return {
            "status": "healthy",
            "service": "RAGProvider",
            "retrieval_mode": pipeline.retriever.mode,
            "embedding_model": pipeline.embedding.model_name,
            "reranker_model": pipeline.reranker.model_name,
            "llm_model": llm_model_name,
            "nli_service_configured": True,
            "device": pipeline.embedding.device,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Lỗi health check: {e}", exc_info=True)
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.post("/query", response_model=RAGResponse, summary="Xử lý câu hỏi với flag rag=True/False")
@app.post("/retrieve", response_model=RAGResponse, summary="Endpoint tương thích /retrieve")
def query_rag(request: RAGRequest):
    """
    Xử lý câu hỏi của người dùng:
    - `rag=True`: query -> query rewrite -> Graph DB -> Reranker -> TopK Chunks -> Enhanced LLM -> NLI Verification.
    - `rag=False`: Direct answer từ LLM -> NLI Verification kiểm tra căn cứ luật.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trường 'query' không được để trống.",
        )

    try:
        pipeline = get_rag_pipeline()
        result = pipeline.run(
            query=request.query,
            rag=request.rag,
            top_k=request.top_k,
            session_id=request.session_id,
            metadata=request.metadata,
        )
        return RAGResponse(**result)

    except Exception as e:
        logger.error(f"[API_ERROR] Lỗi xử lý query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình thực thi RAG Pipeline: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("be.src.RAGProvider.api:app", host="0.0.0.0", port=8002)
