import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def compute_sha256(text: str) -> str:
    """Tính toán SHA-256 hash của một chuỗi văn bản."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NLIVerificationResult(BaseModel):
    """Kết quả kiểm chứng logic pháp lý từ mô hình NLI (BamiBERT-ViLegalNLI)."""
    is_valid: bool = Field(..., description="True nếu Entailment (hợp lệ), False nếu Contradiction (sai lệch)")
    label: str = Field(..., description="Nhãn dự đoán: 'ENTAILMENT/WIN' hoặc 'CONTRADICTION/LOSE'")
    confidence: float = Field(..., description="Độ tin cậy của dự đoán (0.0 -> 1.0)")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Phân phối xác suất 2 nhãn")
    note: Optional[str] = Field(default=None, description="Ghi chú giải thích đánh giá NLI")


class RAGRequest(BaseModel):
    """Schema input cho RAG Provider."""
    query: str = Field(..., description="Câu hỏi hoặc yêu cầu cần tra cứu pháp luật của người dùng")
    rag: bool = Field(
        default=True,
        description="Flag quyết định dùng RAG (True) hay trả lời trực tiếp từ LLM (False)"
    )
    top_k: Optional[int] = Field(default=None, description="Số lượng đoạn văn bản trích dẫn tối đa đưa vào LLM (nếu None sẽ lấy theo RetrievalConfig trong config.py)")
    session_id: Optional[str] = Field(default=None, description="ID phiên hội thoại")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata kèm theo request")


class RetrievedChunk(BaseModel):
    """Chi tiết một đoạn trích dẫn được truy xuất từ Graph DB."""
    id: Optional[str] = Field(default=None, description="ID của chunk / node")
    text: str = Field(..., description="Nội dung trích đoạn pháp lý")
    score: Optional[float] = Field(default=None, description="Điểm số liên quan từ Reranker")
    source_type: Optional[str] = Field(default="text_unit", description="Nguồn (text_unit / entity / community / neo4j / hyde_law)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata bổ sung")
    nli_verification: Optional[NLIVerificationResult] = Field(
        default=None,
        description="Kết quả kiểm định NLI giữa câu hỏi và đoạn trích dẫn này",
    )


class RAGResponse(BaseModel):
    """Schema output JSON trả về cho frontend."""
    request_id: str = Field(..., description="UUID duy nhất của lượt gọi API")
    timestamp: str = Field(..., description="Thời gian thực thi (ISO 8601 UTC)")
    query: str = Field(..., description="Câu hỏi gốc của người dùng")
    rewritten_query: Optional[str] = Field(default=None, description="Câu hỏi đã được viết lại sang văn phong pháp lý")
    rag_used: bool = Field(..., description="Có sử dụng RAG để tra cứu dữ liệu hay không")
    retrieval_mode: Optional[str] = Field(default=None, description="Chế độ truy xuất: 'local_graph_db' hoặc 'neo4j_instance'")
    answer: str = Field(..., description="Câu trả lời từ LLM")
    nli_verification: Optional[NLIVerificationResult] = Field(
        default=None, 
        description="Kết quả xác thực NLI xem câu trả lời có được chứng thực bởi căn cứ luật hay không"
    )
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Danh sách các đoạn văn bản trích dẫn")
    total_chunks: int = Field(default=0, description="Số lượng trích đoạn được sử dụng")
    latency_ms: float = Field(..., description="Tổng thời gian xử lý (ms)")
    input_sha256: str = Field(..., description="Mã SHA-256 hash của payload input")
    session_id: Optional[str] = Field(default=None, description="ID phiên chat")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata tổng hợp")
