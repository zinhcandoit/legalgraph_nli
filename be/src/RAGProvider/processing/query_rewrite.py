from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..logger import logger
from ..models.llm import get_llm

REWRITE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Bạn là chuyên gia tra cứu thông tin pháp luật Việt Nam.
Hãy phân tích câu hỏi người dùng và viết lại thành 1 câu truy vấn ngắn gọn, chuẩn xác thuật ngữ Luật Đất đai 2024 để tra cứu trong cơ sở dữ liệu.

Quy tắc:
1. Giữ nguyên ý nghĩa cốt lõi của câu hỏi.
2. CHỈ TRẢ VỀ DUY NHẤT 1 CÂU TRUY VẤN VIẾT LẠI, KHÔNG GIẢI THÍCH, KHÔNG THÊM BẤT KỲ LỜI DẪN NÀO.

Câu hỏi gốc: {query}
Câu truy vấn viết lại:"""
)


def rewrite_query(query: str, llm=None) -> str:
    """Viết lại câu hỏi người dùng qua chuỗi LangChain LCEL chuẩn: REWRITE_PROMPT | LLM | StrOutputParser."""
    if not query or not query.strip():
        return query

    try:
        model = llm or get_llm()
        llm_instance = model.get()
        chain = REWRITE_PROMPT_TEMPLATE | llm_instance | StrOutputParser()
        rewritten = str(chain.invoke({"query": query.strip()})).strip().strip("'\"")
        
        # Nếu có nhiều dòng, lấy dòng đầu tiên
        if "\n" in rewritten:
            lines = [l.strip() for l in rewritten.split("\n") if l.strip() and not l.strip().startswith("```")]
            rewritten = lines[0] if lines else rewritten

        return rewritten if rewritten else query
    except Exception as e:
        logger.warning(f"Lỗi khi rewrite query '{query}': {e}. Sử dụng query gốc.")
        return query
