from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..logger import logger
from ..models.llm import get_llm

REWRITE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Hãy viết lại câu hỏi sau thành câu truy vấn luật Việt Nam.
Câu hỏi:
{query}
Chỉ trả về câu truy vấn luật viết lại, không giải thích gì thêm.
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
