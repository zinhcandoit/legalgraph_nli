from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..logger import logger
from ..models.llm import get_llm

HYDE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Bạn là Quốc hội Việt Nam. Hãy viết ra 1 điều luật để hỗ trợ cho câu hỏi sau:
{query}
Chỉ trả về duy nhất điều luật đã viết, không giải thích gì thêm."""
)


def generate_hypothetical_law(query: str, llm=None) -> str:
    """
    Tự sinh trích dẫn điều luật giả định (HyDE - Hypothetical Document Embeddings)
    cho chế độ Direct LLM.
    """
    if not query or not query.strip():
        return ""

    try:
        model = llm or get_llm()
        llm_instance = model.get() if hasattr(model, "get") else model
        chain = HYDE_PROMPT_TEMPLATE | llm_instance | StrOutputParser()
        law_text = str(chain.invoke({"query": query.strip()})).strip()
        return law_text
    except Exception as e:
        logger.error(f"Lỗi khi sinh hypothetical law trong HyDE: {e}")
        return ""
