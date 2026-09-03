import os
import sys
from pathlib import Path
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from ..logger import logger

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Nạp biến môi trường từ .env
def load_env():
    for p in [ROOT_DIR, *ROOT_DIR.parents]:
        env_file = p / ".env"
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


load_env()

try:
    from be.src.config.config import config
    DEFAULT_MODEL = config.models.gemini
    DEFAULT_TEMPERATURE = config.models.temperature
    DEFAULT_MAX_TOKENS = config.models.max_new_tokens
except Exception:
    DEFAULT_MODEL = "gemma-4-26b-a4b-it"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 4096


def extract_content_text(response: Any) -> str:
    """
    Cơ chế chính thức từ Google GenAI SDK:
    - Trong Google SDK: Mỗi candidate có parts chứa `thought=True` và parts chứa câu trả lời `thought=None`.
    - Khi qua LangChain AIMessage: parts được lưu trong `content` list hoặc `response_metadata`.
    - Hàm này trích xuất đúng phần text câu trả lời theo đúng thuộc tính chuẩn của Google.
    """
    if hasattr(response, "text") and isinstance(response.text, str) and response.text:
        return response.text.strip()

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                # Google SDK đánh dấu thought qua trường 'type': 'thinking' hoặc 'thought': True
                if part.get("type") == "thinking" or part.get("thought") is True:
                    continue
                if "text" in part:
                    text_parts.append(part["text"])
            elif hasattr(part, "text") and not getattr(part, "thought", False):
                text_parts.append(part.text)
        if text_parts:
            return "\n".join(text_parts).strip()

    return str(content).strip()


class GeminiModel:
    """Wrapper quản lý Google Open-Weights Gemma LLM (gemma-4-26b-a4b-it) qua langchain_google_genai."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model or DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_output_tokens = max_output_tokens or DEFAULT_MAX_TOKENS
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            logger.warning("Chưa tìm thấy GEMINI_API_KEY trong môi trường.")

        logger.info(f"Khởi tạo Open-Weights Gemma LLM: {self.model_name} (temp={self.temperature}, max_tokens={self.max_output_tokens})")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        logger.info("Gemma LLM đã sẵn sàng.")

    def generate(self, prompt: str) -> str:
        """Gửi prompt tới LLM và nhận câu trả lời dạng chuỗi (text)."""
        response = self.llm.invoke(prompt)
        return extract_content_text(response)

    def get(self) -> ChatGoogleGenerativeAI:
        return self.llm


_llm_instance: Optional[GeminiModel] = None


def get_llm() -> GeminiModel:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GeminiModel()
    return _llm_instance
