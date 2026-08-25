import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder
from ..logger import logger

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from be.src.config.config import config
    DEFAULT_MODEL = config.models.reranker
    DEFAULT_DEVICE = config.models.device
    DEFAULT_TOP_K = config.retrieval.rerank_top_k
except Exception:
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_TOP_K = 5


class RerankerModel:
    """Class nạp và thực thi Reranker mô hình Cross-Encoder (bge-reranker-v2-m3) với CPU Fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device or (DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")
        self.top_k = top_k or DEFAULT_TOP_K
        self.cpu_fallback_model = None
        
        logger.info(f"Nạp Reranker Model: {self.model_name} trên {self.device}")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker Model đã sẵn sàng.")
        except Exception as e:
            logger.warning(f"Không thể nạp Reranker trên {self.device}, chuyển sang CPU: {e}")
            self.device = "cpu"
            self.model = CrossEncoder(self.model_name, device="cpu")

    def _get_cpu_model(self):
        if self.cpu_fallback_model is None:
            logger.info(f"Khởi tạo CPU fallback cho Reranker: {self.model_name}")
            self.cpu_fallback_model = CrossEncoder(self.model_name, device="cpu")
        return self.cpu_fallback_model

    def predict_scores(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        if not pairs:
            return []
        try:
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            return [float(s) for s in scores]
        except (RuntimeError, Exception) as e:
            msg = str(e).lower()
            if "cuda" in msg or "out of memory" in msg or "memory allocation" in msg:
                logger.warning("CUDA OOM trong Reranker, tự động fallback sang CPU.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scores = self._get_cpu_model().predict(pairs, batch_size=8, show_progress_bar=False)
                return [float(s) for s in scores]
            raise e

    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        if not documents:
            return []

        k = top_k or self.top_k

        doc_texts = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                doc_texts.append(doc.page_content)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("text") or doc.get("content", str(doc)))
            else:
                doc_texts.append(str(doc))

        pairs = [(query, text) for text in doc_texts]
        scores = self.predict_scores(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
