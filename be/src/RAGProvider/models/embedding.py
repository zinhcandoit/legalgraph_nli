import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from ..logger import logger

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from be.src.config.config import config
    DEFAULT_MODEL = config.models.embedding
    DEFAULT_DEVICE = config.models.device
except Exception:
    DEFAULT_MODEL = "BAAI/bge-m3"
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Class khởi tạo và quản lý mô hình Embedding (BAAI/bge-m3) với cơ chế tự động CPU Fallback khi CUDA OOM."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device or (DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")
        self.cpu_fallback_model = None
        
        logger.info(f"Nạp Embedding Model: {self.model_name} trên {self.device}")
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
            logger.info("Embedding Model đã sẵn sàng.")
        except Exception as e:
            logger.warning(f"Không thể nạp trên {self.device}, chuyển sang CPU: {e}")
            self.device = "cpu"
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
            )

    def _get_cpu_model(self):
        if self.cpu_fallback_model is None:
            logger.info(f"Khởi tạo CPU fallback cho Embedding: {self.model_name}")
            self.cpu_fallback_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
            )
        return self.cpu_fallback_model

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.model.embed_query(text)
        except (RuntimeError, Exception) as e:
            msg = str(e).lower()
            if "cuda" in msg or "out of memory" in msg or "memory allocation" in msg:
                logger.warning("CUDA OOM trong Embedding, tự động fallback sang CPU.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._get_cpu_model().embed_query(text)
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.model.embed_documents(texts)
        except (RuntimeError, Exception) as e:
            msg = str(e).lower()
            if "cuda" in msg or "out of memory" in msg or "memory allocation" in msg:
                logger.warning("CUDA OOM trong Embedding, tự động fallback sang CPU.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._get_cpu_model().embed_documents(texts)
            raise e

    def get(self) -> HuggingFaceEmbeddings:
        return self.model
