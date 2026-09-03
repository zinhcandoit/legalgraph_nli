from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    embedding: str = "BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    gemini: str = "gemma-4-26b-a4b-it"
    nli: str = "TQZinh/BamiBERT-ViLegalNLI"
    device: Optional[str] = "cuda"
    max_new_tokens: int = 2048
    temperature: float = 0.0


@dataclass
class StorageConfig:
    graph_db_path: str = "db/graph_database"
    lancedb_path: str = "db/graph_database/lancedb"
    log_dir: str = "be/src/logs"
    nli_dir: str = "bamibert_vilegalnli"


@dataclass
class RetrievalConfig:
    search_k: int = 20
    rerank_top_k: int = 3
    rewrite_query: bool = True


@dataclass
class PipelineConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


config = PipelineConfig()
