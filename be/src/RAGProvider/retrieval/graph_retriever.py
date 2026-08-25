import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from ..logger import logger

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def find_graph_dir() -> Path:
    candidates = [
        ROOT_DIR / "db" / "graph_database",
        ROOT_DIR / "src" / "db" / "graph_database",
        ROOT_DIR / "db" / "src" / "graph_database",
    ]
    for c in candidates:
        if c.exists() and (c / "text_units.parquet").exists():
            return c
    return candidates[0]


class GraphRetriever:
    """
    Module truy xuất dữ liệu từ Graph Database kết hợp đa tầng:
    1. Vector Similarity Search qua LanceDB (với CPU fallback khi CUDA đầy VRAM)
    2. Full-text Keyword & Term Matching trên toàn bộ 170 Text Units
    3. Community Summaries & Entities Descriptions
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.graph_dir = find_graph_dir()
        self.lancedb_dir = self.graph_dir / "lancedb"

        if self._is_local_available():
            self.mode = "local_graph_db"
            logger.info(f"Sử dụng Local Graph Database tại: {self.graph_dir}")
            self._init_local_db()
        else:
            self.mode = "neo4j_instance"
            logger.info("Chưa có đầy đủ dữ liệu local. Chuyển sang kết nối Neo4j Instance.")
            self._init_neo4j()

    def _is_local_available(self) -> bool:
        text_units_file = self.graph_dir / "text_units.parquet"
        return self.graph_dir.exists() and text_units_file.exists()

    def _init_local_db(self):
        self.df_text_units = None
        self.df_entities = None
        self.df_community_reports = None
        self.lance_table = None

        try:
            tu_path = self.graph_dir / "text_units.parquet"
            if tu_path.exists():
                self.df_text_units = pd.read_parquet(tu_path).set_index("id", drop=False)
                logger.info(f"Đã nạp {len(self.df_text_units)} TextUnits từ Parquet.")

            ent_path = self.graph_dir / "entities.parquet"
            if ent_path.exists():
                self.df_entities = pd.read_parquet(ent_path)

            rep_path = self.graph_dir / "community_reports.parquet"
            if rep_path.exists():
                self.df_community_reports = pd.read_parquet(rep_path)

            if self.lancedb_dir.exists():
                import lancedb
                db = lancedb.connect(str(self.lancedb_dir))
                table_names = db.table_names() if hasattr(db, "table_names") else db.list_tables()
                if "text_unit_text" in table_names:
                    self.lance_table = db.open_table("text_unit_text")
                    logger.info("LanceDB Vector Index text_unit_text đã kết nối thành công.")
        except Exception as e:
            logger.warning(f"Lỗi nạp local graph data: {e}")

    def _init_neo4j(self):
        self.neo4j_driver = None
        self.neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if uri and username and password:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))
                self.neo4j_driver.verify_connectivity()
                logger.info(f"Kết nối Neo4j thành công tại: {uri}")
            except Exception as e:
                logger.error(f"Không thể kết nối Neo4j: {e}")

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        if self.mode == "local_graph_db":
            return self._retrieve_local(query, top_k=top_k)
        else:
            return self._retrieve_neo4j(query, top_k=top_k)

    def _retrieve_local(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        candidates = []
        retrieved_ids = set()

        # 1. Vector search với LanceDB
        if self.lance_table is not None and self.embedding_model is not None:
            try:
                q_vec = self.embedding_model.embed_query(query)
                results = self.lance_table.search(q_vec).limit(top_k * 2).to_pandas()
                
                for _, row in results.iterrows():
                    chunk_id = row["id"]
                    if chunk_id in retrieved_ids:
                        continue
                    
                    text_content = ""
                    if self.df_text_units is not None and chunk_id in self.df_text_units.index:
                        text_content = str(self.df_text_units.loc[chunk_id]["text"])
                    
                    if text_content:
                        retrieved_ids.add(chunk_id)
                        candidates.append({
                            "id": chunk_id,
                            "text": text_content,
                            "source_type": "text_unit",
                            "metadata": {"search_mode": "vector"},
                        })
            except Exception as e:
                logger.warning(f"Vector search qua LanceDB gặp lỗi: {e}")

        # 2. Keyword & Text Unit search (Trích xuất từ khóa pháp lý)
        if self.df_text_units is not None:
            words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
            scored_units = []
            for _, row in self.df_text_units.iterrows():
                chunk_id = row["id"]
                if chunk_id in retrieved_ids:
                    continue
                text = str(row["text"])
                text_lower = text.lower()
                
                match_count = sum(1 for w in words if w in text_lower)
                if match_count > 0:
                    scored_units.append((match_count, chunk_id, text))

            # Sắp xếp theo số lượng từ khóa trùng khớp
            scored_units.sort(key=lambda x: x[0], reverse=True)
            for match_count, chunk_id, text in scored_units:
                if len(candidates) >= top_k * 2:
                    break
                retrieved_ids.add(chunk_id)
                candidates.append({
                    "id": chunk_id,
                    "text": text,
                    "source_type": "text_unit",
                    "metadata": {"search_mode": "keyword", "matches": match_count},
                })

        # 3. Community Reports search nếu vẫn còn chỗ
        if self.df_community_reports is not None and len(candidates) < top_k:
            for _, row in self.df_community_reports.iterrows():
                summary = str(row.get("summary", ""))
                title = str(row.get("title", ""))
                comb = f"{title}: {summary}"
                candidates.append({
                    "id": str(row.get("id", "")),
                    "text": comb,
                    "source_type": "community_report",
                    "metadata": {"rank": row.get("rank", 0)},
                })
                if len(candidates) >= top_k:
                    break

        # 4. Fallback: Nếu không tìm thấy gì, lấy 5 điều luật quan trọng đầu tiên
        if len(candidates) == 0 and self.df_text_units is not None:
            for _, row in self.df_text_units.head(top_k).iterrows():
                candidates.append({
                    "id": row["id"],
                    "text": str(row["text"]),
                    "source_type": "text_unit",
                    "metadata": {"search_mode": "fallback"},
                })

        return candidates[:top_k]

    def _retrieve_neo4j(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.neo4j_driver:
            return []

        candidates = []
        cypher_query = """
        MATCH (t:TextUnit)
        WHERE any(word IN split(toLower($query), ' ') WHERE toLower(t.text) CONTAINS word AND size(word) > 2)
        RETURN t.id AS id, t.text AS text, 'text_unit' AS source_type
        LIMIT $limit
        """
        try:
            with self.neo4j_driver.session(database=self.neo4j_db) as session:
                result = session.run(cypher_query, query=query, limit=top_k)
                for record in result:
                    candidates.append({
                        "id": record["id"],
                        "text": record["text"],
                        "source_type": record["source_type"],
                        "metadata": {"source": "neo4j"},
                    })
        except Exception as e:
            logger.error(f"Lỗi khi query Neo4j: {e}")

        return candidates
