import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from be.src.config.config import config
from ..logger import logger

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

GRAPH_DIR = ROOT_DIR / config.storage.graph_db_path
LANCEDB_DIR = ROOT_DIR / config.storage.lancedb_path


class GraphRetriever:
    """
    Module truy xuất dữ liệu từ Graph Database:
    - Kiểm tra linh hoạt theo từng request (Dynamic Detection):
      1. Nếu thư mục local LanceDB và Graph Parquet tồn tại -> Sử dụng LanceDB Vector Search kết hợp mapping Parquet text.
      2. Nếu local không có / không mở được -> Tự động chuyển sang Neo4j Cloud.
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.graph_dir = GRAPH_DIR
        self.lancedb_dir = LANCEDB_DIR
        self.neo4j_driver = None
        self.neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
        self._neo4j_attempted = False
        self._text_units_map: Optional[Dict[str, str]] = None
        self._community_reports_map: Optional[Dict[str, str]] = None
        self._load_parquet_cache()
        # Chỉ khởi tạo Neo4j nếu không có Local LanceDB
        if self._get_lancedb_table() is None:
            self._init_neo4j()

    def _load_parquet_cache(self):
        """Tải trước ánh xạ id -> text từ các file Parquet để phục vụ tra cứu sau khi vector search."""
        try:
            tu_path = self.graph_dir / "text_units.parquet"
            if tu_path.exists():
                df_tu = pd.read_parquet(tu_path)
                if "id" in df_tu.columns and "text" in df_tu.columns:
                    self._text_units_map = dict(zip(df_tu["id"].astype(str), df_tu["text"].astype(str)))
                    logger.info(f"Đã nạp {len(self._text_units_map)} text units từ {tu_path.name}")
        except Exception as e:
            logger.warning(f"Lỗi khi đọc text_units.parquet: {e}")
            self._text_units_map = {}

        try:
            cr_path = self.graph_dir / "community_reports.parquet"
            if cr_path.exists():
                df_cr = pd.read_parquet(cr_path)
                if "id" in df_cr.columns:
                    cr_map = {}
                    for _, row in df_cr.iterrows():
                        cid = str(row["id"])
                        full = str(row.get("full_content", ""))
                        if not full:
                            title = str(row.get("title", ""))
                            summary = str(row.get("summary", ""))
                            full = f"{title}: {summary}" if title or summary else ""
                        cr_map[cid] = full
                    self._community_reports_map = cr_map
                    logger.info(f"Đã nạp {len(self._community_reports_map)} community reports từ {cr_path.name}")
        except Exception as e:
            logger.warning(f"Lỗi khi đọc community_reports.parquet: {e}")
            self._community_reports_map = {}

    @property
    def mode(self) -> str:
        if self._get_lancedb_table() is not None:
            return "local_lancedb"
        elif self.neo4j_driver is not None:
            return "neo4j_cloud"
        return "none"

    def _get_lancedb_table(self):
        """Mở kết nối LanceDB table động theo thời gian thực (không cache cứng lúc khởi động)."""
        try:
            ldb_dir = self.lancedb_dir
            if ldb_dir.exists():
                import lancedb
                db = lancedb.connect(str(ldb_dir))
                table_names = db.table_names() if hasattr(db, "table_names") else db.list_tables()
                if "text_unit_text" in table_names:
                    return db.open_table("text_unit_text")
        except Exception as e:
            logger.debug(f"Không thể mở LanceDB: {e}")
        return None

    def _init_neo4j(self):
        """Khởi tạo kết nối Neo4j driver từ biến môi trường nếu có cấu hình."""
        if self.neo4j_driver is not None or self._neo4j_attempted:
            return
        self._neo4j_attempted = True
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

        if uri and username and password and "<" not in uri and "<" not in password:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                    connection_timeout=5.0,
                    max_connection_lifetime=300,
                )
                self.neo4j_driver.verify_connectivity()
                logger.info(f"Kết nối Neo4j sẵn sàng tại: {uri}")
            except Exception as e:
                logger.warning(
                    f"Chưa thể kết nối Neo4j ({e}). Sẽ ưu tiên sử dụng Local LanceDB làm kho lưu trữ."
                )
                self.neo4j_driver = None

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Truy xuất ngữ cảnh linh hoạt theo từng request:
        - Kiểm tra trực tiếp thư mục local LanceDB lúc gọi hàm.
        - Nếu có local LanceDB -> Thực hiện Vector Search và map với Parquet Text.
        - Nếu không có local LanceDB hoặc không ra kết quả -> Chuyển sang Neo4j.
        """
        candidates = []
        table = self._get_lancedb_table()

        # 1. Kiểm tra và ưu tiên Local LanceDB Vector Search
        if table is not None and self.embedding_model is not None:
            try:
                if self._text_units_map is None:
                    self._load_parquet_cache()

                q_vec = self.embedding_model.embed_query(query)
                results = table.search(q_vec).limit(top_k * 2).to_pandas()
                for _, row in results.iterrows():
                    chunk_id = str(row.get("id", ""))
                    text_content = ""
                    if self._text_units_map and chunk_id in self._text_units_map:
                        text_content = self._text_units_map[chunk_id]
                    elif "text" in row and pd.notna(row["text"]):
                        text_content = str(row["text"])

                    if text_content:
                        score = 0.0
                        if "_distance" in row:
                            score = round(max(0.0, 1.0 - float(row["_distance"])), 4)
                        candidates.append({
                            "id": chunk_id,
                            "text": text_content,
                            "source_type": "text_unit",
                            "score": score,
                            "metadata": {"search_mode": "lancedb_vector"},
                        })

                if candidates:
                    logger.info(f"Đã truy xuất {len(candidates[:top_k])} chunks từ Local LanceDB.")
                    return candidates[:top_k]
            except Exception as e:
                logger.warning(f"Vector search qua LanceDB không thành công ({e}). Chuyển sang Neo4j...")

        # 2. Nếu không có Local LanceDB hoặc LanceDB không có dữ liệu -> Fallback sang Neo4j
        if self.neo4j_driver is not None:
            logger.info("Chuyển sang truy xuất dữ liệu từ Neo4j Database...")
            neo4j_candidates = self._retrieve_neo4j(query, top_k=top_k)
            if neo4j_candidates:
                return neo4j_candidates

        return candidates[:top_k]

    def _retrieve_neo4j(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Truy vấn dữ liệu từ Neo4j Knowledge Graph theo chuẩn Vector & Fulltext Index:
        1. Thử Vector Index (db.index.vector.queryNodes) với embeddings bge-m3.
        2. Kết hợp Vector Search trên Community Reports.
        3. Fallback sang Fulltext Index hoặc Keyword search nếu Vector Index chưa được khởi tạo.
        """
        if not self.neo4j_driver:
            self._init_neo4j()
        if not self.neo4j_driver:
            logger.warning("Không có kết nối Neo4j khả dụng để truy xuất.")
            return []

        candidates = []
        retrieved_ids = set()

        # 1. Neo4j Vector Search (nếu có embedding_model)
        if self.embedding_model is not None:
            try:
                q_vec = [float(v) for v in self.embedding_model.embed_query(query)]
                
                # 1.1 Vector query trên TextUnit
                cypher_vector_tu = """
                CALL db.index.vector.queryNodes('text_unit_embeddings', $limit, $vector)
                YIELD node, score
                RETURN node.id AS id, node.text AS text, score, 'text_unit' AS source_type
                """
                with self.neo4j_driver.session(database=self.neo4j_db) as session:
                    result = session.run(cypher_vector_tu, vector=q_vec, limit=top_k * 2)
                    for record in result:
                        cid = str(record["id"])
                        if cid not in retrieved_ids:
                            retrieved_ids.add(cid)
                            candidates.append({
                                "id": cid,
                                "text": str(record["text"]),
                                "source_type": record["source_type"],
                                "score": float(record["score"]),
                                "metadata": {"source": "neo4j_vector", "score": float(record["score"])},
                            })

                # 1.2 Vector query trên CommunityReport nếu còn chỗ
                if len(candidates) < top_k:
                    cypher_vector_comm = """
                    CALL db.index.vector.queryNodes('community_report_embeddings', $limit, $vector)
                    YIELD node, score
                    RETURN node.id AS id, node.title + ': ' + node.summary AS text, score, 'community_report' AS source_type
                    """
                    with self.neo4j_driver.session(database=self.neo4j_db) as session:
                        result = session.run(cypher_vector_comm, vector=q_vec, limit=top_k - len(candidates))
                        for record in result:
                            cid = str(record["id"])
                            if cid not in retrieved_ids:
                                retrieved_ids.add(cid)
                                candidates.append({
                                    "id": cid,
                                    "text": str(record["text"]),
                                    "source_type": record["source_type"],
                                    "score": float(record["score"]),
                                    "metadata": {"source": "neo4j_vector_community", "score": float(record["score"])},
                                })

                if candidates:
                    logger.info(f"Đã truy xuất {len(candidates[:top_k])} chunks từ Neo4j Vector Index.")
                    return candidates[:top_k]

            except Exception as e:
                logger.warning(f"Neo4j Vector Search chưa khả dụng hoặc gặp lỗi ({e}). Fallback sang Fulltext/Keyword search...")

        # 2. Fallback sang Neo4j Fulltext Search
        try:
            cypher_fulltext = """
            CALL db.index.fulltext.queryNodes('text_unit_fulltext', $query_text)
            YIELD node, score
            RETURN node.id AS id, node.text AS text, score, 'text_unit' AS source_type
            LIMIT $limit
            """
            with self.neo4j_driver.session(database=self.neo4j_db) as session:
                result = session.run(cypher_fulltext, query_text=query, limit=top_k)
                for record in result:
                    cid = str(record["id"])
                    if cid not in retrieved_ids:
                        retrieved_ids.add(cid)
                        candidates.append({
                            "id": cid,
                            "text": str(record["text"]),
                            "source_type": record["source_type"],
                            "metadata": {"source": "neo4j_fulltext", "score": float(record["score"])},
                        })
            if candidates:
                logger.info(f"Đã truy xuất {len(candidates[:top_k])} chunks từ Neo4j Fulltext Index.")
                return candidates[:top_k]
        except Exception as e:
            logger.debug(f"Fulltext search lỗi: {e}")

        # 3. Fallback cuối: Tìm TextUnit theo từ khóa chứa trong chuỗi
        words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        cypher_textunit = """
        MATCH (t:TextUnit)
        WHERE any(word IN $words WHERE toLower(t.text) CONTAINS word)
        RETURN t.id AS id, t.text AS text, 'text_unit' AS source_type
        LIMIT $limit
        """
        try:
            with self.neo4j_driver.session(database=self.neo4j_db) as session:
                result = session.run(cypher_textunit, words=words, limit=top_k)
                for record in result:
                    cid = str(record["id"])
                    if cid not in retrieved_ids:
                        retrieved_ids.add(cid)
                        candidates.append({
                            "id": cid,
                            "text": str(record["text"]),
                            "source_type": record["source_type"],
                            "metadata": {"source": "neo4j_keyword"},
                        })
        except Exception as e:
            logger.error(f"Lỗi khi query TextUnit trên Neo4j: {e}")

        return candidates[:top_k]
