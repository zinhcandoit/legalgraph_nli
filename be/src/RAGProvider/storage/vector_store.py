import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Thêm ROOT_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ..logger import logger

# Tải biến môi trường từ .env
load_dotenv(ROOT_DIR / ".env")


def sanitize_list(val: Any) -> List[str]:
    """Chuyển numpy array hoặc list thành python list strings an toàn cho Neo4j."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, np.ndarray)):
        return [str(x) for x in val if x is not None]
    return [str(val)]


class Neo4jGraphStore:
    """
    Quản lý đồng bộ và import dữ liệu Knowledge Graph từ GraphRAG Parquet files lên Neo4j.
    Các thực thể (Entities), Mối quan hệ (Relationships), Đoạn văn bản (TextUnits), và Báo cáo cộng đồng (CommunityReports)
    sẽ được import với cấu trúc Node và Edge tối ưu cho truy vấn đồ thị pháp luật.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None

        if not self.uri or not self.username or not self.password:
            raise ValueError(
                "Thiếu thông tin kết nối Neo4j! Vui lòng kiểm tra NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD trong file .env"
            )

        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self.driver.verify_connectivity()
        logger.info(f"Kết nối Neo4j thành công tại: {self.uri} (Database: {self.database})")

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Đã đóng kết nối Neo4j.")

    def create_constraints_and_indexes(self):
        """Tạo Constraints (Unique) và Indexes trên Neo4j để tối ưu hiệu năng import và query."""
        queries = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT text_unit_id_unique IF NOT EXISTS FOR (t:TextUnit) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:CommunityReport) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX entity_title_idx IF NOT EXISTS FOR (e:Entity) ON (e.title)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX text_unit_doc_idx IF NOT EXISTS FOR (t:TextUnit) ON (t.document_id)",
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Lỗi khi tạo schema/index ('{q}'): {e}")
        logger.info("Đã thiết lập xong Neo4j Constraints & Indexes.")

    def clear_database(self):
        """Xóa toàn bộ dữ liệu trong Neo4j (Cẩn trọng)."""
        logger.warning("Đang xóa toàn bộ dữ liệu hiện có trong Neo4j Database...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Đã dọn dẹp sạch Neo4j Database.")

    def import_text_units(self, parquet_path: Path, batch_size: int = 50):
        """Import TextUnits từ text_units.parquet vào node (:TextUnit)."""
        if not parquet_path.exists():
            logger.warning(f"Không tìm thấy file {parquet_path}")
            return

        df = pd.read_parquet(parquet_path)
        logger.info(f"Bắt đầu import {len(df)} TextUnits...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row["id"]),
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "text": str(row.get("text", "")),
                "n_tokens": int(row.get("n_tokens", 0)),
                "document_id": str(row.get("document_id", "")),
            })

        query = """
        UNWIND $batch AS row
        MERGE (t:TextUnit {id: row.id})
        SET t.human_readable_id = row.human_readable_id,
            t.text = row.text,
            t.n_tokens = row.n_tokens,
            t.document_id = row.document_id
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                session.run(query, batch=batch)
        logger.info(f"Đã import thành công {len(records)} TextUnits vào Neo4j.")

    def import_entities(self, parquet_path: Path, batch_size: int = 200):
        """Import Entities từ entities.parquet vào node (:Entity)."""
        if not parquet_path.exists():
            logger.warning(f"Không tìm thấy file {parquet_path}")
            return

        df = pd.read_parquet(parquet_path)
        logger.info(f"Bắt đầu import {len(df)} Entities...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row["id"]),
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "title": str(row.get("title", "")).strip(),
                "name": str(row.get("title", "")).strip(),
                "type": str(row.get("type", "UNKNOWN")),
                "description": str(row.get("description", "")),
                "frequency": int(row.get("frequency", 1)),
                "degree": int(row.get("degree", 0)),
                "text_unit_ids": sanitize_list(row.get("text_unit_ids")),
            })

        # 1. Tạo Entity Node
        node_query = """
        UNWIND $batch AS row
        MERGE (e:Entity {name: row.name})
        SET e.id = row.id,
            e.title = row.title,
            e.type = row.type,
            e.description = row.description,
            e.frequency = row.frequency,
            e.degree = row.degree
        """

        # 2. Tạo liên kết (TextUnit)-[:MENTIONS]->(Entity)
        rel_query = """
        UNWIND $batch AS row
        MATCH (e:Entity {id: row.id})
        UNWIND row.text_unit_ids AS tu_id
        MATCH (t:TextUnit {id: tu_id})
        MERGE (t)-[:MENTIONS]->(e)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                session.run(node_query, batch=batch)
                session.run(rel_query, batch=batch)

        logger.info(f"Đã import thành công {len(records)} Entities và liên kết MENTIONS.")

    def import_relationships(self, parquet_path: Path, batch_size: int = 200):
        """Import Relationships từ relationships.parquet thành quan hệ [:RELATED_TO] giữa các (:Entity)."""
        if not parquet_path.exists():
            logger.warning(f"Không tìm thấy file {parquet_path}")
            return

        df = pd.read_parquet(parquet_path)
        logger.info(f"Bắt đầu import {len(df)} Relationships...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row["id"]),
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "source": str(row.get("source", "")).strip(),
                "target": str(row.get("target", "")).strip(),
                "description": str(row.get("description", "")),
                "weight": float(row.get("weight", 1.0)),
                "combined_degree": int(row.get("combined_degree", 0)),
                "text_unit_ids": sanitize_list(row.get("text_unit_ids")),
            })

        query = """
        UNWIND $batch AS row
        MERGE (src:Entity {name: row.source})
        MERGE (tgt:Entity {name: row.target})
        MERGE (src)-[r:RELATED_TO {id: row.id}]->(tgt)
        SET r.human_readable_id = row.human_readable_id,
            r.description = row.description,
            r.weight = row.weight,
            r.combined_degree = row.combined_degree,
            r.text_unit_ids = row.text_unit_ids
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                session.run(query, batch=batch)

        logger.info(f"Đã import thành công {len(records)} Relationships giữa các Entity.")

    def import_community_reports(self, parquet_path: Path, batch_size: int = 50):
        """Import Community Reports từ community_reports.parquet vào node (:CommunityReport)."""
        if not parquet_path.exists():
            logger.warning(f"Không tìm thấy file {parquet_path}")
            return

        df = pd.read_parquet(parquet_path)
        logger.info(f"Bắt đầu import {len(df)} Community Reports...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row["id"]),
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "community": int(row.get("community", 0)),
                "level": int(row.get("level", 0)),
                "parent": int(row.get("parent", -1)),
                "title": str(row.get("title", "")),
                "summary": str(row.get("summary", "")),
                "full_content": str(row.get("full_content", "")),
                "rank": float(row.get("rank", 0.0)),
                "rating_explanation": str(row.get("rating_explanation", "")),
            })

        query = """
        UNWIND $batch AS row
        MERGE (c:CommunityReport {id: row.id})
        SET c.human_readable_id = row.human_readable_id,
            c.community = row.community,
            c.level = row.level,
            c.parent = row.parent,
            c.title = row.title,
            c.summary = row.summary,
            c.full_content = row.full_content,
            c.rank = row.rank,
            c.rating_explanation = row.rating_explanation
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                session.run(query, batch=batch)

        logger.info(f"Đã import thành công {len(records)} Community Reports vào Neo4j.")

    def import_all_from_dir(self, graph_dir: Optional[Union[str, Path]] = None, clear_first: bool = False):
        """Đồng bộ toàn bộ các bảng Parquet trong thư mục graph_database lên Neo4j."""
        target_dir = Path(graph_dir) if graph_dir else ROOT_DIR / "db" / "graph_database"
        if not target_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục graph database tại: {target_dir}")

        logger.info(f"=== BẮT ĐẦU ĐỒNG BỘ DỮ LIỆU GRAPH TỪ '{target_dir}' LÊN NEO4J ===")

        if clear_first:
            self.clear_database()

        self.create_constraints_and_indexes()

        # 1. Text Units
        self.import_text_units(target_dir / "text_units.parquet")

        # 2. Entities
        self.import_entities(target_dir / "entities.parquet")

        # 3. Relationships
        self.import_relationships(target_dir / "relationships.parquet")

        # 4. Community Reports
        self.import_community_reports(target_dir / "community_reports.parquet")

        # In thống kê sau khi import
        self.print_graph_stats()
        logger.info("=== HOÀN TẤT ĐỒNG BỘ DỮ LIỆU LÊN NEO4J ===")

    def print_graph_stats(self):
        """In tổng số lượng Node và Relationship hiện có trên Neo4j."""
        with self.driver.session(database=self.database) as session:
            node_counts = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count")
            logger.info("--- Thống kê Node trong Neo4j ---")
            for r in node_counts:
                logger.info(f"Node [:{r['label']}]: {r['count']}")

            rel_counts = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            logger.info("--- Thống kê Relationship trong Neo4j ---")
            for r in rel_counts:
                logger.info(f"Relationship [:{r['type']}]: {r['count']}")


def sync_graph_to_neo4j(graph_dir: Optional[str] = None, clear_first: bool = False):
    """Hàm wrapper tiện lợi để gọi từ bên ngoài hoặc CLI."""
    store = Neo4jGraphStore()
    try:
        store.import_all_from_dir(graph_dir=graph_dir, clear_first=clear_first)
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đồng bộ Knowledge Graph từ Parquet (GraphRAG) lên Neo4j")
    parser.add_argument(
        "--graph-dir",
        type=str,
        default=str(ROOT_DIR / "db" / "graph_database"),
        help="Đường dẫn đến thư mục graph_database chứa các file parquet",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa sạch dữ liệu cũ trong Neo4j trước khi import",
    )
    args = parser.parse_args()

    sync_graph_to_neo4j(graph_dir=args.graph_dir, clear_first=args.clear)
