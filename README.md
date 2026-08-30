# VietLegal AI - Hệ thống RAG & NLI Luật Tiếng Việt

**Lưu ý**: Hiện tại GraphDB đang tối ưu cho **Luật Đất Đai** nên cần chỉnh lại nếu muốn tối ưu cho các luật khác. Chi tiết trong phần **III.2.**.
---

## I. Cấu trúc thư mục dự án

```text
root/
├── .env                      # File cấu hình API keys (GEMINI_API_KEY, NEO4J_*, NVIDIA_KEY, HF_KEY)
├── pyproject.toml            # Cấu hình gói phụ thuộc Backend qua uv
├── README.md                 # Tài liệu hướng dẫn sử dụng
│
├── be/                       # [Backend] Microservices & Models
│   └── src/
│       ├── config/           # Cấu hình hệ thống (config.py, train.yaml)
│       ├── logs/             # Thư mục chứa log xoay vòng (Rotating File Logs)
│       ├── NLIProvider/      # Microservice NLI (FastAPI, Model BamiBERT, Training Script)
│       └── RAGProvider/      # Microservice GraphRAG (FastAPI, LangChain LCEL, Dual Retriever)
│
├── db/                       # [Database] Graph Database & Benchmark Data
│   ├── settings.yaml         # Cấu hình Microsoft GraphRAG Indexing
│   ├── input/                # Văn bản Luật đầu vào
│   ├── graph_database/       # Output GraphRAG (Parquet + LanceDB + Script Import Neo4j)
│   ├── ViLegalNLI/           # Dataset NLI tải từ ntphuc149/ViLegalNLI
│   └── eval/                 # Tập dữ liệu kiểm thử (vlsp_nli.parquet, ...)
│
└── fe/                       # [Frontend] Web Application (React + TypeScript + Tailwind CSS)
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    └── src/
        ├── components/       # Sidebar, Header, ChatArea, ChatInput, ChunkModal
        ├── hooks/            # useChatStorage (Lưu lịch sử localStorage)
        └── services/         # API Client kết nối FastAPI
```

---

## II. Cài đặt & Chuẩn bị Môi trường

### 1. Cài đặt Backend qua uv
Mở terminal tại thư mục gốc `root/`:
```bash
# Đồng bộ môi trường Python & các thư viện (GraphRAG, Transformers, Torch CUDA, FastAPI)
uv sync
```

### 2. Thiết lập cấu hình `.env`
- Tạo file `.env` tại thư mục gốc:
```bash
cp .env.example .env
```
- Điền vào các thông tin:
```env
GEMINI_API_KEY=AIzaSy...  # Google AI Studio Key
NVIDIA_KEY=nvapi-...      # NVIDIA NIM KEY
HF_KEY=hf_...             # Huggingface Key

# Cấu hình Neo4j (Tùy chọn khi đồng bộ dữ liệu đồ thị)
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password_cua_ban>
NEO4J_DATABASE=neo4j
```

---

## III. Huấn luyện Model & Xây dựng Graph Database

### 1. Huấn luyện Mô hình NLI (BamiBERT)
Script huấn luyện tự động tải dataset `ntphuc149/ViLegalNLI` về `db/ViLegalNLI/` và xuất mô hình tốt nhất vào `be/src/NLIProvider/models/bamibert_vilegalnli/`:
```bash
uv run python be/src/NLIProvider/models/train.py --config be/src/config/train.yaml
```

### 2. Xây dựng Knowledge Graph Database (Microsoft GraphRAG)
- Tiền đề: Khởi động Ollama chứa embedding model `bge-m3`:
```bash
ollama run bge-m3
```
- Bỏ các file tài liệu vào `db/input/`
- (Tùy chọn) Tự động sinh Prompts cho lĩnh vực luật mới (Auto-Prompt Tuning):
```bash
uv run python --env-file .env -m graphrag prompt-tune --root db --domain "your_domain" --selection-method random --language Vietnamese --output db/prompts
```
- Chạy pipeline indexing của Microsoft GraphRAG từ thư mục `db/`:
```bash
uv run python -m graphrag index --root db
```
*Dữ liệu đồ thị sau khi trích xuất (entities, relationships, text units, community reports, lancedb vector index) sẽ được lưu tự động trong `db/graph_database/`.*
- (Tuỳ chọn) Đồng bộ Knowledge Graph từ `db/graph_database/` lên Neo4j Database:
```bash
uv run python be/src/RAGProvider/storage/vector_store.py
```
*Ghi chú: Thêm cờ `--clear` nếu muốn xóa sạch dữ liệu cũ trong Neo4j trước khi import:*
---

## IV. Khởi chạy hệ thống Microservices

### Bước 1: Khởi chạy RAG Backend (Port 8002)
```bash
uv run uvicorn be.src.RAGProvider.api:app --host 0.0.0.0 --port 8002
```
* **Swagger API Docs**: `http://localhost:8002/docs`

### Bước 2: Khởi chạy NLI Backend (Port 8001 - Tùy chọn)
```bash
uv run uvicorn be.src.NLIProvider.api:app --host 0.0.0.0 --port 8001
```
* **Swagger API Docs**: `http://localhost:8001/docs`

### Bước 3: Khởi chạy Frontend UI (Port 3000)
Mở một terminal mới tại thư mục `fe/` và chạy:
```bash
cd fe
npm install
npm run dev
```
* Truy cập giao diện tại: **`http://localhost:3000`**

---

## V. Luồng xử lý Pipeline (RAG + NLI Verification)

```text
User Query => Query Rewriter => Graph DB (LanceDB/Neo4j) => TopK Chunks => Enhanced Prompt => Gemini Answer => NLI Verification (BamiBERT) => UI
```

* **Chế độ `rag = True` (w/ RAG)**: Tra cứu căn cứ pháp lý trong đồ thị Luật, rerank văn bản và chạy NLI kiểm chứng xem câu trả lời có được chứng thực (**Entailment**) bởi điều luật hay không.
* **Chế độ `rag = False` (w/o RAG)**: Hỏi trực tiếp LLM và chạy NLI để phát hiện **Ảo giác (Hallucination / Contradiction)**.