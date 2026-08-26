import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .logger import logger
from .models.service import get_nli_engine
from .schemas import (
    NLIBatchRequest,
    NLIItem,
    NLIPrediction,
    NLIRequest,
    NLIResponse,
    compute_sha256,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp trước mô hình NLI vào bộ nhớ khi ứng dụng khởi chạy."""
    logger.info("🚀 Đang khởi động FastAPI NLI Provider...")
    engine = get_nli_engine()
    logger.info(f"🎯 Model sẵn sàng phục vụ trên thiết bị: {engine.device}")
    yield
    logger.info("🛑 FastAPI NLI Provider đã dừng.")


app = FastAPI(
    title="NLI Provider API",
    description="API phân loại NLI (Natural Language Inference) cho văn bản pháp luật tiếng Việt",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Kiểm tra trạng thái dịch vụ và model")
def health_check():
    engine = get_nli_engine()
    return {
        "status": "healthy",
        "service": "NLIProvider",
        "model_name": engine.model_name,
        "local_model_dir": str(engine.model_dir),
        "device": str(engine.device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=NLIResponse, summary="Dự đoán NLI đơn lẻ (1 cặp câu)")
def predict_single(request: NLIRequest):
    """
    Input: JSON chứa `specific_question` (hoặc `question`) và `legal_document` (hoặc `context`).
    Output: JSON chứa kết quả phân loại, xác suất, metadata, latency và SHA-256 hash.
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Tính SHA-256 của toàn bộ payload để monitor/tracking/cache
        input_sha256 = compute_sha256(request.model_dump_json())

        # Tạo sha256 cho cặp câu hỏi - tài liệu
        item = NLIItem(
            specific_question=request.specific_question,
            legal_document=request.legal_document,
            id=request.id,
            metadata=request.metadata,
        )
        item_sha256 = item.get_sha256()

        # Thực thi mô hình NLI
        engine = get_nli_engine()
        raw_res = engine.predict(
            question=request.specific_question,
            context=request.legal_document,
        )

        prediction = NLIPrediction(
            id=request.id,
            label=str(raw_res["label"]),
            label_id=int(raw_res["label_id"]),
            confidence=float(raw_res["confidence"]),
            probabilities=raw_res["probabilities"],
            item_sha256=item_sha256,
            metadata=request.metadata,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"[PREDICT] req_id={request_id} session_id={request.session_id} "
            f"label={prediction.label} conf={prediction.confidence} "
            f"latency={latency_ms}ms input_sha256={input_sha256[:12]}..."
        )

        return NLIResponse(
            request_id=request_id,
            timestamp=timestamp,
            model_name=engine.model_name,
            device=str(engine.device),
            latency_ms=latency_ms,
            input_sha256=input_sha256,
            prediction=prediction,
            total_items=1,
            session_id=request.session_id,
            metadata=request.metadata,
        )

    except Exception as e:
        logger.error(f"[PREDICT_ERROR] req_id={request_id} detail={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi dự đoán NLI: {str(e)}",
        )


@app.post("/predict_batch", response_model=NLIResponse, summary="Dự đoán NLI theo lô (Batch)")
def predict_batch(request: NLIBatchRequest):
    """
    Input: JSON chứa `items: [{"specific_question": "...", "legal_document": "..."}, ...]`.
    Output: JSON danh sách kết quả, độ trễ và metadata tổng thể.
    """
    if not request.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách 'items' không được để trống.",
        )

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        input_sha256 = compute_sha256(request.model_dump_json())
        engine = get_nli_engine()

        pairs = [(item.specific_question, item.legal_document) for item in request.items]
        raw_results = engine.predict_batch(pairs)

        predictions: List[NLIPrediction] = []
        for item, raw_res in zip(request.items, raw_results):
            predictions.append(
                NLIPrediction(
                    id=item.id,
                    label=str(raw_res["label"]),
                    label_id=int(raw_res["label_id"]),
                    confidence=float(raw_res["confidence"]),
                    probabilities=raw_res["probabilities"],
                    item_sha256=item.get_sha256(),
                    metadata=item.metadata,
                )
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"[PREDICT_BATCH] req_id={request_id} session_id={request.session_id} "
            f"items_count={len(predictions)} latency={latency_ms}ms"
        )

        return NLIResponse(
            request_id=request_id,
            timestamp=timestamp,
            model_name=engine.model_name,
            device=str(engine.device),
            latency_ms=latency_ms,
            input_sha256=input_sha256,
            predictions=predictions,
            total_items=len(predictions),
            session_id=request.session_id,
            metadata=request.metadata,
        )

    except Exception as e:
        logger.error(f"[PREDICT_BATCH_ERROR] req_id={request_id} detail={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý batch NLI: {str(e)}",
        )


@app.post("/evaluate", summary="Đánh giá tập dữ liệu giống evaluate.ipynb")
def evaluate_dataset(payload: List[Dict[str, Any]]):
    """
    Input: Danh sách mẫu dữ liệu từ DataFrame/Parquet (như evaluate.ipynb).
    Output: JSON gồm metrics (Accuracy, F1, Precision, Recall) và dự đoán chi tiết.
    """
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tập dữ liệu không được để trống.",
        )

    start_time = time.perf_counter()
    engine = get_nli_engine()

    pairs = []
    y_true = []
    has_labels = True

    for sample in payload:
        q = sample.get("specific_question") or sample.get("question", "")
        ctx = sample.get("legal_document") or sample.get("context", "")
        pairs.append((q, ctx))

        if "answer" in sample:
            ans = sample["answer"]
            if isinstance(ans, int):
                y_true.append(1 if ans == 0 else 0)
            else:
                y_true.append(int(ans))
        elif "label" in sample:
            y_true.append(int(sample["label"]))
        else:
            has_labels = False

    raw_results = engine.predict_batch(pairs)
    y_pred = [r["label_id"] for r in raw_results]

    metrics = None
    if has_labels and len(y_true) == len(y_pred):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        metrics = {
            "total_samples": len(y_true),
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "F1-Score": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "Macro F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        }

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"[EVALUATE] total_samples={len(pairs)} latency={latency_ms}ms metrics={metrics}")

    return {
        "status": "success",
        "total_samples": len(pairs),
        "latency_ms": latency_ms,
        "metrics": metrics,
        "predictions": raw_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("be.src.NLIProvider.api:app", host="0.0.0.0", port=8001, reload=True)
