import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..logger import logger

# Xác định các đường dẫn gốc
CURRENT_DIR = Path(__file__).resolve().parent  # be/src/NLIProvider/models
ROOT_DIR = Path(__file__).resolve().parents[4]  # ML_final
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "train.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Nạp file cấu hình huấn luyện YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_token() -> Optional[str]:
    """Tìm Hugging Face Token trong các file .env."""
    for p in [ROOT_DIR, CURRENT_DIR, *CURRENT_DIR.parents]:
        env_file = p / ".env"
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() in ("HF_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
                                return v.strip().strip("'\"")
            except Exception:
                pass
    return os.getenv("HF_KEY") or os.getenv("HF_TOKEN")


def ensure_dataset(data_cfg: Dict[str, Any]) -> tuple[Path, Path]:
    """Tải và lưu các file dataset train.csv và val.csv từ Hugging Face Hub về thư mục local nếu chưa có."""
    repo_id = data_cfg.get("hf_dataset_repo", "ntphuc149/ViLegalNLI")
    raw_data_dir = data_cfg.get("data_dir", "db/ViLegalNLI")

    # Xác định đường dẫn tuyệt đối cho data_dir
    data_dir = Path(raw_data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT_DIR / data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_cfg.get("train_file", "train.csv")
    val_file = data_cfg.get("val_file", "val.csv")

    train_path = data_dir / train_file
    val_path = data_dir / val_file

    token = load_env_token()

    # Tải train.csv nếu chưa tồn tại
    if not train_path.exists():
        logger.info(f"Đang tải {train_file} từ repo '{repo_id}' về {data_dir}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=train_file,
            repo_type="dataset",
            local_dir=str(data_dir),
            token=token,
        )

    # Tải val.csv nếu chưa tồn tại
    if not val_path.exists():
        logger.info(f"Đang tải {val_file} từ repo '{repo_id}' về {data_dir}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=val_file,
            repo_type="dataset",
            local_dir=str(data_dir),
            token=token,
        )

    return train_path, val_path


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Tính toán các chỉ số đánh giá: Accuracy, Macro-F1, Macro-Precision, Macro-Recall."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def prepare_datasets(config: Dict[str, Any], tokenizer: AutoTokenizer):
    """Đọc dữ liệu từ file CSV, làm sạch và tokenize theo cấu hình."""
    data_cfg = config.get("data", {})
    tok_cfg = config.get("tokenizer", {})

    train_path, val_path = ensure_dataset(data_cfg)

    logger.info(f"Đọc dữ liệu train từ: {train_path}")
    logger.info(f"Đọc dữ liệu val từ: {val_path}")

    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_val = pd.read_csv(val_path, encoding="utf-8")

    text_col = data_cfg.get("text_column", "question")
    pair_col = data_cfg.get("text_pair_column", "context")
    label_col = data_cfg.get("label_column", "label")

    # Xử lý dropna và ép kiểu nhãn
    df_train = df_train.dropna(subset=[text_col, pair_col, label_col]).reset_index(drop=True)
    df_val = df_val.dropna(subset=[text_col, pair_col, label_col]).reset_index(drop=True)
    df_train[label_col] = df_train[label_col].astype(int)
    df_val[label_col] = df_val[label_col].astype(int)

    logger.info(f"Train dataset shape: {df_train.shape} | Phân bố nhãn: {dict(df_train[label_col].value_counts())}")
    logger.info(f"Val dataset shape: {df_val.shape} | Phân bố nhãn: {dict(df_val[label_col].value_counts())}")

    max_length = tok_cfg.get("max_length", 2048)
    truncation_strategy = tok_cfg.get("truncation", "only_second")
    padding_strategy = tok_cfg.get("padding", False)

    def preprocess_nli(examples):
        return tokenizer(
            text=examples[text_col],
            text_pair=examples[pair_col],
            max_length=max_length,
            truncation=truncation_strategy,
            padding=padding_strategy,
        )

    raw_train_ds = Dataset.from_pandas(df_train[[text_col, pair_col, label_col]])
    raw_val_ds = Dataset.from_pandas(df_val[[text_col, pair_col, label_col]])

    logger.info(f"Tokenizing datasets (max_length={max_length}, truncation='{truncation_strategy}')...")
    train_dataset = raw_train_ds.map(preprocess_nli, batched=True, remove_columns=[text_col, pair_col])
    val_dataset = raw_val_ds.map(preprocess_nli, batched=True, remove_columns=[text_col, pair_col])

    return train_dataset, val_dataset


def train(config_path: Optional[str] = None):
    """Hàm thực thi toàn bộ quy trình huấn luyện mô hình NLI trên local."""
    # 1. Nạp file cấu hình
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    model_cfg = cfg.get("model", {})
    tok_cfg = cfg.get("tokenizer", {})
    train_cfg = cfg.get("training", {})
    early_stop_cfg = cfg.get("early_stopping", {})

    logger.info("=== BẮT ĐẦU CẤU HÌNH HUẤN LUYỆN NLI (LOCAL) ===")
    logger.info(f"Seed: {seed}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"Device Count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} | VRAM: {props.total_memory / (1024**3):.2f} GB")

    # 2. Khởi tạo Tokenizer
    model_name = model_cfg.get("base_model_name", "Qualcomm-AI-Research/BamiBERT")
    logger.info(f"Nạp tokenizer từ: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Chuẩn bị Dataset (tự động tải từ Hugging Face về db/ViLegalNLI nếu cần)
    train_dataset, val_dataset = prepare_datasets(cfg, tokenizer)

    # 4. Khởi tạo Model
    num_labels = model_cfg.get("num_labels", 2)
    id2label = {int(k): v for k, v in model_cfg.get("id2label", {0: "CONTRADICTION/LOSE", 1: "ENTAILMENT/WIN"}).items()}
    label2id = {str(k): int(v) for k, v in model_cfg.get("label2id", {"CONTRADICTION/LOSE": 0, "ENTAILMENT/WIN": 1}).items()}

    logger.info(f"Nạp model sequence classification: {model_name} (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("Đã kích hoạt Gradient Checkpointing.")

    pad_to_multiple_of = tok_cfg.get("pad_to_multiple_of", 8)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    # 5. Thiết lập TrainingArguments
    raw_output_dir = train_cfg.get("output_dir", "be/src/NLIProvider/models/checkpoints")
    raw_best_dir = train_cfg.get("best_model_dir", "be/src/NLIProvider/models/bamibert_vilegalnli")

    output_dir = Path(raw_output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    best_model_dir = Path(raw_best_dir)
    if not best_model_dir.is_absolute():
        best_model_dir = ROOT_DIR / best_model_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(train_cfg.get("fp16", True)) and cuda_available

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 2)),
        num_train_epochs=int(train_cfg.get("num_train_epochs", 10)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_steps=int(train_cfg.get("warmup_steps", 100)),
        optim=train_cfg.get("optim", "adamw_torch"),
        fp16=use_fp16,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "f1"),
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),
        logging_steps=int(train_cfg.get("logging_steps", 25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        report_to=train_cfg.get("report_to", "none"),
    )

    # Callbacks
    callbacks = []
    if early_stop_cfg:
        patience = early_stop_cfg.get("patience", 2)
        threshold = early_stop_cfg.get("threshold", 0.001)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )

    # Dọn dẹp bộ nhớ GPU
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()

    # 6. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 7. Huấn luyện
    logger.info("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ===")
    train_result = trainer.train()
    logger.info(f"Kết quả huấn luyện: {train_result}")

    # 8. Lưu Best Model & Tokenizer
    logger.info(f"Lưu mô hình tốt nhất vào: {best_model_dir}")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info("Lưu mô hình hoàn tất.")

    # 9. Đánh giá tổng hợp trên Validation Set
    logger.info("=== ĐÁNH GIÁ TRÊN TẬP VALIDATION ===")
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình NLI ViLegalNLI với BamiBERT")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Đường dẫn đến file cấu hình YAML (mặc định: src/config/train.yaml)",
    )
    args = parser.parse_args()

    train(config_path=args.config)
