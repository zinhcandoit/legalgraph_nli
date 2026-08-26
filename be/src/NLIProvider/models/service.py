import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaTokenizerFast

from ..logger import logger

# Định nghĩa đường dẫn gốc ROOT_DIR (ML_final)
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[4]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Nạp cấu hình từ be/src/config/config.py
try:
    from be.src.config.config import config
    MODEL_NAME = config.models.nli
except ImportError:
    try:
        from config.config import config
        MODEL_NAME = config.models.nli
    except ImportError:
        MODEL_NAME = "TQZinh/BamiBERT-ViLegalNLI"

# Thư mục lưu model cục bộ
LOCAL_MODEL_DIR = CURRENT_DIR / "bamibert_vilegalnli"

# Định nghĩa nhãn kết quả
LABEL_MAPPING = {
    0: "CONTRADICTION/LOSE",
    1: "ENTAILMENT/WIN",
}


def load_env_token() -> Optional[str]:
    """Tìm token Hugging Face trong .env tại ROOT_DIR."""
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


class NLIEngine:
    """Class quản lý tải và chạy dự đoán mô hình NLI."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or LOCAL_MODEL_DIR
        self.model_name = MODEL_NAME

        # 1. Đảm bảo mô hình đã tồn tại cục bộ (nếu chưa có thì tải từ Hugging Face)
        self._ensure_model_exists()

        # 2. Nạp tokenizer và mô hình
        logger.info(f"Nạp mô hình từ: {self.model_dir} (Thiết bị: {self.device})")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Mô hình NLI đã sẵn sàng phục vụ.")

    def _ensure_model_exists(self):
        """Kiểm tra và tự động tải mô hình từ Hugging Face nếu chưa có sẵn."""
        has_config = (self.model_dir / "config.json").exists()
        has_weights = (self.model_dir / "model.safetensors").exists() or (self.model_dir / "pytorch_model.bin").exists()

        if has_config and has_weights:
            return

        logger.info(f"Chưa tìm thấy mô hình tại '{self.model_dir}'. Đang tải '{self.model_name}' từ Hugging Face...")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        token = load_env_token()
        snapshot_download(
            repo_id=self.model_name,
            local_dir=str(self.model_dir),
            token=token,
        )
        logger.info(f"Tải mô hình thành công về: {self.model_dir}")

    def predict(self, question: str, context: str) -> Dict[str, Union[str, int, float, Dict[str, float]]]:
        """Dự đoán quan hệ giữa câu hỏi/nhận định và ngữ cảnh luật."""
        inputs = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = int(torch.argmax(probabilities).item())

        prob_contradiction = float(probabilities[0].item())
        prob_entailment = float(probabilities[1].item())

        return {
            "label_id": pred_id,
            "label": LABEL_MAPPING.get(pred_id, "UNKNOWN"),
            "confidence": round(float(probabilities[pred_id].item()), 4),
            "probabilities": {
                LABEL_MAPPING[0]: round(prob_contradiction, 4),
                LABEL_MAPPING[1]: round(prob_entailment, 4),
            },
        }

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Union[str, int, float, Dict[str, float]]]]:
        """Dự đoán theo danh sách (batch) nhiều cặp câu."""
        results = []
        for question, context in pairs:
            results.append(self.predict(question, context))
        return results


# Khởi tạo singleton dùng chung
_engine: Optional[NLIEngine] = None


def get_nli_engine() -> NLIEngine:
    global _engine
    if _engine is None:
        _engine = NLIEngine()
    return _engine
