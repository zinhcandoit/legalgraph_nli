import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Thư mục gốc ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Thư mục log đặt tại be/src/logs
LOG_DIR = ROOT_DIR / "be" / "src" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_provider.log"

logger = logging.getLogger("rag_provider")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Handler xoay vòng file (Max 10MB, giữ lại 5 files backup)
    file_handler = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
