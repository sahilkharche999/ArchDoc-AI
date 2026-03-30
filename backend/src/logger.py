import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime 
from zoneinfo import ZoneInfo

def setup_logger(logger_name: str) -> logging.Logger:
    """
    Returns a logger that writes to the mounted EC2 host filesystem.
    Assumes the host path is bind-mounted at /var/log/myapp inside the container.
    """
    log_dir = os.getenv("LOG_DIR", "/var/log/dax")
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.now(ZoneInfo('Asia/Kolkata'))
    now=now.strftime('%d-%m-%Y')

    log_file = os.path.join(log_dir, f"dax-{now}.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler — 10 MB per file, keep 5 backups
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Optional: also log to stdout so `docker logs` still works
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger