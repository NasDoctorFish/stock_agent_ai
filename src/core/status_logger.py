import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class StatusLogger:
    def __init__(self, name: str = "system", log_dir: str = "logs", backup_days: int = 60):
        self.log_dir = Path(log_dir)
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            Path(log_dir).mkdir(exist_ok=True)

            # 로그 파일 이름
            log_file = self.log_dir / f"{name.lower()}.log"

            # ✅ 공통 포매터 (콘솔 & 파일 둘 다 적용)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s | %(name)s: %(message)s",
                "%Y-%m-%d %H:%M:%S"
            )

            # ---------------------------
            # 🖥 콘솔 핸들러
            # ---------------------------
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)

            # ---------------------------
            # 💾 파일 핸들러 (매일 회전)
            # ---------------------------
            file_handler = TimedRotatingFileHandler(
                filename=log_file,
                when="midnight",
                interval=1,
                backupCount=backup_days,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)  # ✅ 여기가 핵심!
            file_handler.setLevel(logging.INFO)

            # 핸들러 추가
            self.logger.addHandler(stream_handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)


'''
사용예시

from core.status_logger import StatusLogger

class DBHandler:
    def __init__(self):
        self.logger = StatusLogger("DBHandler")
        self.logger.info("Connected to DuckDB")

    def execute(self, query):
        try:
            self.logger.info(f"Executing query: {query[:50]}...")
        except Exception as e:
            self.logger.error(f"Query failed: {e}")

'''