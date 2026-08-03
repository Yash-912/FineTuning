from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_endpoint: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    threshold: float = 0.85
    mode: str = "soft_flag"
    model_path: str = "models/qwen-injection-detector/best"
    max_length: int = 512
    host: str = "0.0.0.0"
    port: int = 8080

    shadow_enabled: bool = False
    shadow_model_path: str = ""
    review_queue_path: str = "data/review_queue.db"
    review_near_threshold: float = 0.1

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
