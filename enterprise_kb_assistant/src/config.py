import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    chroma_persist_dir: str
    chroma_collection_name: str
    app_api_key: str


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未读取到 DASHSCOPE_API_KEY，请检查 .env 文件。")

    app_api_key = os.getenv("APP_API_KEY")
    if not app_api_key:
        raise ValueError("未读取到 APP_API_KEY，请检查 .env 文件。")

    return Settings(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        chat_model=os.getenv("CHAT_MODEL", "qwen3.6-plus"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "enterprise_kb"),
        app_api_key=app_api_key,
    )