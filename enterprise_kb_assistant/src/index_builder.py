from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import Settings


def build_retriever(
    docs: list[Document],
    settings: Settings,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    k: int = 3,
    force_rebuild: bool = False,
):
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        check_embedding_ctx_length=False
    )

    persist_dir = Path(settings.chroma_persist_dir)

    if force_rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    if persist_dir.exists() and any(persist_dir.iterdir()) and not force_rebuild:
        print(f"检测到已有 Chroma 索引，直接加载: {persist_dir}")

        vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

        return vector_store.as_retriever(search_kwargs={"k": k})

    print("未检测到已有索引，开始构建新的 Chroma 索引...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    split_docs = splitter.split_documents(docs)

    print(f"原始 Document 数量: {len(docs)}")
    print(f"切块后 Document 数量: {len(split_docs)}")

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        persist_directory=str(persist_dir),
    )

    print(f"Chroma 索引已持久化到: {persist_dir}")

    return vector_store.as_retriever(search_kwargs={"k": k})