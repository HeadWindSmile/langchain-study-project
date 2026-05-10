# 程序入口

from src.config import load_settings
from src.document_loader import load_documents_from_dir
from src.index_builder import build_retriever
from src.rag_service import RagService


def main():
    settings = load_settings()

    docs = load_documents_from_dir("data/docs")

    retriever = build_retriever(
        docs=docs,
        settings=settings,
        chunk_size=500,
        chunk_overlap=80,
        k=3,
        force_rebuild=False,
    )

    rag_service = RagService(retriever, settings)

    questions = [
        "RAG 的核心流程是什么？",
        "LangChain 的 Agent 是什么？",
        "企业知识库助手为什么需要 RAG？",
    ]

    for question in questions:
        result = rag_service.answer(question)

        print("\n" + "=" * 60)
        print("用户问题:")
        print(question)

        print("\n答案:")
        print(result["answer"])

        print("\n来源:")
        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    main()