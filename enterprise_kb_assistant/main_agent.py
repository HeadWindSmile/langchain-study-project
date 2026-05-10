# RAG Agent 入口
from src.config import load_settings
from src.document_loader import load_documents
from src.index_builder import build_retriever
from src.rag_agent_service import RagAgentService


def main():
    settings = load_settings()

    docs = load_documents("data/docs/langchain_intro.txt")
    retriever = build_retriever(docs, settings)

    rag_agent_service = RagAgentService(retriever, settings)

    questions = [
        "你好，你能做什么？",
        "LangChain 的 agent 是什么？",
        "RAG 的核心流程是什么？",
    ]

    for question in questions:
        print("\n=== 用户问题 ===")
        print(question)

        answer = rag_agent_service.answer(question)

        print("\n=== Agent 回答 ===")
        print(answer)


if __name__ == "__main__":
    main()