# 负责检索和生成答案

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import Settings


class RagService:
    def __init__(self, retriever, settings: Settings):
        self.retriever = retriever

        self.model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0,
            timeout=30.0,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个企业知识库问答助手。"
                "请严格基于提供的上下文回答问题。"
                "如果上下文中没有答案，请明确说：根据当前知识库资料无法确定。"
            ),
            (
                "human",
                "问题：{question}\n\n"
                "上下文：\n{context}"
            )
        ])

        self.chain = self.prompt | self.model

    def answer(self, question: str) -> dict:
        retrieved_docs = self.retriever.invoke(question)

        context = self._format_context(retrieved_docs)

        response = self.chain.invoke({
            "question": question,
            "context": context,
        })

        return {
            "answer": response.content,
            "sources": self._format_sources(retrieved_docs),
        }

    @staticmethod
    def _format_context(docs) -> str:
        if not docs:
            return "没有检索到相关上下文。"

        return "\n\n".join(
            f"[文档{i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    @staticmethod
    def _format_sources(docs) -> list[dict]:
        return [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:120],
            }
            for doc in docs
        ]