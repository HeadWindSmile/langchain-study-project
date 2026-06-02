import json
from typing import Iterator

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
                "问题：{question}\n\n上下文：\n{context}"
            ),
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

    def stream_answer(self, question: str) -> Iterator[str]:
        """
        流式回答。

        SSE 事件顺序：
        1. sources：先返回检索来源
        2. token：逐步返回模型生成内容
        3. done：生成结束
        4. error：异常
        """
        try:
            retrieved_docs = self.retriever.invoke(question)
            context = self._format_context(retrieved_docs)
            sources = self._format_sources(retrieved_docs)

            yield self._sse_event("sources", {"sources": sources})

            for chunk in self.chain.stream({
                "question": question,
                "context": context,
            }):
                content = getattr(chunk, "content", None)
                if content:
                    yield self._sse_event("token", {"content": content})

            yield self._sse_event("done", {"message": "completed"})

        except Exception as e:
            yield self._sse_event("error", {"message": str(e)})

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

    @staticmethod
    def _sse_event(event: str, data: dict) -> str:
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {json_data}\n\n"