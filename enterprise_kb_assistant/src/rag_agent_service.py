# RAG Agent 服务
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.config import Settings


class RagAgentService:
    def __init__(self, retriever, settings: Settings):
        self.retriever = retriever

        self.model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0,
            timeout=30.0,
        )

        self.retrieve_knowledge = self._build_retrieve_tool()

        self.agent = create_agent(
            model=self.model,
            tools=[self.retrieve_knowledge],
            system_prompt=(
                "你是一个企业知识库问答助手。"
                "如果用户问题涉及企业文档、知识库、LangChain、RAG、agent 等资料内容，"
                "你必须优先调用 retrieve_knowledge 工具检索资料，再基于工具结果回答。"
                "如果工具结果中没有答案，请明确说明：根据当前知识库资料无法确定。"
                "如果用户只是寒暄或问你能力范围，可以直接回答，不需要调用工具。"
                "回答要简洁、准确，并尽量说明依据。"
            ),
            debug=True,
        )

    def _build_retrieve_tool(self):
        retriever = self.retriever

        @tool
        def retrieve_knowledge(query: str) -> str:
            """当用户问题需要企业知识库资料时，调用此工具检索相关文档内容。"""
            retrieved_docs = retriever.invoke(query)

            if not retrieved_docs:
                return "没有检索到相关知识库内容。"

            context_parts = []
            for i, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page")

                source_info = f"source={source}"
                if page is not None:
                    source_info += f", page={page}"

                context_parts.append(
                    f"[文档{i} | {source_info}]\n{doc.page_content}"
                )

            return "\n\n".join(context_parts)

        return retrieve_knowledge

    def answer(self, question: str) -> str:
        result = self.agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        messages = result.get("messages", [])
        if not messages:
            return "没有获得有效回答。"

        return messages[-1].content