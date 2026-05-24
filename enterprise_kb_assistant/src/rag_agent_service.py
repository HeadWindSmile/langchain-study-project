from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from src.config import Settings


class RagAgentService:
    """基于 LangChain Agent 的 RAG 问答服务。

    和普通 RAG 不同，Agent 会先读用户问题，再自行决定是否调用检索工具。
    如果问题只是寒暄或能力询问，Agent 可以不查知识库直接回答。
    """

    def __init__(self, retriever, settings: Settings):
        # retriever 是外部已经构建好的向量检索器，负责从知识库中找相关文档。
        self.retriever = retriever
        self.settings = settings

        # ChatOpenAI 这里只是 LangChain 的 OpenAI 兼容模型封装。
        # 具体模型名、API Key、base_url 都从配置文件读取，方便切换不同服务商。
        self.model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            # temperature=0 让回答尽量稳定，适合知识库问答场景。
            temperature=0,
            timeout=30.0,
        )

    def answer(self, question: str) -> dict:
        """
        使用 RAG Agent 回答问题。

        注意：
        这里每次请求创建一个带局部 sources 的 tool，
        是为了避免多请求并发时 sources 串数据。
        """
        # 用于收集本次请求实际检索到的来源，最后会返回给 API 层。
        # 它定义在 answer 方法内部，因此每个请求都有独立列表。
        retrieved_sources: list[dict] = []

        # 闭包里直接访问 self 有时不够直观，这里先绑定一个局部变量给工具函数使用。
        retriever = self.retriever

        @tool
        def retrieve_knowledge(query: str) -> str:
            """
            当用户问题需要基于企业知识库、内部文档、LangChain、RAG、Agent 等资料回答时，
            调用此工具检索相关文档内容。
            """
            # Agent 决定调用工具后，会把它认为适合检索的 query 传进来。
            # retriever.invoke 会返回最相似的若干个 Document。
            docs = retriever.invoke(query)

            if not docs:
                return "没有检索到相关知识库内容。"

            # context_parts 会被拼成一段文本返回给大模型，让大模型基于这些内容生成答案。
            context_parts = []

            for i, doc in enumerate(docs, start=1):
                # metadata 是文档加载或切分时保存的附加信息，例如来源文件和页码。
                source = doc.metadata.get("source")
                page = doc.metadata.get("page")
                # snippet 只截取前 120 个字符，用于接口返回时展示来源摘要。
                snippet = doc.page_content[:120]

                retrieved_sources.append({
                    "source": source,
                    "page": page,
                    "snippet": snippet,
                })

                source_info = f"source={source}"
                if page is not None:
                    source_info += f", page={page}"

                # 给每段上下文加编号和来源信息，方便模型在回答时理解证据来自哪里。
                context_parts.append(
                    f"[文档{i} | {source_info}]\n{doc.page_content}"
                )

            return "\n\n".join(context_parts)

        # 每次 answer 调用都创建一个 Agent，并把本次请求专属的 retrieve_knowledge 工具交给它。
        # 这样可以让工具函数安全地把 sources 写入本次请求的 retrieved_sources。
        agent = create_agent(
            model=self.model,
            tools=[retrieve_knowledge],
            system_prompt=(
                "你是一个企业知识库问答助手。"
                "如果用户问题涉及企业文档、知识库、LangChain、RAG、Agent、项目资料等内容，"
                "你必须优先调用 retrieve_knowledge 工具检索资料，再基于工具结果回答。"
                "如果工具结果中没有答案，请明确说明：根据当前知识库资料无法确定。"
                "如果用户只是寒暄、问候、闲聊，或者询问你的能力范围，可以直接回答，不需要调用工具。"
                "回答要简洁、准确，并尽量说明依据。"
            ),
        )

        # LangChain Agent 使用 messages 格式接收对话，这里只传入当前用户问题。
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        # Agent 的返回结果里通常包含完整消息历史，最后一条就是最终回答。
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "没有获得有效回答。"

        # 统一返回 answer + sources，方便 api.py 转成标准响应模型。
        return {
            "answer": answer,
            "sources": retrieved_sources,
        }
