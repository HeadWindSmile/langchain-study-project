import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api_models import ApiResponse, AskData, AskRequest, SourceItem
from src.auth import verify_api_key
from src.config import load_settings
from src.document_loader import load_documents_from_dir
from src.exceptions import AppException
from src.index_builder import build_retriever
from src.rag_service import RagService
from src.rag_agent_service import RagAgentService


# 配置全局日志格式，后续中间件、异常处理和业务接口都会用同一个 logger 输出链路日志。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

# 这两个服务会在 FastAPI 启动阶段初始化，并在进程生命周期内复用。
# rag_service：传统 RAG，固定“检索 -> 生成”的流程。
# rag_agent_service：Agentic RAG，由大模型判断是否调用检索工具。
rag_service: RagService | None = None
rag_agent_service: RagAgentService | None = None


def success_response(data: Any, request_id: str) -> dict:
    """构造统一成功响应，保持所有接口返回结构一致。"""
    return {
        "code": 0,
        "message": "success",
        "data": data,
        "request_id": request_id,
    }


def error_response(code: int, message: str, request_id: str) -> dict:
    """构造统一错误响应，异常处理器会统一调用它。"""
    return {
        "code": code,
        "message": message,
        "data": None,
        "request_id": request_id,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子。

    yield 之前：应用启动时执行，用来加载配置、读取文档、构建向量检索器和服务对象。
    yield 之后：应用关闭时执行，可以放资源释放逻辑。
    """
    global rag_service, rag_agent_service

    logger.info("正在初始化 RAG 服务...")

    # 读取 .env / 环境变量中的模型配置、向量库配置等。
    settings = load_settings()
    # 从本地 data/docs 目录加载企业知识库文档。
    docs = load_documents_from_dir("data/docs")

    # 构建检索器：内部会进行文档切分、向量化，并连接或创建 Chroma 向量库。
    retriever = build_retriever(
        docs=docs,
        settings=settings,
        chunk_size=500,
        chunk_overlap=80,
        k=3,
        force_rebuild=False,
    )

    # 两种问答服务共用同一个 retriever，避免重复构建索引和重复占用资源。
    rag_service = RagService(retriever, settings)
    rag_agent_service = RagAgentService(retriever, settings)

    logger.info("RAG 服务初始化完成。")

    # yield 之后 FastAPI 才开始正常接收请求。
    yield

    logger.info("RAG 服务关闭。")


# 创建 FastAPI 应用，并把 lifespan 绑定进去，让启动时自动初始化 RAG 服务。
app = FastAPI(
    title="Enterprise Knowledge Base Assistant",
    description="基于 LangChain + Chroma 的企业知识库问答服务",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """统一请求中间件。

    每个 HTTP 请求都会先进入这里：生成 request_id、记录开始日志、调用实际路由、
    记录结束日志，并把 request_id 放到响应头，方便前端和后端日志对齐排查。
    """
    # 给每个请求生成唯一 ID，并挂到 request.state，后续路由和异常处理器都能读取。
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    # 请求进入时记录方法和路径。
    logger.info(
        "request_start request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        # 放行到下一个中间件或真正的路由函数。
        response = await call_next(request)
    except Exception:
        # 如果请求处理过程中抛出异常，这里记录耗时和堆栈，然后继续交给异常处理器。
        elapsed = time.time() - start_time
        logger.exception(
            "request_error request_id=%s elapsed=%.3fs",
            request_id,
            elapsed,
        )
        raise

    elapsed = time.time() - start_time

    # 请求结束时记录状态码和总耗时。
    logger.info(
        "request_end request_id=%s status_code=%s elapsed=%.3fs",
        request_id,
        response.status_code,
        elapsed,
    )

    # 把 request_id 返回给调用方，调用方可以用它快速定位服务端日志。
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """处理项目自定义业务异常，例如服务未初始化等可预期错误。"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.warning(
        "app_exception request_id=%s code=%s message=%s",
        request_id,
        exc.code,
        exc.message,
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response(exc.code, exc.message, request_id),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理 FastAPI/Starlette 抛出的 HTTP 异常，例如鉴权失败。"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.warning(
        "http_exception request_id=%s status_code=%s detail=%s",
        request_id,
        exc.status_code,
        exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.status_code, str(exc.detail), request_id),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求体或参数校验失败，例如缺少 question 字段。"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.warning(
        "validation_exception request_id=%s errors=%s",
        request_id,
        exc.errors(),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response(
            422,
            "请求参数校验失败",
            request_id,
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理，避免内部堆栈直接暴露给调用方。"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.exception(
        "unhandled_exception request_id=%s error=%s",
        request_id,
        str(exc),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response(
            500,
            "服务内部错误",
            request_id,
        ),
    )


@app.get("/health")
def health_check(request: Request):
    """健康检查接口，通常给容器、负载均衡或人工排查使用。"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return success_response({"status": "ok"}, request_id)


@app.post(
    "/ask",
    response_model=ApiResponse[AskData],
    dependencies=[Depends(verify_api_key)],
)
def ask(request: Request, body: AskRequest):
    """
    classic RAG：
    固定流程：先检索，再生成。
    """
    # 从中间件写入的 request.state 中取链路 ID，兜底情况下重新生成一个。
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # 启动初始化失败或尚未完成时，避免调用 None.answer 导致不可读的系统异常。
    if rag_service is None:
        raise AppException(code=50001, message="RAG 服务尚未初始化完成")

    logger.info(
        "classic_rag_start request_id=%s question=%s",
        request_id,
        body.question,
    )

    start_time = time.time()

    # 传统 RAG 的核心调用：服务内部会先检索知识库，再把上下文交给模型生成答案。
    result = rag_service.answer(body.question)

    elapsed = time.time() - start_time

    logger.info(
        "classic_rag_end request_id=%s elapsed=%.3fs sources_count=%s",
        request_id,
        elapsed,
        len(result.get("sources", [])),
    )

    # 将底层 dict 转为 Pydantic 模型，再 dump 成统一响应里的 data 字段。
    # 这样可以保证 sources 中的字段结构和 OpenAPI 文档一致。
    data = AskData(
        answer=result["answer"],
        sources=[
            SourceItem(
                source=item.get("source"),
                page=item.get("page"),
                snippet=item.get("snippet", ""),
            )
            for item in result.get("sources", [])
        ],
    )

    # 最终返回统一响应结构：code/message/data/request_id。
    return success_response(data.model_dump(), request_id)


@app.post(
    "/ask-agent",
    response_model=ApiResponse[AskData],
    dependencies=[Depends(verify_api_key)],
)
def ask_agent(request: Request, body: AskRequest):
    """
    agentic RAG：
    由 Agent 判断是否需要调用知识库检索工具。
    """
    # 和 /ask 一样，所有业务日志都使用同一个 request_id 串起来。
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Agent 服务同样在应用启动时初始化，这里做防御性检查。
    if rag_agent_service is None:
        raise AppException(code=50002, message="RAG Agent 服务尚未初始化完成")

    logger.info(
        "agentic_rag_start request_id=%s question=%s",
        request_id,
        body.question,
    )

    start_time = time.time()

    # Agentic RAG 的核心调用：Agent 会根据问题决定是否调用 retrieve_knowledge 工具。
    result = rag_agent_service.answer(body.question)

    elapsed = time.time() - start_time

    logger.info(
        "agentic_rag_end request_id=%s elapsed=%.3fs sources_count=%s",
        request_id,
        elapsed,
        len(result.get("sources", [])),
    )

    # 将 Agent 服务返回的答案和引用来源转换成接口层响应模型。
    data = AskData(
        answer=result["answer"],
        sources=[
            SourceItem(
                source=item.get("source"),
                page=item.get("page"),
                snippet=item.get("snippet", ""),
            )
            for item in result.get("sources", [])
        ],
    )

    # 返回格式和 /ask 保持一致，前端可以用同一套解析逻辑处理两种 RAG。
    return success_response(data.model_dump(), request_id)
