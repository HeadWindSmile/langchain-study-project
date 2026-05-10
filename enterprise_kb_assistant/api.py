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

# ==========================================
# 1. 基础配置与全局变量
# ==========================================

# 配置 Python 标准日志库，输出时间、日志级别和具体信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# 全局 RAG 服务实例，在服务启动时初始化，常驻内存
rag_service: RagService | None = None


# ==========================================
# 2. 统一响应体构建工具 (类似 Java 的 Result<T>)
# ==========================================

def success_response(data: Any, request_id: str) -> dict:
    """构建标准化的成功响应 JSON 结构"""
    return {
        "code": 0,  # 业务状态码 0 代表成功
        "message": "success",
        "data": data,  # 实际的业务数据载荷
        "request_id": request_id,  # 链路追踪 ID，方便查日志
    }


def error_response(code: int, message: str, request_id: str) -> dict:
    """构建标准化的错误响应 JSON 结构"""
    return {
        "code": code,  # 非 0 的业务错误码
        "message": message,  # 给前端展示的错误提示
        "data": None,  # 错误时通常无业务数据
        "request_id": request_id,
    }


# ==========================================
# 3. 生命周期管理 (应用启动/关闭时的回调)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器。
    yield 之前的代码在服务启动前执行（负责加载文档、建立向量索引）。
    yield 之后的代码在服务关闭时执行（负责资源释放）。
    """
    global rag_service

    logger.info("正在初始化 RAG 服务...")

    # 加载配置和本地文档
    settings = load_settings()
    docs = load_documents_from_dir("data/docs")

    # 构建基于 Chroma 的向量检索器 (耗时操作)
    retriever = build_retriever(
        docs=docs,
        settings=settings,
        chunk_size=500,
        chunk_overlap=80,
        k=3,
        force_rebuild=False,
    )

    # 注入检索器和配置，实例化问答服务
    rag_service = RagService(retriever, settings)
    logger.info("RAG 服务初始化完成。")

    yield  # 服务器在此处挂起，开始监听外部 HTTP 请求

    logger.info("RAG 服务关闭。")


# ==========================================
# 4. FastAPI 实例初始化
# ==========================================

app = FastAPI(
    title="Enterprise Knowledge Base Assistant",
    description="基于 LangChain + Chroma 的企业知识库问答服务",
    version="0.2.0",
    lifespan=lifespan,
)


# ==========================================
# 5. 全局中间件 (Middleware)
# ==========================================

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """
    HTTP 拦截器：所有请求进出都会经过这里。
    职责：生成请求唯一 ID (Trace ID)，记录请求出入参，统计耗时。
    """
    # 1. 为每次请求生成 UUID，并挂载到 request.state 上，方便后续环节提取
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()
    logger.info(
        "request_start request_id=%s method=%s path=%s",
        request_id, request.method, request.url.path,
    )

    try:
        # 2. 放行请求，将控制权交给具体的路由函数 (或抛出异常给下游 handler)
        response = await call_next(request)
    except Exception:
        # 如果路由内部发生严重错误（且未被 exception_handler 捕获），记录异常并向上抛出
        elapsed = time.time() - start_time
        logger.exception(
            "request_error request_id=%s elapsed=%.3fs",
            request_id, elapsed,
        )
        raise

    # 3. 记录请求处理完成的耗时和状态码
    elapsed = time.time() - start_time
    logger.info(
        "request_end request_id=%s status_code=%s elapsed=%.3fs",
        request_id, response.status_code, elapsed,
    )

    # 4. 将 Request ID 塞进 HTTP 响应头中，方便前端排查问题
    response.headers["X-Request-ID"] = request_id
    return response


# ==========================================
# 6. 全局异常处理 (Exception Handlers)
# ==========================================

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """捕获自定义的业务异常 (如：余额不足、无权限等)"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning("app_exception request_id=%s code=%s message=%s", request_id, exc.code, exc.message)

    # 统一转换为 HTTP 400，并返回我们定义的标准错误 JSON
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response(exc.code, exc.message, request_id),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """捕获 FastAPI 原生的 HTTP 异常 (如我们写在鉴权依赖里的 raise HTTPException)"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning("http_exception request_id=%s status_code=%s detail=%s", request_id, exc.status_code, exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.status_code, str(exc.detail), request_id),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """捕获 Pydantic 数据校验异常 (例如前端漏传了 question 字段)"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning("validation_exception request_id=%s errors=%s", request_id, exc.errors())

    # 转换为 422 状态码 (Unprocessable Entity)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response(422, "请求参数校验失败", request_id),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底拦截所有未预料到的系统异常 (防止堆栈信息泄露给前端)"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception("unhandled_exception request_id=%s error=%s", request_id, str(exc))

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response(500, "服务内部错误", request_id),
    )


# ==========================================
# 7. 业务路由接口 (Endpoints)
# ==========================================

@app.get("/health")
def health_check(request: Request):
    """探针接口：用于 K8s/Docker 等容器检查服务存活状态"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return success_response({"status": "ok"}, request_id)


@app.post(
    "/ask",
    response_model=ApiResponse[AskData],  # 自动生成正确的 Swagger API 文档
    dependencies=[Depends(verify_api_key)],  # 路由级别的依赖注入，先执行 API Key 校验逻辑
)
def ask(request: Request, body: AskRequest):
    """核心问答接口：接收前端提问，调用大模型生成答案并返回"""

    # 从上下文中提取 Request ID，保证业务日志能和外部请求串联
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # 防御性编程：确保服务可用
    if rag_service is None:
        raise AppException(code=50001, message="RAG 服务尚未初始化完成")

    logger.info("ask_start request_id=%s question=%s", request_id, body.question)
    start_time = time.time()

    # ---> 核心调用点：执行 LangChain 的检索和生成逻辑 <---
    result = rag_service.answer(body.question)

    elapsed = time.time() - start_time
    logger.info(
        "ask_end request_id=%s elapsed=%.3fs sources_count=%s",
        request_id, elapsed, len(result.get("sources", [])),
    )

    # 将底层返回的字典，映射到严格类型的 Pydantic 响应模型中
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

    # 返回最终的标准格式 JSON
    return success_response(data.model_dump(), request_id)