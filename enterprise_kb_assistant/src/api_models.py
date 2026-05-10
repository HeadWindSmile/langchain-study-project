from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = Field(..., description="业务状态码，0 表示成功")
    message: str = Field(..., description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    request_id: str = Field(..., description="请求 ID")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


class SourceItem(BaseModel):
    source: Optional[str] = Field(None, description="来源文件")
    page: Optional[int] = Field(None, description="页码")
    snippet: str = Field("", description="命中的文本片段")


class AskData(BaseModel):
    answer: str = Field(..., description="回答")
    sources: list[SourceItem] = Field(default_factory=list, description="来源列表")