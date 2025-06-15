import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Construction QA Retrieval API",
    description="建筑工程知识库检索接口",
    version="1.0.0",
    openapi_tags=[{
        "name": "检索",
        "description": "知识库检索相关接口"
    }]
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 数据模型 ---
class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    source: Optional[str] = Field(None, example="GB/T 50081-2019")
    page: Optional[int] = Field(None, example=12)
    timestamp: Optional[datetime] = Field(None, example="2023-01-01T00:00:00")


class DocumentResponse(BaseModel):
    """检索结果模型"""
    id: str = Field(..., example="doc_123")
    content: str = Field(..., example="混凝土强度检测标准...")
    metadata: DocumentMetadata
    score: float = Field(..., ge=0, le=1, example=0.85)


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., min_length=1, example="混凝土强度标准")
    top_k: Optional[int] = Field(5, gt=0, le=20, example=3)
    keyword_filter: Optional[str] = Field(None, example="钢筋")
    metadata_filter: Optional[dict] = Field(None, example={"source": "GB"})


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., example="OK")
    version: str = Field(..., example="1.0.0")


# --- 核心逻辑 ---
def initialize_retriever():
    """初始化检索器（实际项目应使用依赖注入）"""
    from knowledge_base.retrieval.vector_retriever import VectorRetriever
    from core.utils.embedding_utils import load_embedding_model

    try:
        return VectorRetriever(
            persist_path="data/vector_db",
            embedding_function=load_embedding_model(),
            top_k=10
        )
    except Exception as e:
        logger.error(f"检索器初始化失败: {str(e)}")
        raise


retriever = initialize_retriever()


# --- API端点 ---
@app.get("/health", response_model=HealthCheckResponse, tags=["系统"])
async def health_check():
    """服务健康检查"""
    return {
        "status": "OK",
        "version": "1.0.0"
    }


@app.post("/search",
          response_model=List[DocumentResponse],
          tags=["检索"],
          summary="文档检索",
          responses={
              200: {"description": "成功返回检索结果"},
              400: {"description": "无效请求参数"},
              500: {"description": "服务器内部错误"}
          })
async def search_documents(request: QueryRequest):
    """
    执行文档检索，支持以下方式：
    - 纯向量检索
    - 关键词过滤检索
    - 元数据过滤检索
    """
    try:
        logger.info(f"收到检索请求: {request.dict()}")

        # 参数验证
        if len(request.query) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本过长（最大500字符）"
            )

        # 执行检索
        if request.keyword_filter or request.metadata_filter:
            docs = retriever.hybrid_search(
                query=request.query,
                keyword=request.keyword_filter,
                filter_conditions=request.metadata_filter
            )
        else:
            docs = retriever.similarity_search(
                query=request.query,
                filter_conditions=request.metadata_filter
            )

        # 格式化结果
        results = []
        for doc in docs[:request.top_k]:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}

            results.append({
                "id": str(hash(doc.page_content)),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0)
            })

        logger.info(f"返回 {len(results)} 条结果")
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="检索服务暂时不可用"
        )


@app.get("/document/{doc_id}",
         response_model=DocumentResponse,
         tags=["检索"],
         summary="按ID获取文档")
async def get_document(doc_id: str):
    """通过文档ID获取完整内容"""
    try:
        doc = retriever.get_by_id(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文档不存在"
            )

        return {
            "id": doc_id,
            "content": doc.page_content,
            "metadata": doc.metadata or {},
            "score": 1.0
        }
    except Exception as e:
        logger.error(f"获取文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档获取失败"
        )


# --- 启动配置 ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "default"
                }
            },
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO"
            }
        }
    )