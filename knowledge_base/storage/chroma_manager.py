import uuid
from typing import Optional, List, Union
import chromadb
from chromadb import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import logging


class ChromaManager:
    """ChromaDB向量数据库的高级封装管理类

    特性：
    - 支持本地和HTTP两种连接模式
    - 自动持久化管理
    - 线程安全连接
    - 完善的错误处理
    """

    def __init__(self,
                 chroma_server_type: str = "local",
                 host: str = "localhost",
                 port: int = 8000,
                 persist_path: str = "chroma_db",
                 collection_name: str = "langchain",
                 embed_model: Optional[Embeddings] = None):
        """
        初始化ChromaDB连接

        Args:
            chroma_server_type: 连接类型 ("local"|"http")
            host: 服务器地址 (HTTP模式必需)
            port: 服务器端口 (HTTP模式必需)
            persist_path: 本地持久化路径 (本地模式必需)
            collection_name: 集合名称
            embed_model: 嵌入模型实例
        """
        self._validate_init_params(chroma_server_type, host, port, persist_path)

        self.client = self._create_client(chroma_server_type, host, port, persist_path)
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.logger = logging.getLogger(__name__)

        try:
            self.store = Chroma(
                collection_name=collection_name,
                embedding_function=embed_model,
                client=self.client,
                persist_directory=persist_path if chroma_server_type == "local" else None
            )
            self.logger.info(f"ChromaDB initialized successfully. Mode: {chroma_server_type}")
        except Exception as e:
            self.logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")

    def _validate_init_params(self, server_type: str, host: str, port: int, path: str):
        """参数验证"""
        if server_type not in ["local", "http"]:
            raise ValueError(f"Invalid server type: {server_type}. Must be 'local' or 'http'")

        if server_type == "http" and not all([host, port]):
            raise ValueError("Host and port must be specified for HTTP mode")

        if server_type == "local" and not path:
            raise ValueError("Persist path must be specified for local mode")

    def _create_client(self, server_type: str, host: str, port: int, path: str) -> chromadb.Client:
        """创建Chroma客户端"""
        try:
            if server_type == "http":
                return chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(allow_reset=True)
                )
            else:
                return chromadb.PersistentClient(
                    path=path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
        except Exception as e:
            logging.error(f"Chroma client creation failed: {str(e)}")
            raise

    def add_documents(self, docs: Union[List[Document], List[str]]) -> List[str]:
        """
        添加文档到集合

        Args:
            docs: 文档列表，可以是Document对象或纯文本

        Returns:
            插入文档的ID列表
        """
        try:
            if not docs:
                self.logger.warning("Attempted to add empty documents list")
                return []
            processed_docs = self._prepare_documents(docs)
            doc_ids = self.store.add_documents(documents=processed_docs)
            self.logger.info(f"Added {len(doc_ids)} documents to collection '{self.collection_name}'")
            return doc_ids
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise RuntimeError(f"Document addition failed: {str(e)}")

    def _prepare_documents(self, docs: Union[List[Document], List[str], List[dict]]) -> List[Document]:
        """将输入转换为Document对象列表"""
        if not docs:
            return []

        # 如果已经是Document对象，直接返回
        if isinstance(docs[0], Document):
            return docs

        # 处理字符串输入
        if isinstance(docs[0], str):
            return [Document(
                page_content=text,
                metadata={"source": "user_input", "id": str(uuid.uuid4())}
            ) for text in docs]

        # 处理字典输入
        if isinstance(docs[0], dict):
            return [Document(
                page_content=doc["page_content"],
                metadata=doc.get("metadata", {})
            ) for doc in docs]

        raise ValueError(f"Unsupported document type: {type(docs[0])}")

    def query(self, query_text: str, k: int = 5,  ** kwargs) -> List[Document]:
        """
        相似性查询

        Args:
            query_text: 查询文本
            k: 返回结果数量
            **kwargs: 额外查询参数

        Returns:
            匹配的文档列表
        """
        try:
            results = self.store.similarity_search(query_text, k=k,  ** kwargs)
            self.logger.debug(f"Query returned {len(results)} results for: {query_text}")
            return results
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise RuntimeError(f"Query operation failed: {str(e)}")

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            raise RuntimeError(f"Collection stats retrieval failed: {str(e)}")

    @property
    def store(self) -> Chroma:
        """获取LangChain Chroma实例"""
        return self._store

    @store.setter
    def store(self, value):
        self._store = value