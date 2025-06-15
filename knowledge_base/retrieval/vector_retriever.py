from typing import List, Dict, Optional
from langchain_core.documents import Document
from ..storage.chroma_manager import ChromaManager


class VectorRetriever:
    def __init__(
            self,
            chroma_server_type: str = "local",
            persist_path: str = "chroma_db",
            collection_name: str = "construction_docs",
            embedding_function: Optional[object] = None,
            top_k: int = 5
    ):
        """向量检索器

        Args:
            top_k: 返回最相关的K个结果
            score_threshold: 相似度阈值
        """
        self.chroma_db = ChromaManager(
            chroma_server_type=chroma_server_type,
            persist_path=persist_path,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        self.top_k = top_k

    def similarity_search(
            self,
            query: str,
            filter_conditions: Optional[Dict] = None
    ) -> List[Document]:
        """相似度搜索"""
        collection = self.chroma_db.collection

        # 获取查询向量（假设embedding_function已配置）
        query_embedding = self.chroma_db.embedding_function.embed_query(query)

        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            where=filter_conditions
        )

        # 转换为Document对象
        docs = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i] or {}
            )
            docs.append(doc)

        return docs

    def hybrid_search(
            self,
            query: str,
            keyword: Optional[str] = None,
            filter_conditions: Optional[Dict] = None
    ) -> List[Document]:
        """混合检索（向量+关键词）"""
        # 先执行向量搜索
        vector_results = self.similarity_search(query, filter_conditions)

        # 如果有关键词，进行过滤
        if keyword:
            filtered = [
                doc for doc in vector_results
                if keyword.lower() in doc.page_content.lower()
            ]
            return filtered[:self.top_k]

        return vector_results

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """根据ID获取文档"""
        result = self.chroma_db.collection.get(ids=[doc_id])
        if not result['documents']:
            return None

        return Document(
            page_content=result['documents'][0],
            metadata=result['metadatas'][0] or {}
        )