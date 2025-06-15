import pytest
from unittest.mock import MagicMock, patch
from core.knowledge_base.retrieval.vector_retriever import VectorRetriever
from langchain_core.documents import Document


@pytest.fixture
def mock_embedding():
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 768
    return mock


@pytest.fixture
def test_retriever(tmp_path, mock_embedding):
    """初始化检索器"""
    return VectorRetriever(
        persist_path=str(tmp_path / "retrieval_db"),
        embedding_function=mock_embedding,
        top_k=3
    )


class TestVectorRetriever:
    @patch("chromadb.Collection.query")
    def test_similarity_search(self, mock_query, test_retriever):
        # 模拟返回结果
        mock_query.return_value = {
            'ids': [['id1', 'id2']],
            'documents': [['doc1内容', 'doc2内容']],
            'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]]
        }

        results = test_retriever.similarity_search("混凝土强度")
        assert len(results) == 2
        assert isinstance(results[0], Document)
        mock_query.assert_called_once()

    @patch("chromadb.Collection.query")
    def test_hybrid_search(self, mock_query, test_retriever):
        mock_query.return_value = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [
                ['混凝土强度标准', '钢筋检测规范', '施工安全要求']
            ],
            'metadatas': [[{}, {}, {}]]
        }

        # 测试关键词过滤
        results = test_retriever.hybrid_search(
            query="施工标准",
            keyword="混凝土"
        )
        assert len(results) == 1
        assert "混凝土" in results[0].page_content

    @patch("chromadb.Collection.get")
    def test_get_by_id(self, mock_get, test_retriever):
        mock_get.return_value = {
            'ids': ['id1'],
            'documents': ['测试文档内容'],
            'metadatas': [{'source': 'test'}]
        }

        doc = test_retriever.get_by_id("id1")
        assert doc.page_content == "测试文档内容"
        assert doc.metadata['source'] == "test"


@pytest.mark.integration
class TestRetrievalIntegration:
    def test_real_retrieval(self, tmp_path, mock_embedding):
        """集成测试"""
        # 先创建测试数据
        from ..builders.pdf_processor import PDFProcessor
        processor = PDFProcessor(
            directory="tests/test_files",
            persist_path=str(tmp_path / "retrieval_db"),
            embedding_function=mock_embedding
        )
        processor.process_pdfs()

        # 测试检索
        retriever = VectorRetriever(
            persist_path=str(tmp_path / "retrieval_db"),
            embedding_function=mock_embedding
        )

        results = retriever.similarity_search("混凝土强度")
        assert len(results) > 0
        assert "混凝土" in results[0].page_content