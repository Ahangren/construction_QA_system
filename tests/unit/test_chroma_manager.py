import pytest
from unittest.mock import MagicMock
from knowledge_base.storage.chroma_manager import ChromaManager

class TestChromaManager:
    @pytest.fixture
    def mock_embedding(self):
        mock = MagicMock()
        mock.embed_documents.return_value = [[0.1]*768]
        return mock

    def test_local_init(self, tmp_path, mock_embedding):
        """测试本地模式初始化"""
        db = ChromaManager(
            chroma_server_type="local",
            persist_path=str(tmp_path),
            embed_model=mock_embedding
        )
        assert db.store is not None

    def test_add_documents(self, tmp_path, mock_embedding):
        """测试文档添加功能"""
        db = ChromaManager(
            chroma_server_type="local",
            persist_path=str(tmp_path),
            embed_model=mock_embedding
        )
        test_docs = ["Test document 1", "Test document 2"]
        doc_ids = db.add_documents(test_docs)
        assert len(doc_ids) == 2