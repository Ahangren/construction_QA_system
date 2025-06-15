import pytest
from unittest.mock import patch
from core.utils.embedding_utils import load_embedding_model


class TestEmbeddingUtils:
    @patch('langchain.embeddings.HuggingFaceEmbeddings')
    def test_load_huggingface(self, mock_embeddings):
        """测试加载HuggingFace模型"""
        model = load_embedding_model(model_type="huggingface")
        mock_embeddings.assert_called_once()

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    @patch('langchain.embeddings.OpenAIEmbeddings')
    def test_load_openai(self, mock_embeddings):
        """测试加载OpenAI模型"""
        model = load_embedding_model(model_type="openai")
        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            deployment=None,
            openai_api_key="test_key"
        )

    def test_invalid_model_type(self):
        """测试无效模型类型"""
        with pytest.raises(ValueError):
            load_embedding_model(model_type="invalid_type")