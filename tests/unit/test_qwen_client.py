import pytest
from core.llm.qwen_client import get_qwen_models


class TestQwenClient:
    def test_model_initialization(self):
        """测试模型是否能成功初始化"""
        llm, chat, embed = get_qwen_models()
        assert llm is not None
        assert chat is not None
        assert embed is not None
        return llm, chat, embed

    def test_invalid_config(self, monkeypatch):
        """测试配置错误情况"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "")
        with pytest.raises(RuntimeError):
            get_qwen_models()

