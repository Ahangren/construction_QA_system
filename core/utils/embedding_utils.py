import os
from typing import Union
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings


def load_embedding_model(model_type: str = "huggingface", ** kwargs) -> Embeddings:
    """
    加载嵌入模型（支持多种后端）

    参数:
        model_type: huggingface|openai|sentence-transformers
        kwargs: 各模型特有的参数

    返回:
        初始化好的嵌入模型实例
    """
    # 环境变量配置检查
    if model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key 未配置")

    # 模型加载逻辑
    try:
        if model_type in ["huggingface", "sentence-transformers"]:
            model_name = kwargs.get(
                "model_name",
                "GanymedeNil/text2vec-large-chinese"
            )
            device = kwargs.get("device", "cpu")

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={
                    'normalize_embeddings': kwargs.get("normalize_embeddings", True)
                }
            )

        elif model_type == "openai":
            return OpenAIEmbeddings(
                model=kwargs.get("model", "text-embedding-3-small"),
                deployment=kwargs.get("deployment", None),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


def get_embedding_model_config() -> dict:
    """获取默认模型配置"""
    return {
        "huggingface": {
            "model_name": "GanymedeNil/text2vec-large-chinese",
            "device": "cpu"
        },
        "openai": {
            "model": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY"
        }
    }