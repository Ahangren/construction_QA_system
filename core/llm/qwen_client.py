from dotenv import load_dotenv
import os
from typing import Tuple
from langchain_community.llms.tongyi import Tongyi
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qwen_config() -> bool:
    """
    加载千问环境变量配置
    Returns:
        bool: 是否加载成功
    """
    try:
        current_dir = os.path.dirname(__file__)
        conf_file_path_qwen = os.path.join(current_dir, '..', 'conf', '.qwen')

        if not os.path.exists(conf_file_path_qwen):
            logger.error(f"Qwen config file not found at: {conf_file_path_qwen}")
            return False

        load_dotenv(dotenv_path=conf_file_path_qwen)
        return True
    except Exception as e:
        logger.exception("Failed to load Qwen configuration")
        return False


def get_qwen_models() -> Tuple[Tongyi, ChatTongyi, DashScopeEmbeddings]:
    """
    初始化并返回千问系列大模型组件

    Returns:
        Tuple: (llm, chat, embed) 三元组
    Raises:
        RuntimeError: 当配置加载失败或初始化失败时抛出
    """
    if not load_qwen_config():
        raise RuntimeError("Qwen configuration loading failed")

    try:
        # 初始化LLM
        llm = Tongyi(
            model="qwen-max",
            temperature=0.1,
            top_p=0.7,
            max_tokens=1024,
            verbose=True
        )

        # 初始化Chat模型
        chat = ChatTongyi(
            model="qwen-max",
            temperature=0.01,
            top_p=0.2,
            max_tokens=1024
        )

        # 初始化Embedding模型
        embed = DashScopeEmbeddings(
            model="text-embedding-v3"
        )

        logger.info("Qwen models initialized successfully")
        return llm, chat, embed

    except Exception as e:
        logger.exception("Failed to initialize Qwen models")
        raise RuntimeError(f"Model initialization failed: {str(e)}")