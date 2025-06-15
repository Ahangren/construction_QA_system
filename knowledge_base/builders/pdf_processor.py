import os
import logging
import time
from tqdm import tqdm
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# 修改导入路径为新的项目结构
from knowledge_base.storage.chroma_manager import ChromaManager

class PDFProcessor:
    """
    PDF文档处理管道，负责：
    - 从指定目录加载PDF文件
    - 提取文本内容
    - 分块处理文本
    - 将文本块存入向量数据库

    参数说明：
        directory: PDF文件所在目录路径
        chroma_server_type: ChromaDB服务器类型("local"或"http")
        persist_path: ChromaDB持久化存储路径(本地模式使用)
        embed: 文本嵌入模型实例
        file_group_num: 每组处理的文件数(默认80)
        batch_num: 每次插入的批次数量(默认6)
        chunksize: 文本分块大小(默认500字符)
        overlap: 分块重叠大小(默认100字符)
    """

    def __init__(self,
                 directory: str,
                 chroma_server_type: str = "local",
                 persist_path: str = "chroma_db",
                 embedding_function: Optional[object] = None,
                 file_group_num: int = 80,
                 batch_num: int = 6,
                 chunksize: int = 500,
                 overlap: int = 100):

        # 参数初始化
        self.directory = directory
        self.file_group_num = file_group_num
        self.batch_num = batch_num
        self.chunksize = chunksize
        self.overlap = overlap

        # 初始化ChromaDB连接（更新类名）
        self.chroma_db = ChromaManager(
            chroma_server_type=chroma_server_type,
            persist_path=persist_path,
            embedding_function=embedding_function
        )

        # 配置日志系统（日志文件路径调整为相对路径）
        self._setup_logging()

        # 验证目录存在
        if not os.path.isdir(self.directory):
            raise ValueError(f"指定目录不存在: {self.directory}")

    def _setup_logging(self):
        """配置日志系统"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "pdf_processor.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    def load_pdf_files(self) -> List[str]:
        """
        扫描目录并返回所有PDF文件路径

        返回:
            包含完整PDF文件路径的列表

        异常:
            ValueError: 如果目录中没有PDF文件
        """
        pdf_files = []
        for file in os.listdir(self.directory):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.directory, file))

        if not pdf_files:
            raise ValueError(f"目录中没有找到PDF文件: {self.directory}")

        self.logger.info(f"发现 {len(pdf_files)} 个PDF文件")
        return pdf_files

    def load_pdf_content(self, pdf_path: str) -> List[Document]:
        """
        使用PyMuPDF加载单个PDF文件内容

        参数:
            pdf_path: PDF文件路径

        返回:
            LangChain Document对象列表

        异常:
            RuntimeError: 如果文件加载失败
        """
        try:
            loader = PyMuPDFLoader(file_path=pdf_path)
            docs = loader.load()
            self.logger.debug(f"成功加载: {pdf_path} (共 {len(docs)} 页)")
            return docs
        except Exception as e:
            self.logger.error(f"加载PDF失败 {pdf_path}: {str(e)}")
            raise RuntimeError(f"无法加载PDF文件: {pdf_path}")

    def split_text(self, documents: List[Document]) -> List[Document]:
        """
        使用递归字符分割器将文档分块

        参数:
            documents: 待分割的Document列表

        返回:
            分割后的Document列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunksize,
            chunk_overlap=self.overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 中文友好分割符
        )

        try:
            docs = text_splitter.split_documents(documents)
            self.logger.info(f"文本分割完成: 原始 {len(documents)} 块 → 分割后 {len(docs)} 块")
            return docs
        except Exception as e:
            self.logger.error(f"文本分割失败: {str(e)}")
            raise RuntimeError("文本分割过程中发生错误")

    def insert_docs_chromadb(self, docs: List[Document], batch_size: int = 6) -> None:
        """
        将文档分批插入ChromaDB，带进度条和性能监控
        """
        if not docs:
            self.logger.warning("尝试插入空文档列表")
            return

        self.logger.info(f"开始插入 {len(docs)} 个文档到ChromaDB")
        start_time = time.time()
        total_docs_inserted = 0
        total_batches = (len(docs) + batch_size - 1) // batch_size

        try:
            with tqdm(total=total_batches, desc="插入进度", unit="batch") as pbar:
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]

                    # 更新方法调用（原add_with_langchain改为更标准的方法名）
                    self.chroma_db.add_documents(batch)
                    total_docs_inserted += len(batch)

                    # 计算吞吐量(每分钟处理文档数)
                    elapsed_time = time.time() - start_time
                    tpm = (total_docs_inserted / elapsed_time) * 60 if elapsed_time > 0 else 0

                    # 更新进度条
                    pbar.set_postfix({
                        "TPM": f"{tpm:.2f}",
                        "文档数": total_docs_inserted
                    })
                    pbar.update(1)

            self.logger.info(f"文档插入完成! 总耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            self.logger.error(f"文档插入失败: {str(e)}")
            raise RuntimeError(f"文档插入失败: {str(e)}")

    def process_pdfs_group(self, pdf_files_group: List[str]) -> None:
        """
        处理一组PDF文件(读取→分割→存储)

        参数:
            pdf_files_group: PDF文件路径列表
        """
        try:
            # 阶段1: 加载所有PDF内容
            pdf_contents = []
            for pdf_path in pdf_files_group:
                documents = self.load_pdf_content(pdf_path)
                pdf_contents.extend(documents)

            # 阶段2: 文本分割
            if pdf_contents:
                docs = self.split_text(pdf_contents)

                # 阶段3: 存储到向量数据库
                if docs:
                    self.insert_docs_chromadb(docs, self.batch_num)
        except Exception as e:
            self.logger.error(f"处理PDF组失败: {str(e)}")
            # 可以选择继续处理下一组而不是终止
            # raise

    def process_pdfs(self) -> None:
        """
        主处理流程: 扫描目录→分组处理所有PDF文件
        """
        self.logger.info("=== 开始PDF处理流程 ===")
        start_time = time.time()

        try:
            pdf_files = self.load_pdf_files()

            # 分组处理PDF文件
            for i in range(0, len(pdf_files), self.file_group_num):
                group = pdf_files[i:i + self.file_group_num]
                self.logger.info(
                    f"正在处理文件组 {i // self.file_group_num + 1}/{(len(pdf_files) - 1) // self.file_group_num + 1}")
                self.process_pdfs_group(group)

            self.logger.info(f"=== 处理完成! 总耗时: {time.time() - start_time:.2f}秒 ===")
            print("PDF处理成功完成!")
        except Exception as e:
            self.logger.error(f"PDF处理流程失败: {str(e)}")
            raise RuntimeError(f"PDF处理失败: {str(e)}")