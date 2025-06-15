# demo.py
from knowledge_base.builders.pdf_processor import PDFProcessor
from knowledge_base.retrieval.vector_retriever import VectorRetriever

# 1. 处理PDF
processor = PDFProcessor("data/pdfs")
documents = processor.process_pdf("data/pdfs/混凝土规范.pdf")  # 准备测试PDF文件

# 2. 构建向量库
vector_db = VectorRetriever("data/vector_db")
vector_db.vector_db.from_documents(
    documents=documents,
    embedding=vector_db.embedding,
    persist_directory="data/vector_db"
)

# 3. 测试检索
results = vector_db.search("混凝土强度标准", top_k=2)
for i, doc in enumerate(results):
    print(f"结果 {i+1}: {doc.page_content[:50]}...")