import os
import pytest
from knowledge_base.builders.pdf_processor import PDFProcessor


@pytest.fixture
def test_resources(tmp_path):
    """测试资源准备"""
    # 创建PDF测试目录
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    # 复制预制PDF（或动态生成）
    test_pdf = os.path.join(os.path.dirname(__file__), "test_files", "sample.pdf")
    target_pdf = pdf_dir / "test.pdf"

    with open(test_pdf, "rb") as src, open(target_pdf, "wb") as dst:
        dst.write(src.read())

    return {
        "pdf_dir": str(pdf_dir),
        "db_dir": str(tmp_path / "chroma_db"),
        "pdf_path": str(target_pdf)
    }


def test_pdf_processing(test_resources):
    processor = PDFProcessor(
        directory=test_resources["pdf_dir"],
        persist_path=test_resources["db_dir"]
    )

    processor.process_pdfs()

    # 验证数据库
    assert os.path.exists(test_resources["db_dir"])
    assert any(os.listdir(test_resources["db_dir"]))