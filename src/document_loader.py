from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".pptx"}


def load_documents(data_dir: str | Path) -> list[dict]:
    """Load supported knowledge base files from the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    documents = []
    for file_path in sorted(data_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = load_file_content(file_path)

        if content:
            documents.append(
                {
                    "source": str(file_path),
                    "content": content,
                }
            )

    return documents


def load_txt_documents(data_dir: str | Path) -> list[dict]:
    """Backward-compatible alias for older callers."""
    return load_documents(data_dir)


def load_file_content(file_path: Path) -> str:
    """Extract plain text from one supported knowledge base file."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return load_txt_content(file_path)
    if suffix == ".pdf":
        return load_pdf_content(file_path)
    if suffix == ".docx":
        return load_docx_content(file_path)
    if suffix == ".pptx":
        return load_pptx_content(file_path)

    return ""


def load_txt_content(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="gbk", errors="ignore").strip()


def load_pdf_content(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要先安装 pypdf，请运行 pip install -r requirements.txt。") from exc

    reader = PdfReader(str(file_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"第 {page_number} 页\n{text}")
    return "\n\n".join(pages).strip()


def load_docx_content(file_path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError("读取 Word 需要先安装 python-docx，请运行 pip install -r requirements.txt。") from exc

    document = DocxDocument(str(file_path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    tables = []

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                tables.append(" | ".join(cells))

    return "\n".join([text for text in paragraphs + tables if text]).strip()


def load_pptx_content(file_path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("读取 PPT 需要先安装 python-pptx，请运行 pip install -r requirements.txt。") from exc

    presentation = Presentation(str(file_path))
    slides = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())

            if not getattr(shape, "has_table", False):
                continue

            for row in shape.table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    texts.append(" | ".join(cells))

        if texts:
            slides.append(f"第 {slide_number} 页\n" + "\n".join(texts))

    return "\n\n".join(slides).strip()
